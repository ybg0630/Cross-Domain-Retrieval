"""运行：python -m unittest -v。只验证接口逻辑，不声称验证模型效果。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analyze import analyze
from demo import CANDIDATES, CORPUS, QUERIES, DemoScorer
from rerank import load_dataset, main, read_cached_scores, rerank


class RerankTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".test_", dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name) / "scores.sqlite3"

    def run_model(self, corpus=None, candidates=None, **kwargs):
        return rerank(corpus or CORPUS, QUERIES, candidates or CANDIDATES,
                      self.cache, kwargs.pop("scorer", DemoScorer()), dataset="demo",
                      measure=lambda op: (op(), 12.0), **kwargs)

    def test_ranking_cache_and_timing(self):
        run, rows, meta = self.run_model()
        self.assertEqual(list(run["q1"])[0], "d1")
        self.assertEqual(list(run["q2"])[0], "d3")
        self.assertEqual(meta["inferred_pairs"], 6)
        self.assertEqual(meta["mean_full_rerank_ms"], 12)
        self.assertEqual(meta["ms_per_inferred_pair"], 4)
        second, rows, meta = self.run_model()
        self.assertEqual(run, second)
        self.assertEqual(read_cached_scores(self.cache), run)
        self.assertEqual(meta["inferred_pairs"], 0)
        self.assertIsNone(meta["mean_full_rerank_ms"])
        self.assertIsNone(meta["ms_per_inferred_pair"])

    def test_changed_text_invalidates_only_affected_pairs(self):
        self.run_model()
        corpus = copy.deepcopy(CORPUS)
        corpus["d1"]["text"] += " Updated text."
        _, _, meta = self.run_model(corpus=corpus)
        self.assertEqual(meta["inferred_pairs"], 2)

    def test_new_candidate_only_scores_new_pair(self):
        candidates = copy.deepcopy(CANDIDATES)
        del candidates["q1"]["d2"]
        self.run_model(candidates=candidates)
        _, _, meta = self.run_model()
        self.assertEqual(meta["inferred_pairs"], 1)

    def test_refresh_runs_all_pairs(self):
        self.run_model()
        _, _, meta = self.run_model(refresh=True)
        self.assertTrue(meta["full_fresh_pass"])
        self.assertEqual(meta["inferred_pairs"], 6)

    def test_config_mismatch_rejected(self):
        self.run_model()
        scorer = DemoScorer()
        scorer.config = {"model": "different-model"}
        with self.assertRaisesRegex(ValueError, "配置不同"):
            self.run_model(scorer=scorer)

    def test_missing_query_or_doc_rejected(self):
        with self.assertRaises(ValueError):
            self.run_model(candidates={"q1": CANDIDATES["q1"]})
        candidates = copy.deepcopy(CANDIDATES)
        candidates["q2"]["nonexistent"] = 10
        with self.assertRaises(ValueError):
            self.run_model(candidates=candidates)

    def test_resume_after_failure(self):
        class FailingScorer(DemoScorer):
            calls = 0

            def predict(self, pairs, batch_size):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("simulated interruption")
                return super().predict(pairs, batch_size)

        with self.assertRaises(RuntimeError):
            self.run_model(scorer=FailingScorer())
        _, _, meta = self.run_model()
        self.assertEqual(meta["inferred_pairs"], 3)

    def test_invalid_prediction_not_cached(self):
        class BadScorer(DemoScorer):
            def predict(self, pairs, batch_size):
                return [float("nan")] * len(pairs)

        with self.assertRaises(ValueError):
            self.run_model(scorer=BadScorer())
        self.assertEqual(read_cached_scores(self.cache), {})

    def test_analysis_uses_injected_evaluator(self):
        run, _, _ = self.run_model()
        calls = []

        def evaluator(qrels, candidate_run):
            calls.append(set(qrels))
            # 固定接口桩：仅测试传参与差值，不实现 nDCG。
            return {"nDCG@10": 0.8 if candidate_run == {q: run[q] for q in qrels} else 0.3,
                    "Recall@100": 1.0}

        qrels = {"q1": {"d1": 1}, "q2": {"d3": 1}}
        summary, rows = analyze(qrels, CANDIDATES, run, evaluator)
        self.assertEqual(len(calls), 6)
        self.assertEqual(summary["counts"]["improved"], 2)
        self.assertAlmostEqual(rows[0]["delta"], 0.5)
        with self.assertRaises(ValueError):
            analyze(qrels, {"q1": {"d1": 1}, "q2": {"d3": 1}}, run, evaluator)

    def test_local_beir_loading_and_cli_outputs(self):
        root = Path(self.temp.name)
        (root / "qrels").mkdir()
        (root / "corpus.jsonl").write_text("\n".join(
            json.dumps({"_id": did, **doc}) for did, doc in CORPUS.items()), encoding="utf-8")
        (root / "queries.jsonl").write_text("\n".join(
            json.dumps({"_id": qid, "text": text})
            for qid, text in {**QUERIES, "train_only": "Other query"}.items()), encoding="utf-8")
        (root / "qrels" / "test.tsv").write_text(
            "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td3\t1\n", encoding="utf-8")
        (root / "candidates.json").write_text(json.dumps(CANDIDATES), encoding="utf-8")
        corpus, queries, qrels = load_dataset(root)
        self.assertEqual(queries, QUERIES)
        self.assertEqual(corpus, CORPUS)
        self.assertEqual(qrels, {"q1": {"d1": 1}, "q2": {"d3": 1}})
        argv = ["rerank.py", "--dataset", "scifact", "--data-dir", str(root),
                "--candidates", str(root / "candidates.json"), "--output-dir", str(root)]
        with patch("sys.argv", argv), patch("rerank.CrossEncoderScorer", return_value=DemoScorer()), \
                patch("builtins.print"):
            main()
        run = json.loads((root / "results" / "scifact_hybrid_rerank.json").read_text())
        self.assertEqual(list(run["q1"])[0], "d1")
        self.assertTrue((root / "reports" / "scifact_timing.csv").exists())
        meta = json.loads((root / "reports" / "scifact_timing.json").read_text())
        self.assertEqual(meta["inferred_pairs"], 6)


if __name__ == "__main__":
    unittest.main()
