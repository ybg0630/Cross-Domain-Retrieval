"""Small synthetic fixtures only; never scientific result data."""

import json
import tempfile
import unittest
from pathlib import Path

from member5.adaptive import (choose_run, load_run, matched_random,
                              normalized_score_gap, overlap_at_10, ranked,
                              selected_queries, should_rerank, thresholds, validate_runs)
from src.adaptive import search


class PolicyTests(unittest.TestCase):
    def test_ties_and_signals(self):
        self.assertEqual(ranked({"b": 1, "a": 1, "z": 0}), ["a", "b", "z"])
        self.assertEqual(overlap_at_10({"a": 2, "b": 1}, {"b": 2, "a": 1}), 1)
        self.assertEqual(normalized_score_gap({"a": 1, "b": 1}), 0)
        self.assertAlmostEqual(normalized_score_gap({"a": 3, "b": 2, "c": 1}), .5)

    def test_extremes_and_query_selection(self):
        signals = {"q1": {"overlap": .1}, "q2": {"overlap": .9}}
        self.assertEqual(selected_queries(signals, "overlap", 0, "low"), set())
        self.assertEqual(selected_queries(signals, "overlap", 1, "low"), {"q1", "q2"})
        self.assertFalse(should_rerank(.5, .5, "low"))
        self.assertTrue(should_rerank(.9, .5, "high"))
        self.assertEqual(len(thresholds([.1, .9])), 3)
        h = {"q1": {"a": 1, "b": 0}, "q2": {"a": 1, "b": 0}}
        f = {"q1": {"b": 2, "a": 1}, "q2": {"b": 2, "a": 1}}
        self.assertEqual(choose_run(h, f, set()), h)
        self.assertEqual(choose_run(h, f, set(h)), f)
        self.assertEqual(choose_run(h, f, {"q1"})["q1"], f["q1"])
        self.assertEqual(choose_run(h, f, {"q1"})["q2"], h["q2"])

    def test_random_matching_and_schema(self):
        ids = [f"q{i}" for i in range(10)]
        self.assertEqual(matched_random(ids, 4, 42), matched_random(ids, 4, 42))
        self.assertEqual(len(matched_random(ids, 4, 42)), 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(json.dumps({"1": {"a": float("nan")}}))
            with self.assertRaises(ValueError): load_run(path)
        qrels = {"q": {"a": 1}}
        base = {"q": {"a": 2, "b": 1}}
        validate_runs(qrels, base, base, base, base, expected_depth=2)
        with self.assertRaises(ValueError):
            validate_runs(qrels, base, base, base, {"q": {"a": 1, "c": 2}}, expected_depth=2)
        with self.assertRaises(ValueError):
            validate_runs(qrels, base, base, base, {"other": base["q"]}, expected_depth=2)

    def test_live_gate_precedes_rerank(self):
        class FakeBackend:
            def __init__(self): self.calls = 0
            def retrieve(self, query, dataset, alpha):
                return {"bm25": {"a": 2, "b": 1}, "dense": {"c": 2, "d": 1},
                        "hybrid": {"a": .9, "c": .8},
                        "documents": {x: {"title": x, "abstract": x} for x in "abcd"}}
            def rerank(self, query, dataset, docs):
                self.calls += 1
                return {"c": 2, "a": 1}
        fake = FakeBackend()
        result = search("example", "scifact", "adaptive", .6, 0, backend=fake)
        self.assertFalse(result["reranked"])
        self.assertEqual(fake.calls, 0)
        result = search("example", "scifact", "adaptive", .6, .1, backend=fake)
        self.assertTrue(result["reranked"])
        self.assertEqual(fake.calls, 1)
        self.assertEqual(result["results"][0][0], "c")


if __name__ == "__main__":
    unittest.main()
