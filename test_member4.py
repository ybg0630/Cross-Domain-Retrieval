"""成员 1 的计时器和成员 3 的校准候选接入检查。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from run_member4 import find_repo, get_candidates, team_measure


class IntegrationTests(unittest.TestCase):
    def test_team_timer_is_used_in_milliseconds(self):
        timer = Mock()
        timer.elapsed_ms = 12.5
        manager = Mock()
        manager.__enter__ = Mock(return_value=timer)
        manager.__exit__ = Mock(return_value=None)
        utils = SimpleNamespace(Timer=Mock(return_value=manager))
        value, elapsed = team_measure(utils)(lambda: "done")
        self.assertEqual((value, elapsed), ("done", 12.5))
        utils.Timer.assert_called_once()
        manager.__exit__.assert_called_once()

    def test_calibration_uses_frozen_alpha_and_upstream_fusion(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            directory = root / "member3/results/dev"
            directory.mkdir(parents=True)
            bm25, dense = {"q": {"d1": 2.0}}, {"q": {"d2": 0.8}}
            for name, run in (("nfcorpus_bm25", bm25), ("nfcorpus_dense_minilm", dense)):
                (directory / f"{name}.json").write_text(json.dumps(run), encoding="utf-8")
            settings = {"top_k": 100, "method": "wsum", "dense_model": "minilm",
                        "datasets": {"nfcorpus": {"selection_split": "dev", "alpha": 0.5}}}
            hybrid = SimpleNamespace(weighted_sum=Mock(return_value={"d2": 0.5, "d1": 0.4}))
            run, sources = get_candidates(root, "nfcorpus", "dev", settings, hybrid)
            hybrid.weighted_sum.assert_called_once_with(bm25["q"], dense["q"], 0.5)
            self.assertEqual(run, {"q": {"d2": 0.5, "d1": 0.4}})
            self.assertEqual(len(sources), 2)
            self.assertTrue(all(len(digest) == 64 for digest in sources.values()))
            with self.assertRaises(ValueError):
                get_candidates(root, "nfcorpus", "train", settings, hybrid)

    def test_missing_repository_is_reported(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            with self.assertRaises(FileNotFoundError):
                find_repo(folder)


if __name__ == "__main__":
    unittest.main()
