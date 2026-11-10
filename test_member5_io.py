import unittest

from member5_io import choose_results, estimate_cost


class Member5InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.bundle = {"queries": {"q1": "one", "q2": "two"},
                       "hybrid": {"q1": {"a": 1.0}, "q2": {"b": 1.0}},
                       "reranked": {"q1": {"a": 9.0}, "q2": {"b": 8.0}},
                       "rerank_ms": {"q1": 100.0, "q2": 300.0}}

    def test_cost_is_amortized_over_all_queries(self):
        cost = estimate_cost(self.bundle, ["q1", "q1"])
        self.assertEqual(cost, {"rerank_fraction": 0.5, "mean_extra_rerank_ms": 50.0,
                                "total_rerank_ms": 100.0})

    def test_mixed_results_preserve_unselected_queries(self):
        run = choose_results(self.bundle, ["q2"])
        self.assertEqual(run, {"q1": {"a": 1.0}, "q2": {"b": 8.0}})
        run["q2"]["b"] = 0
        self.assertEqual(self.bundle["reranked"]["q2"]["b"], 8.0)

    def test_unknown_query_is_rejected(self):
        for operation in (choose_results, estimate_cost):
            with self.assertRaises(ValueError):
                operation(self.bundle, ["missing"])


if __name__ == "__main__":
    unittest.main()
