"""无需上游的接口演示；示例数据不是三个正式数据集的实验结果。"""

import argparse
import json
from pathlib import Path

if __package__:
    from .rerank import CrossEncoderScorer, rerank, save_csv, save_json
else:
    from rerank import CrossEncoderScorer, rerank, save_csv, save_json

CORPUS = {
    "d1": {"title": "Bonds", "text": "Bond prices usually fall when interest rates rise."},
    "d2": {"title": "Nutrition", "text": "Oats contain soluble fiber that can lower cholesterol."},
    "d3": {"title": "Astronomy", "text": "Mars is known as the red planet."},
}
QUERIES = {"q1": "What happens to bond prices when interest rates rise?",
           "q2": "Which planet is known as the red planet?"}
CANDIDATES = {"q1": {"d3": 3, "d2": 2, "d1": 1},
              "q2": {"d1": 3, "d2": 2, "d3": 1}}


class DemoScorer:
    """仅验证程序流程的固定假分数；不是检索算法或正式评估。"""
    config = {"model": "DEMO_FAKE_SCORES"}
    hardware = {"device": "demo-no-model"}

    def prepare(self, example):
        pass

    def predict(self, pairs, batch_size):
        return [9.0 if (("bond" in q and "Bond" in d) or
                        ("planet" in q and "Mars" in d)) else -1.0 for q, d in pairs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-model", action="store_true", help="下载并使用真实 Cross-Encoder")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("demo_output"))
    args = parser.parse_args()
    scorer = CrossEncoderScorer(device=args.device) if args.real_model else DemoScorer()
    kind = "real_model" if args.real_model else "fake_scores"
    out = args.output_dir / kind
    run, rows, meta = rerank(CORPUS, QUERIES, CANDIDATES, out / "cache.sqlite3",
                             scorer, dataset="demo", batch_size=2)
    meta["demo_only"] = True
    save_json(out / "run.json", run)
    save_csv(out / "timing.csv", rows)
    save_json(out / "metadata.json", meta)
    print(json.dumps({"demo_only": True, "output": str(out), "run": run,
                      "inferred_pairs": meta["inferred_pairs"]}, indent=2))


if __name__ == "__main__":
    main()
