"""通过成员 1 的 evaluate 计算总体指标与逐查询变化，不另写指标公式。"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

if __package__:
    from .rerank import load_dataset, save_csv, save_json
else:
    from rerank import load_dataset, save_csv, save_json


def analyze(qrels, baseline, reranked, evaluate, ndcg_key="nDCG@10"):
    """evaluate(qrels, run) -> 指标字典；返回 (summary, per_query_rows)。"""
    if not qrels or set(qrels) != set(baseline) or set(qrels) != set(reranked):
        raise ValueError("qrels、baseline、reranked 必须覆盖完全相同的非空查询集合")
    for qid in qrels:
        if set(baseline[qid]) != set(reranked[qid]):
            raise ValueError(f"前后候选集合不同，不能作为纯重排序对照: {qid}")
    rows = []
    for qid in qrels:
        before = float(evaluate({qid: qrels[qid]}, {qid: baseline[qid]})[ndcg_key])
        after = float(evaluate({qid: qrels[qid]}, {qid: reranked[qid]})[ndcg_key])
        if not math.isfinite(before) or not math.isfinite(after):
            raise ValueError(f"评估函数返回了非有限指标: {qid}")
        delta = after - before
        status = "improved" if delta > 1e-9 else "worsened" if delta < -1e-9 else "unchanged"
        rows.append({"query_id": qid, "ndcg_before": before, "ndcg_after": after,
                     "delta": delta, "status": status})
    counts = {s: sum(row["status"] == s for row in rows)
              for s in ("improved", "worsened", "unchanged")}
    summary = {"baseline": evaluate(qrels, baseline), "reranked": evaluate(qrels, reranked),
               "counts": counts, "fractions": {s: n / len(rows) for s, n in counts.items()}}
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--reranked", type=Path, required=True)
    parser.add_argument("--utils", type=Path, required=True, help="成员 1 的 utils.py 路径")
    parser.add_argument("--ndcg-key", default="nDCG@10")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.utils.resolve().parent))
    spec = importlib.util.spec_from_file_location("member1_utils", args.utils)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _, _, qrels = load_dataset(args.data_dir, args.split)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    # 与 rerank 的 top-100 截取、同分规则保持一致。
    baseline = {q: dict(sorted(d.items(), key=lambda x: (-float(x[1]), x[0]))[:100])
                for q, d in baseline.items()}
    reranked = json.loads(args.reranked.read_text(encoding="utf-8"))
    summary, rows = analyze(qrels, baseline, reranked, module.evaluate, args.ndcg_key)
    save_json(str(args.output_prefix) + "_metrics.json", summary)
    save_csv(str(args.output_prefix) + "_delta.csv", rows)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist([row["delta"] for row in rows], bins=20, edgecolor="white")
        axes[0].set(xlabel="Change in nDCG@10", ylabel="Query count")
        axes[1].bar(summary["fractions"].keys(), summary["fractions"].values())
        axes[1].set(ylabel="Fraction of queries", ylim=(0, 1))
        fig.tight_layout()
        fig.savefig(str(args.output_prefix) + "_delta.png", dpi=160)
        plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
