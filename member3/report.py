"""Select a default fusion method and summarize the test ablation."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from member3.hybrid import load_run, rrf, validate_queries, weighted_sum
from src import utils

RESULTS = Path("member3/results")
REPORT = Path("member3/report")
DATASETS = {"scifact": ("SciFact", "train"), "nfcorpus": ("NFCorpus", "dev"), "fiqa": ("FiQA", "dev")}
METHODS = ("BM25", "MiniLM", "RRF", "Weighted sum")
METRICS = ("nDCG@10", "Recall@100")
COLORS = ("#738197", "#187f69", "#a64e20", "#3458b7")


def load_qrels(dataset, split):
    qrels = {}
    with Path(f"datasets/{dataset}/qrels/{split}.tsv").open(encoding="utf-8", newline="") as file:
        rows = csv.reader(file, delimiter="\t")
        next(rows)
        for query, doc, score in rows:
            qrels.setdefault(query, {})[doc] = int(score)
    return qrels


def write_csv(filename, rows):
    with (REPORT / filename).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    REPORT.mkdir(exist_ok=True)
    selection, test, weights = [], [], {}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    for ax, (dataset, (title, split)) in zip(axes, DATASETS.items(), strict=True):
        with (RESULTS / f"{dataset}_alpha_sweep.csv").open(encoding="utf-8", newline="") as file:
            sweep = [{k: float(row[k]) for k in ("alpha", *METRICS)}
                     for row in csv.DictReader(file) if row["split"] == split]
        if [row["alpha"] for row in sweep] != [i / 10 for i in range(11)]:
            raise ValueError(f"Incomplete alpha sweep for {dataset}")
        best = max(sweep, key=lambda row: (row["nDCG@10"], row["Recall@100"], row["alpha"]))
        weights[dataset] = alpha = best["alpha"]
        qrels = load_qrels(dataset, split)
        bm25 = load_run(RESULTS / split, f"{dataset}_bm25.json")
        dense = load_run(RESULTS / split, f"{dataset}_dense_minilm.json")
        validate_queries(bm25, dense, qrels)
        reference = utils.evaluate(qrels, {q: rrf(bm25[q], dense[q]) for q in qrels})
        weighted = utils.evaluate(qrels, {q: weighted_sum(bm25[q], dense[q], alpha) for q in qrels})
        for method, param, scores in (("RRF", "", reference), ("Weighted sum", alpha, weighted)):
            selection.append({"dataset": dataset, "split": split, "method": method, "alpha": param, **scores})

        ax.plot([r["alpha"] for r in sweep], [r["nDCG@10"] for r in sweep],
                color=COLORS[3], marker="o", markersize=4, label="Weighted sum")
        ax.axhline(reference["nDCG@10"], color=COLORS[2], ls="--", label="RRF (k=60)")
        ax.scatter(alpha, weighted["nDCG@10"], color="#111827", zorder=4, label=f"Selected alpha = {alpha:.1f}")
        ax.set(title=f"{title} ({split})", xlabel="MiniLM weight (alpha)", ylabel="nDCG@10",
               xticks=[0, .2, .4, .6, .8, 1])
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="y", alpha=.2)

        qrels = load_qrels(dataset, "test")
        bm25 = load_run(RESULTS, f"{dataset}_bm25.json")
        dense = load_run(Path("results"), f"{dataset}_dense_minilm.json")
        fused_rrf = load_run(RESULTS, f"{dataset}_hybrid_rrf.json")
        fused_wsum = load_run(RESULTS, f"{dataset}_hybrid_wsum.json")
        validate_queries(bm25, dense, qrels)
        validate_queries(fused_rrf, fused_wsum, qrels)
        if any(fused_rrf[q] != rrf(bm25[q], dense[q]) or len(fused_rrf[q]) != 100
               or fused_wsum[q] != weighted_sum(bm25[q], dense[q], alpha)
               or len(fused_wsum[q]) != 100 for q in qrels):
            raise ValueError(f"Saved test fusion does not match the selected settings for {dataset}")
        for method, run in zip(METHODS, (bm25, dense, fused_rrf, fused_wsum), strict=True):
            test.append({"dataset": dataset, "split": "test", "method": method, **utils.evaluate(qrels, run)})

    fig.suptitle("Fusion weight selection on train / dev")
    fig.savefig(REPORT / "alpha_ndcg.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    for ax, metric in zip(axes, METRICS, strict=True):
        for i, (method, color) in enumerate(zip(METHODS, COLORS, strict=True)):
            scores = [row[metric] for row in test if row["method"] == method]
            bars = ax.bar([j + (i - 1.5) * .2 for j in range(3)], scores, width=.19, color=color, label=method)
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        ax.set(title=metric, xticks=range(3), xticklabels=[title for title, _ in DATASETS.values()], ylim=(0, 1.04))
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Test ablation across retrieval methods")
    fig.savefig(REPORT / "test_ablation.png", dpi=220)
    plt.close(fig)

    write_csv("selection_metrics.csv", selection)
    write_csv("test_ablation.csv", test)
    averages = {method: sum(r["nDCG@10"] for r in selection if r["method"] == method) / len(DATASETS)
                for method in ("RRF", "Weighted sum")}
    winner = max(averages, key=averages.get)
    suffix = "wsum" if winner == "Weighted sum" else "rrf"
    settings = {
        "method": suffix, "dense_model": "minilm", "top_k": 100,
        "selection_metric": "macro average train/dev nDCG@10",
        "selection_scores": averages,
        "datasets": {
            ds: {"selection_split": split, **({"alpha": weights[ds]} if suffix == "wsum" else {"k": 60}),
                 "test_candidates": f"member3/results/{ds}_hybrid_{suffix}.json"}
            for ds, (_, split) in DATASETS.items()
        },
    }
    (REPORT / "default_fusion.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Member 3 fusion handoff", "",
        f"Use **{winner} with MiniLM** for the top 100 candidates per query.",
        f"Mean train/dev nDCG@10: RRF {averages['RRF']:.4f}, weighted sum {averages['Weighted sum']:.4f}.",
        "SciFact uses train. NFCorpus and FiQA use dev. Test labels are used for final reporting.", "",
        "| Dataset | Selection split | Parameter | Test candidates |", "|---|---|---|---|",
    ]
    for ds, item in settings["datasets"].items():
        file = Path(item["test_candidates"]).name
        param = f"alpha = {item['alpha']:.1f}" if suffix == "wsum" else "k = 60"
        lines.append(f"| {DATASETS[ds][0]} | {item['selection_split']} | {param} | [{file}](../results/{file}) |")
    lines.extend([
        "", "For weighted sum, alpha is the MiniLM weight. Min-max normalize each retriever's "
        "per-query top-100 scores before fusion. Missing scores contribute zero. At alpha 0 or 1, "
        "use the original single-retriever ranking.",
        "", "Sort candidate JSON scores in descending order and break ties by ascending document ID. "
        "For reranker calibration, apply the same parameters to the saved train/dev BM25 and MiniLM runs.",
        "", "See the [selection table](selection_metrics.csv), [test ablation table](test_ablation.csv), "
        "[alpha curve](alpha_ndcg.png), and [test ablation chart](test_ablation.png).",
        "", "Reproduce from the repository root with python -m member3.report (Matplotlib required).", "",
    ])
    (REPORT / "HANDOFF.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()
