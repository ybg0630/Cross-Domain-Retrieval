"""Create final S5 presentation figures from frozen held-out test summaries.

Run from repository root:
    python member5/plot_final_summary.py

Reads member5/outputs/combined_test_summary.csv and writes only member5/figures/.
No thresholds, metrics, or model outputs are recalculated or changed.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless output; works in VSCode terminal
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "combined_test_summary.csv"
OUT = ROOT / "figures"
DATASETS = ("scifact", "nfcorpus", "fiqa")
DISPLAY = ("SciFact", "NFCorpus", "FiQA")
METHODS = ("No rerank", "Full rerank", "Adaptive rerank", "Random rerank")


def load_results():
    if not DATA.exists():
        raise FileNotFoundError(f"Missing S5 results: {DATA}")
    with DATA.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keyed = {(r["dataset"], r["method"]): r for r in rows}
    for ds in DATASETS:
        for method in METHODS:
            if (ds, method) not in keyed:
                raise ValueError(f"Missing row for {ds} / {method}")
    return keyed


def measure(rows, ds, method, key):
    return float(rows[(ds, method)][key])


def plot_ndcg_gain(rows):
    """One chart, three datasets; compare gains/losses against Hybrid."""
    methods = ("Full rerank", "Adaptive rerank", "Random rerank")
    fig, ax = plt.subplots(figsize=(9.1, 4.8))
    width = 0.24
    xs = list(range(len(DATASETS)))
    for j, method in enumerate(methods):
        heights = [100 * (measure(rows, ds, method, "ndcg10")
                          - measure(rows, ds, "No rerank", "ndcg10"))
                   for ds in DATASETS]
        positions = [x + (j - 1) * width for x in xs]
        bars = ax.bar(positions, heights, width=width, label=(
            "Random (20-seed mean)" if method == "Random rerank" else method))
        for bar, value in zip(bars, heights):
            ax.annotate(f"{value:+.2f}",
                        (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 5 if value >= 0 else -5),
                        textcoords="offset points", ha="center",
                        va="bottom" if value >= 0 else "top", fontsize=10)
    ax.axhline(0, linewidth=1.0)
    ax.set_xticks(xs, DISPLAY)
    ax.set_ylabel("Change in nDCG@10 vs Hybrid (×100)")
    ax.set_xlabel("Dataset")
    ax.set_title("Does reranking improve the ranking?", pad=12)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.set_ylim(-1.35, 2.55)
    fig.text(.08, .025,
             "Zero = unchanged from Hybrid. Adaptive policy fixed on calibration data. "
             "Random bars are means (20 seeds); see CSV for SD.",
             fontsize=8.1, alpha=0.8)
    fig.subplots_adjust(left=.10, right=.98, top=.88, bottom=.20)
    path = OUT / "s5_ndcg_gain_by_dataset.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_rerank_share(rows):
    """The Adaptive method's selection budget; no quality claim made here."""
    shares = [100 * measure(rows, ds, "Adaptive rerank", "rerank_ratio")
              for ds in DATASETS]
    saved_extra = []
    for ds in DATASETS:
        full = measure(rows, ds, "Full rerank", "mean_extra_rerank_ms")
        adaptive = measure(rows, ds, "Adaptive rerank", "mean_extra_rerank_ms")
        saved_extra.append(100 * (1 - adaptive / full))
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    bars = ax.bar(DISPLAY, shares, width=0.53)
    for b, share in zip(bars, shares):
        ax.text(b.get_x() + b.get_width()/2, share + 2.1,
                f"{share:.1f}%", ha="center", va="bottom", fontsize=13)
    ax.axhline(100, linestyle="--", linewidth=1.0, label="Full rerank = 100%")
    ax.set_ylabel("Queries sent to Cross-Encoder (%)")
    ax.set_xlabel("Dataset")
    ax.set_ylim(0, 112)
    ax.set_title("How often does Adaptive rerank?", pad=12)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    fig.text(.08, .035,
             "Extra Cross-Encoder time saved vs Full: "
             + "  |  ".join(f"{name} {pct:.1f}%" for name, pct in zip(DISPLAY, saved_extra))
             + ". Not end-to-end latency.",
             fontsize=8.3, alpha=0.8)
    fig.subplots_adjust(left=.11, right=.97, top=.88, bottom=.21)
    path = OUT / "s5_rerank_share.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_result_table(rows):
    """Show the final test nDCG@10 as a full 4-column table (no empty corner)."""
    labels = ("No rerank (Hybrid)", "Full rerank", "Adaptive (frozen)",
              "Random (20-seed mean)")
    table_rows = [["Method", *DISPLAY]]
    table_rows += [
        [label, *(f"{measure(rows, ds, method, 'ndcg10'):.4f}" for ds in DATASETS)]
        for label, method in zip(labels, METHODS)
    ]

    fig, ax = plt.subplots(figsize=(9.4, 3.5))
    ax.axis("off")
    # An explicit first column creates a proper "Method" header cell and
    # avoids the open top-left corner produced by rowLabels + colLabels.
    table = ax.table(cellText=table_rows, cellLoc="center",
                     colWidths=[0.40, 0.20, 0.20, 0.20],
                     bbox=[0.02, 0.11, 0.96, 0.77])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.7)
        if row == 0:
            cell.get_text().set_weight("bold")
        if row == 3:  # Adaptive row
            cell.get_text().set_weight("bold")
        if col == 0:
            cell.get_text().set_ha("left")
            cell.PAD = 0.06
    ax.set_title("Held-out test nDCG@10", fontsize=14, pad=10)
    fig.text(.03, .034,
             "Adaptive policy fixed on calibration data; Random is the mean of 20 matched-budget seeds.",
             fontsize=8.5, alpha=.8)
    fig.subplots_adjust(left=.04, right=.96, top=.89, bottom=.11)
    path = OUT / "s5_ndcg_summary_table.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_results()
    for path in (plot_ndcg_gain(rows), plot_rerank_share(rows), plot_result_table(rows)):
        print(f"Created {path}")


if __name__ == "__main__":
    main()
