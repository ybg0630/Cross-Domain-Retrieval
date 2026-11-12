"""Run S5's calibration-frozen offline counterfactual experiments.

From repository root: PYTHONPATH=member5/vendor python -m member5.run_experiments
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from member5.adaptive import (choose_run, load_labels, load_run, matched_random,
                              selected_queries, signals_for_run, thresholds, validate_runs)
from member5.evaluator import team_evaluator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "member5" / "inputs" / "s4_handoff_20261110"
CALIBRATION = {"scifact": "train", "nfcorpus": "dev", "fiqa": "dev"}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_bundle(root, dataset, split, evaluate):
    """Read S4's full fresh handoff and validate exact S3 candidate provenance."""
    stem = dataset if split == "test" else f"{dataset}_{split}"
    base = root / "inputs" / dataset / split
    paths = {"bm25": base / "bm25.json", "dense": base / "dense.json",
             "qrels": base / "qrels.json", "queries": base / "queries.json",
             "hybrid": root / "results" / f"{stem}_rerank_candidates.json",
             "full": root / "results" / f"{stem}_hybrid_rerank.json",
             "timing": root / "reports" / "member4" / f"{stem}_timing.csv",
             "metadata": root / "reports" / "member4" / f"{stem}_timing.json"}
    expected_hashes = {relative: digest for digest, relative in
                       (line.split("  ", 1) for line in
                        (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines())}
    for name, path in paths.items():
        relative = path.relative_to(root).as_posix()
        if relative not in expected_hashes or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hashes[relative]:
            raise ValueError(f"Missing or altered S4 handoff file: {relative}")
    runs = {name: load_run(paths[name]) for name in ("bm25", "dense", "hybrid", "full")}
    qrels = load_labels(paths["qrels"])
    queries = {str(q): str(v) for q, v in json.loads(paths["queries"].read_text(encoding="utf-8")).items()}
    validate_runs(qrels, **runs)
    if set(queries) != set(qrels):
        raise ValueError("Query text/qrels IDs differ")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if metadata["smoke_only"] or not metadata["full_fresh_pass"] or metadata["split"] != split:
        raise ValueError("S4 timing is not a full fresh pass for this split")
    times = {}
    with paths["timing"].open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["candidate_pairs"]) != 100 or int(row["inferred_pairs"]) != 100 or int(row["cached_pairs"]) != 0:
                raise ValueError("S4 timing is not a 100-pair fresh inference")
            times[row["query_id"]] = float(row["inference_ms"])
    if set(times) != set(qrels) or any(not math.isfinite(x) or x < 0 for x in times.values()):
        raise ValueError("Timing query IDs or values invalid")
    if not math.isclose(statistics.mean(times.values()), metadata["mean_full_rerank_ms"], abs_tol=1e-8):
        raise ValueError("S4 timing summary mismatch")
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    item = next(x for x in manifest["splits"] if x["dataset"] == dataset and x["split"] == split)
    if len(qrels) != item["queries"]:
        raise ValueError("Manifest query count mismatch")
    for name, key in (("hybrid", "baseline"), ("full", "reranked")):
        actual = evaluate(qrels, runs[name])
        if any(not math.isclose(actual[metric], item[key][metric], abs_tol=1e-10)
               for metric in ("nDCG@10", "Recall@100")):
            raise ValueError(f"Team evaluator and S4 {name} metrics differ")
    if split == "test":
        s3 = load_run(ROOT / "member3" / "results" / f"{dataset}_hybrid_wsum.json")
        if s3 != runs["hybrid"]:
            raise ValueError("S4 candidates differ from S3 canonical weighted-sum run")
        upstream_bm25 = ROOT / "member3" / "results" / f"{dataset}_bm25.json"
        upstream_dense = ROOT / "results" / f"{dataset}_dense_minilm.json"
    else:
        upstream_bm25 = ROOT / "member3" / "results" / split / f"{dataset}_bm25.json"
        upstream_dense = ROOT / "member3" / "results" / split / f"{dataset}_dense_minilm.json"
    if load_run(upstream_bm25) != runs["bm25"] or load_run(upstream_dense) != runs["dense"]:
        raise ValueError("S4 signal inputs differ from canonical S3 BM25/MiniLM runs")
    return {**runs, "qrels": qrels, "queries": queries, "times": times,
            "metadata": metadata, "paths": {k: str(v) for k, v in paths.items()},
            "manifest": item}


def per_query_delta(bundle, evaluate):
    """Use S1's evaluator on singleton queries; verify its aggregate average."""
    rows = []
    for q in sorted(bundle["qrels"]):
        before = evaluate({q: bundle["qrels"][q]}, {q: bundle["hybrid"][q]})["nDCG@10"]
        after = evaluate({q: bundle["qrels"][q]}, {q: bundle["full"][q]})["nDCG@10"]
        delta = after - before
        rows.append({"query_id": q, "hybrid_ndcg10": before, "full_ndcg10": after,
                     "delta_ndcg10": delta, "status": "improved" if delta > 1e-12 else
                     "worsened" if delta < -1e-12 else "unchanged"})
    for key, field in (("hybrid", "hybrid_ndcg10"), ("full", "full_ndcg10")):
        if not math.isclose(statistics.mean(row[field] for row in rows),
                            evaluate(bundle["qrels"], bundle[key])["nDCG@10"], abs_tol=1e-10):
            raise ValueError("Singleton/team aggregate mismatch")
    return rows


def sweep(bundle, signals, delta, names, budget):
    """Explore every signal/direction on calibration labels at a fixed maximum budget."""
    baseline = statistics.mean(row["hybrid_ndcg10"] for row in delta)
    gain = {row["query_id"]: row["delta_ndcg10"] for row in delta}
    rows = []
    for name in names:
        values = [v[name] for v in signals.values()]
        for direction in ("low", "high"):
            for threshold in thresholds(values):
                selected = selected_queries(signals, name, threshold, direction)
                ratio = len(selected) / len(signals)
                rows.append({"signal": name, "direction": direction, "threshold": threshold,
                             "selected_queries": len(selected), "rerank_ratio": ratio,
                             "ndcg10": baseline + sum(gain[q] for q in selected) / len(signals),
                             "mean_extra_rerank_ms": sum(bundle["times"][q] for q in selected) / len(signals),
                             "within_budget": ratio <= budget + 1e-12})
    return rows


def choose_policy(rows, budget):
    """Freeze best calibration nDCG under the budget; favor less reranking on ties."""
    eligible = [r for r in rows if r["rerank_ratio"] <= budget + 1e-12]
    return sorted(eligible, key=lambda r: (-round(r["ndcg10"], 12), r["rerank_ratio"],
                                          r["signal"], r["direction"], r["threshold"]))[0]


def summarize(bundle, signals, policy, evaluate, seeds):
    """Evaluate all four strategies with S1 and matched-count random controls."""
    selected = selected_queries(signals, policy["signal"], policy["threshold"], policy["direction"])
    n = len(signals)
    def metrics(name, ids):
        run = choose_run(bundle["hybrid"], bundle["full"], ids)
        value = evaluate(bundle["qrels"], run)
        if not math.isclose(value["Recall@100"], evaluate(bundle["qrels"], bundle["hybrid"])["Recall@100"], abs_tol=1e-12):
            raise ValueError("Recall@100 changed despite fixed candidate sets")
        return {"method": name, "selected_queries": len(ids), "rerank_ratio": len(ids)/n,
                "ndcg10": value["nDCG@10"], "recall100": value["Recall@100"],
                "mean_extra_rerank_ms": sum(bundle["times"][q] for q in ids)/n}
    rows = [metrics("No rerank", set()), metrics("Full rerank", set(signals)),
            metrics("Adaptive rerank", selected)]
    random_rows = [metrics("Random rerank", matched_random(signals, len(selected), seed))
                   for seed in range(42, 42 + seeds)]
    rows.append({"method": "Random rerank", "selected_queries": len(selected),
                 "rerank_ratio": len(selected)/n,
                 "ndcg10": statistics.mean(x["ndcg10"] for x in random_rows),
                 "recall100": random_rows[0]["recall100"],
                 "mean_extra_rerank_ms": statistics.mean(x["mean_extra_rerank_ms"] for x in random_rows),
                 "ndcg10_sd": statistics.stdev(x["ndcg10"] for x in random_rows) if seeds > 1 else 0,
                 "extra_ms_sd": statistics.stdev(x["mean_extra_rerank_ms"] for x in random_rows) if seeds > 1 else 0})
    for row in rows[:3]:
        row["ndcg10_sd"] = 0.0
        row["extra_ms_sd"] = 0.0
    return rows, random_rows


def plot_diagnostics(path, signals, delta, signal):
    gain = {r["query_id"]: r["delta_ndcg10"] for r in delta}
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter([v[signal] for v in signals.values()], [gain[q] for q in signals], s=8, alpha=.35)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(xlabel=signal, ylabel="Full minus Hybrid per-query nDCG@10",
           title="Signal versus rerank benefit (real queries)")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_curve(path, sweep_rows, policy, x, title):
    rows = [r for r in sweep_rows if r["signal"] == policy["signal"] and r["direction"] == policy["direction"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter([r[x] for r in rows], [r["ndcg10"] for r in rows], s=9, alpha=.55)
    ax.set(xlabel="Reranked query fraction" if x == "rerank_ratio" else "Mean additional rerank inference (ms/query)",
           ylabel="nDCG@10", title=title)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["scifact", "nfcorpus", "fiqa", "all"], default="all")
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "member5" / "outputs")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "member5" / "figures")
    parser.add_argument("--signal", choices=["auto", "overlap", "gap", "query_length"], default="auto")
    parser.add_argument("--budget", type=float, default=.5)
    parser.add_argument("--random-seeds", type=int, default=20)
    args = parser.parse_args()
    if not 0 <= args.budget <= 1 or args.random_seeds < 1:
        parser.error("budget must be in [0,1] and random-seeds positive")
    evaluate, evaluator_sha = team_evaluator()
    datasets = CALIBRATION if args.dataset == "all" else {args.dataset: CALIBRATION[args.dataset]}
    combined = []
    for dataset, calibration in datasets.items():
        bundles = {split: load_bundle(args.bundle_root, dataset, split, evaluate)
                   for split in (calibration, "test")}
        signals = {split: signals_for_run(b["bm25"], b["dense"], b["hybrid"], b["queries"])
                   for split, b in bundles.items()}
        deltas = {split: per_query_delta(b, evaluate) for split, b in bundles.items()}
        names = ("overlap", "gap", "query_length") if args.signal == "auto" else (args.signal,)
        calibration_sweep = sweep(bundles[calibration], signals[calibration], deltas[calibration], names, args.budget)
        policy = choose_policy(calibration_sweep, args.budget)
        dataset_out = args.output_dir / dataset
        dataset_out.mkdir(parents=True, exist_ok=True)
        args.figure_dir.mkdir(parents=True, exist_ok=True)
        write_csv(dataset_out / "threshold_sweep.csv", calibration_sweep)
        for split in (calibration, "test"):
            b = bundles[split]
            sig = signals[split]
            delta = deltas[split]
            write_csv(dataset_out / f"{split}_per_query.csv", [
                {**row, **sig[row["query_id"]], "rerank_ms": b["times"][row["query_id"]]}
                for row in delta])
            rows, random_rows = summarize(b, sig, policy, evaluate, args.random_seeds)
            write_csv(dataset_out / f"{split}_summary.csv", [{"dataset": dataset, "split": split, "queries": len(sig), **row} for row in rows])
            write_csv(dataset_out / f"{split}_random_seeds.csv", [{"seed": seed, **row} for seed, row in zip(range(42,42+args.random_seeds), random_rows)])
            exploratory = sweep(b, sig, delta, (policy["signal"],), 1.0)
            if split == "test":
                write_csv(dataset_out / "test_curve_exploratory_only.csv", exploratory)
                plot_diagnostics(args.figure_dir / f"{dataset}_signal_vs_delta.png", sig, delta, policy["signal"])
                plot_curve(args.figure_dir / f"{dataset}_quality_vs_ratio.png", exploratory, policy,
                           "rerank_ratio", "Test sweep: exploratory only; selected policy frozen on calibration")
                plot_curve(args.figure_dir / f"{dataset}_quality_vs_extra_latency.png", exploratory, policy,
                           "mean_extra_rerank_ms", "Test sweep: exploratory only; extra S4 GPU inference")
            metadata = {"dataset": dataset, "split": split, "queries": len(sig),
                        "policy": {k: policy[k] for k in ("signal", "direction", "threshold", "rerank_ratio", "ndcg10")},
                        "selection_split": calibration, "max_budget": args.budget,
                        "selection_rule": "maximum calibration nDCG@10 within rerank fraction budget; tie: lower fraction, signal, direction, threshold",
                        "random_seeds": list(range(42, 42+args.random_seeds)),
                        "evaluator": "unchanged src.utils.evaluate AST", "evaluator_sha256": evaluator_sha,
                        "s4_source": "origin/s4:deliverables/member5_handoff_20261110.zip",
                        "paths": b["paths"], "hardware": b["metadata"]["hardware"],
                        "latency_scope": b["metadata"]["timing_scope"],
                        "latency_warning": "Measured S4 extra inference only; Hybrid retrieval and signal overhead unavailable on same hardware; not total search latency.",
                        "test_sweep_warning": "Exploratory only; never used to choose the reported test policy."}
            (dataset_out / f"{split}_provenance.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
            if split == "test":
                combined.extend({"dataset": dataset, "queries": len(sig), "selected_signal": policy["signal"],
                                 "direction": policy["direction"], "threshold": policy["threshold"], **row}
                                for row in rows)
            print(f"{dataset}/{split}: {len(sig)} queries, {policy['signal']} {policy['direction']} {policy['threshold']:.6g}, "
                  f"adaptive nDCG={rows[2]['ndcg10']:.6f}, selected={rows[2]['selected_queries']}")
    write_csv(args.output_dir / "combined_test_summary.csv", combined)


if __name__ == "__main__":
    main()
