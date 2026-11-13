"""Fuse saved BM25 and dense retrieval results."""
import argparse
import csv
import json
from pathlib import Path


def rrf(bm25_scores, dense_scores, k=60, top_k=100):
    """Return the highest-scoring documents for one query."""
    fused = {}
    for scores in (bm25_scores, dense_scores):
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1 / (k + rank)
    ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return dict(ranked[:top_k])


def normalize(scores):
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if low == high:
        return dict.fromkeys(scores, 0.0)
    return {doc: (score - low) / (high - low) for doc, score in scores.items()}


def weighted_sum(bm25_scores, dense_scores, alpha, top_k=100):
    """Alpha is the weight on dense retrieval; an absent score contributes zero."""
    if alpha in (0, 1):
        scores = bm25_scores if alpha == 0 else dense_scores
    else:
        bm25 = normalize(bm25_scores)
        dense = normalize(dense_scores)
        scores = {
            doc: (1 - alpha) * bm25.get(doc, 0.0) + alpha * dense.get(doc, 0.0)
            for doc in bm25.keys() | dense.keys()
        }
    return dict(sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k])


def load_run(directory, filename):
    return json.loads((directory / filename).read_text(encoding="utf-8"))


def validate_queries(bm25, dense, qrels):
    if bm25.keys() != qrels.keys() or dense.keys() != qrels.keys():
        raise ValueError("BM25, dense and relevance labels must cover the same query IDs")


def tune_weight(dataset, model, results_dir):
    from src import utils

    split = "train" if dataset == "scifact" else "dev"
    tune_dir = results_dir / split
    bm25 = load_run(tune_dir, f"{dataset}_bm25.json")
    dense = load_run(tune_dir, f"{dataset}_dense_{model}.json")
    qrels = {}
    with Path(f"datasets/{dataset}/qrels/{split}.tsv").open(encoding="utf-8", newline="") as labels:
        rows = csv.reader(labels, delimiter="\t")
        next(rows)
        for query_id, doc_id, relevance in rows:
            qrels.setdefault(query_id, {})[doc_id] = int(relevance)
    validate_queries(bm25, dense, qrels)

    rows = []
    for step in range(11):
        alpha = step / 10
        run = {qid: weighted_sum(bm25[qid], dense[qid], alpha) for qid in qrels}
        rows.append({"split": split, "alpha": alpha, **utils.evaluate(qrels, run)})
    best = max(rows, key=lambda row: (row["nDCG@10"], row["Recall@100"], row["alpha"]))
    with (results_dir / f"{dataset}_alpha_sweep.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(best))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Selected on {split}: {best}")
    return best["alpha"]


def main():
    from src import utils

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["scifact", "nfcorpus", "fiqa"])
    parser.add_argument("--model", required=True, choices=["minilm", "bge"])
    parser.add_argument("--method", choices=["rrf", "wsum"], default="rrf")
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parent / "results"
    alpha = tune_weight(args.dataset, args.model, results_dir) if args.method == "wsum" else None
    bm25 = load_run(results_dir, f"{args.dataset}_bm25.json")
    dense = load_run(results_dir.parent.parent / "results", f"{args.dataset}_dense_{args.model}.json")
    _, _, qrels = utils.load_dataset(args.dataset)
    validate_queries(bm25, dense, qrels)

    if alpha is None:
        fused = {qid: rrf(bm25[qid], dense[qid]) for qid in sorted(qrels)}
    else:
        fused = {qid: weighted_sum(bm25[qid], dense[qid], alpha) for qid in sorted(qrels)}
    for name, run in (("BM25", bm25), (args.model, dense), (args.method.upper(), fused)):
        print(f"{name}: {utils.evaluate(qrels, run)}")

    output = results_dir / f"{args.dataset}_hybrid_{args.method}.json"
    output.write_text(json.dumps(fused, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output} ({len(fused)} queries)")


if __name__ == "__main__":
    main()
