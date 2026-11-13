"""Pure query-level policy functions for saved retrieval runs."""

import csv
import json
import math
import random
from pathlib import Path


def ranked(scores):
    """Sort high scores first, breaking ties by ascending string document ID."""
    return sorted(scores, key=lambda doc: (-scores[doc], doc))


def load_run(path):
    """Load a finite-score run and canonicalize all query/document IDs to strings."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Empty or invalid run: {path}")
    result = {}
    for qid, docs in raw.items():
        qid = str(qid)
        if qid in result or not isinstance(docs, dict) or not docs:
            raise ValueError(f"Duplicate query or empty document list: {qid}")
        result[qid] = {}
        for did, value in docs.items():
            did = str(did)
            if did in result[qid] or isinstance(value, bool):
                raise ValueError(f"Duplicate document or invalid score: {qid}/{did}")
            try:
                score = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Non-numeric score: {qid}/{did}") from exc
            if not math.isfinite(score):
                raise ValueError(f"Non-finite score: {qid}/{did}")
            result[qid][did] = score
    return result


def load_labels(path):
    """Read S4's BEIR qrels JSON or an original BEIR TSV."""
    path = Path(path)
    result = {}
    if path.suffix == ".tsv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                qid = str(row.get("query-id", row.get("query_id", "")))
                did = str(row.get("corpus-id", row.get("doc_id", "")))
                result.setdefault(qid, {})[did] = int(row.get("score", row.get("relevance", "")))
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        result = {str(q): {str(d): int(v) for d, v in docs.items()} for q, docs in raw.items()}
    if not result or any(not docs for docs in result.values()):
        raise ValueError("Empty relevance labels")
    return result


def validate_runs(qrels, bm25, dense, hybrid, full, expected_depth=100):
    """Require complete matching query IDs, depth and reranker candidate sets."""
    expected = set(qrels)
    if not expected:
        raise ValueError("No query IDs")
    for name, run in (("bm25", bm25), ("dense", dense), ("hybrid", hybrid), ("full", full)):
        if set(run) != expected:
            raise ValueError(f"{name} query IDs differ: missing={len(expected-set(run))}, extra={len(set(run)-expected)}")
        if any(len(docs) != expected_depth for docs in run.values()):
            raise ValueError(f"{name} does not have {expected_depth} candidates per query")
    for qid in expected:
        if set(hybrid[qid]) != set(full[qid]):
            raise ValueError(f"Hybrid/full candidate mismatch for {qid}")


def overlap_at_10(bm25, dense):
    """Fraction of shared top-10 IDs; short lists use available minimum depth."""
    n = min(10, len(bm25), len(dense))
    if not n:
        raise ValueError("Overlap requires two nonempty lists")
    return len(set(ranked(bm25)[:n]) & set(ranked(dense)[:n])) / n


def normalized_score_gap(hybrid):
    """Top-1 minus top-2 score divided by score range; heuristic, not probability."""
    if len(hybrid) < 2:
        raise ValueError("Gap requires at least two candidates")
    scores = [hybrid[d] for d in ranked(hybrid)]
    span = scores[0] - scores[-1]
    return (scores[0] - scores[1]) / span if span > 0 else 0.0


def signals_for_run(bm25, dense, hybrid, queries=None):
    """Extract inexpensive signals before any cross-encoder scores are consulted."""
    if set(bm25) != set(dense) or set(bm25) != set(hybrid):
        raise ValueError("Signal input query IDs differ")
    if queries is not None and set(queries) != set(hybrid):
        raise ValueError("Query text IDs differ")
    return {q: {"overlap": overlap_at_10(bm25[q], dense[q]),
                "gap": normalized_score_gap(hybrid[q]),
                **({"query_length": len(queries[q].split())} if queries is not None else {})}
            for q in sorted(hybrid)}


def should_rerank(signal, threshold, direction):
    """Make one whole-query decision; equality does not trigger reranking."""
    if not math.isfinite(float(signal)) or not math.isfinite(float(threshold)):
        raise ValueError("Signal and threshold must be finite")
    if direction == "low":
        return signal < threshold
    if direction == "high":
        return signal > threshold
    raise ValueError("direction must be low or high")


def selected_queries(signals, name, threshold, direction):
    """Return query IDs selected by a frozen policy."""
    return {q for q, values in signals.items() if should_rerank(values[name], threshold, direction)}


def choose_run(hybrid, full, selected):
    """Choose the entire hybrid or full rerank ranking per query."""
    selected = set(selected)
    if not selected <= hybrid.keys() or set(hybrid) != set(full):
        raise ValueError("Selection or run query IDs differ")
    for q in selected:
        if set(hybrid[q]) != set(full[q]):
            raise ValueError(f"Candidate mismatch for selected query {q}")
    return {q: dict(full[q] if q in selected else hybrid[q]) for q in sorted(hybrid)}


def matched_random(query_ids, count, seed):
    """Sample exactly count distinct queries using a local seeded generator."""
    ids = sorted(query_ids)
    if count < 0 or count > len(ids):
        raise ValueError("Invalid random query count")
    return set(random.Random(seed).sample(ids, count))


def thresholds(values):
    """Include the zero/full extremes and all distinct decision boundaries."""
    unique = sorted(set(values))
    if not unique:
        raise ValueError("No signal values")
    return [math.nextafter(unique[0], -math.inf)] + [
        (a + b) / 2 for a, b in zip(unique, unique[1:])
    ] + [math.nextafter(unique[-1], math.inf)]
