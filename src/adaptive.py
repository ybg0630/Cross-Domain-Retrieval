"""S5 live-search boundary for S6's five-argument Streamlit call.

Install a dataset-aware backend once with configure_backend(), or pass it
explicitly. Supply a calibrated policy to override the overlap control.
"""

import time

from member5.adaptive import (normalized_score_gap, overlap_at_10, ranked,
                              should_rerank)

_backend = None


def configure_backend(backend):
    """Register an object with retrieve(query,dataset,alpha) and rerank(query,dataset,docs)."""
    global _backend
    if not callable(getattr(backend, "retrieve", None)) or not callable(getattr(backend, "rerank", None)):
        raise TypeError("Backend needs retrieve and rerank methods")
    _backend = backend


def search(query, dataset, mode, alpha, threshold, *, backend=None, policy=None):
    """Retrieve first, compute signals, then conditionally invoke CrossEncoder.

    retrieve returns bm25/dense/hybrid score maps and documents mapping IDs to
    {title, abstract}; rerank returns scores for exactly the supplied IDs.
    The UI threshold controls overlap< threshold in adaptive mode.
    """
    engine = backend or _backend
    if engine is None:
        raise RuntimeError("S5 live backend not configured: BM25 index, MiniLM document embeddings, corpus text and S4 scorer are required")
    if mode not in {"bm25", "dense", "hybrid", "hybrid_rerank", "adaptive"}:
        raise ValueError(f"Unknown mode: {mode}")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be nonempty text")
    if dataset not in {"scifact", "nfcorpus", "fiqa"} or not 0 <= float(alpha) <= 1:
        raise ValueError("Unknown dataset or alpha outside [0,1]")
    start = time.perf_counter()
    signals = {}
    reranked = mode == "hybrid_rerank"
    single = getattr(engine, "retrieve_single", None)
    if mode in {"bm25", "dense"} and callable(single):
        scores, documents = single(query, dataset, mode)
    else:
        data = engine.retrieve(query, dataset, alpha)
        for key in ("bm25", "dense", "hybrid", "documents"):
            if key not in data:
                raise ValueError(f"Backend retrieve omitted {key}")
        bm25, dense, hybrid, documents = (data[key] for key in ("bm25", "dense", "hybrid", "documents"))
        signals = {"top10_overlap": overlap_at_10(bm25, dense),
                   "normalized_score_gap": normalized_score_gap(hybrid),
                   "query_length": len(query.split())}
        scores = {"bm25": bm25, "dense": dense}.get(mode, hybrid)
        if mode == "adaptive":
            signal, cutoff, direction = policy or ("top10_overlap", threshold, "low")
            value = signals[signal]
            reranked = should_rerank(value, cutoff, direction)
            signals["decision"] = {"signal": signal, "value": value, "threshold": cutoff,
                                   "direction": direction, "calibrated": policy is not None}
    if reranked:
        ids = ranked(scores)
        scores = engine.rerank(query, dataset, ids)
        if set(scores) != set(ids):
            raise ValueError("Live reranker must score the exact Hybrid candidate set")
    rows = []
    for did in ranked(scores):
        doc = documents.get(did)
        if doc is None:
            raise ValueError(f"Missing document text for {did}")
        rows.append((str(did), str(doc.get("title", "")), str(doc.get("abstract", "")), float(scores[did])))
    return {"results": rows, "reranked": bool(reranked),
            "latency_ms": (time.perf_counter()-start)*1000, "signals": signals}
