"""成员 5 离线接口：读交付包、按查询选择结果、估计额外重排耗时。"""

import csv
import json
from pathlib import Path


def load_bundle(dataset, split="test", root=None):
    root = Path(root) if root else Path(__file__).resolve().parent
    stem = dataset + ("" if split == "test" else f"_{split}")
    def read(path):
        return json.loads((root / path).read_text(encoding="utf-8"))

    inputs = f"inputs/{dataset}/{split}"
    bundle = {key: read(f"{inputs}/{key}.json") for key in ("queries", "qrels", "bm25", "dense")}
    bundle["hybrid"] = read(f"results/{stem}_rerank_candidates.json")
    bundle["reranked"] = read(f"results/{stem}_hybrid_rerank.json")
    bundle["metadata"] = read(f"reports/member4/{stem}_timing.json")
    with (root / f"reports/member4/{stem}_timing.csv").open(encoding="utf-8-sig", newline="") as f:
        bundle["rerank_ms"] = {r["query_id"]: float(r["inference_ms"]) for r in csv.DictReader(f)}
    query_ids = set(bundle["queries"])
    if not query_ids or any(set(bundle[k]) != query_ids for k in
                            ("qrels", "bm25", "dense", "hybrid", "reranked", "rerank_ms")):
        raise ValueError("交付文件的查询 ID 集合不一致")
    if bundle["metadata"]["smoke_only"] or not bundle["metadata"]["full_fresh_pass"]:
        raise ValueError("需要完整、无缓存命中的正式实测结果")
    return bundle


def choose_results(bundle, rerank_query_ids):
    """策略由成员 5 决定；这里仅按所选查询拼出 run，不读取相关性标注。"""
    selected = set(rerank_query_ids)
    if not selected <= bundle["queries"].keys():
        raise ValueError("策略包含未知查询 ID")
    return {q: dict(bundle["reranked"][q] if q in selected else bundle["hybrid"][q])
            for q in bundle["queries"]}


def estimate_cost(bundle, rerank_query_ids):
    """离线模拟重排额外耗时；平均值分母是所有查询，不含上游检索耗时。"""
    selected = set(rerank_query_ids)
    if not selected <= bundle["queries"].keys():
        raise ValueError("策略包含未知查询 ID")
    total_ms = sum(bundle["rerank_ms"][q] for q in selected)
    return {"rerank_fraction": len(selected) / len(bundle["queries"]),
            "mean_extra_rerank_ms": total_ms / len(bundle["queries"]),
            "total_rerank_ms": total_ms}
