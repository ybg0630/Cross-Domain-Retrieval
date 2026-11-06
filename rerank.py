"""成员 4：候选重排序、断点缓存和真实推理计时。"""

import argparse
import csv
import hashlib
import json
import math
import platform
import sqlite3
from contextlib import closing
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def save_json(path, value):
    """先写临时文件再替换，避免中断后留下半个 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_dataset(data_dir, split="test"):
    """读取 BEIR 本地文件；只返回该 split 的有标注查询。"""
    root = Path(data_dir)
    def read_jsonl(name):
        with (root / name).open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    corpus = {str(d["_id"]): {"title": d.get("title", ""), "text": d["text"]}
              for d in read_jsonl("corpus.jsonl")}
    queries = {str(q["_id"]): q["text"] for q in read_jsonl("queries.jsonl")}
    qrels = {}
    with (root / "qrels" / f"{split}.tsv").open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return corpus, {qid: queries[qid] for qid in qrels}, qrels


def measure_ms(operation):
    """可由成员 1 的 Timer 适配器替换：返回 (执行结果, 毫秒)。"""
    start = perf_counter()
    value = operation()
    return value, (perf_counter() - start) * 1000


class CrossEncoderScorer:
    """延迟加载：全部命中缓存时不加载模型。返回原始 logits。"""

    def __init__(self, model=MODEL, revision=None, device=None, max_length=512):
        if max_length < 1:
            raise ValueError("max_length 必须为正")
        self.config = {"model": model, "revision": revision, "max_length": max_length,
                       "text_format": "title + space + text", "activation": "identity"}
        self.device = device
        self.model = None
        self.hardware = {"platform": platform.platform(), "cpu": platform.processor(),
                         "requested_device": device or "auto"}

    def prepare(self, example):
        import torch
        from sentence_transformers import CrossEncoder

        torch.manual_seed(42)
        self.model = CrossEncoder(self.config["model"], revision=self.config["revision"],
                                  device=self.device, max_length=self.config["max_length"])
        self.hardware.update(device=str(self.model.device), torch=torch.__version__)
        self.hardware["packages"] = {name: version(name) for name in
                                     ("sentence-transformers", "transformers", "numpy")}
        if self.model.device.type == "cuda":
            self.hardware["gpu"] = torch.cuda.get_device_name(self.model.device)
        self.predict([example], 1)  # 预热不计入查询推理耗时。

    def predict(self, pairs, batch_size):
        import torch

        scores = self.model.predict(pairs, batch_size=batch_size,
                                    activation_fn=torch.nn.Identity(),
                                    show_progress_bar=False, convert_to_numpy=True)
        if self.model.device.type == "cuda":
            torch.cuda.synchronize(self.model.device)
        return scores.tolist()


def read_cached_scores(cache_path):
    """给成员 5：读取缓存中全部已算分数，返回 {qid: {docid: score}}。"""
    uri = Path(cache_path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        result = {}
        for qid, did, score in db.execute("SELECT qid, did, score FROM scores"):
            result.setdefault(qid, {})[did] = score
    return result


def rerank(corpus, queries, candidates, cache_path, scorer, *, dataset,
           batch_size=32, refresh=False, measure=measure_ms):
    """返回 (run, timing_rows, metadata)；scorer 接口见 README。"""
    if batch_size < 1 or not queries or set(queries) != set(candidates):
        raise ValueError("batch_size 必须为正；queries 与 candidates 的查询 ID 必须完整一致且非空")
    # 在昂贵推理前验证整个输入，避免漏查询或缺文档悄悄改变评估集合。
    selected = {}
    for qid, docs in candidates.items():
        if not isinstance(qid, str) or not isinstance(queries[qid], str) or not docs:
            raise ValueError(f"查询 ID/文本必须为字符串，候选不能为空: {qid}")
        for did, score in docs.items():
            if not isinstance(did, str) or did not in corpus or not math.isfinite(float(score)):
                raise ValueError(f"无效文档 ID 或候选分数: {qid}/{did}")
            if not all(isinstance(corpus[did].get(k, ""), str) for k in ("title", "text")):
                raise ValueError(f"文档标题、正文必须是字符串: {did}")
        selected[qid] = sorted(docs, key=lambda d: (-float(docs[d]), d))[:100]

    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = json.dumps({"dataset": dataset, **scorer.config}, sort_keys=True)
    run, timing, prepared = {}, [], False
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS scores "
                   "(qid TEXT, did TEXT, fingerprint TEXT, score REAL, PRIMARY KEY(qid, did))")
        old = db.execute("SELECT value FROM metadata WHERE key='config'").fetchone()
        if old and old[0] != config:
            raise ValueError("缓存的数据集或模型配置不同，请使用新的缓存文件")
        db.execute("INSERT OR REPLACE INTO metadata VALUES ('config', ?)", (config,))
        db.commit()
        for qid, doc_ids in selected.items():
            cached = {did: (digest, score) for did, digest, score in db.execute(
                "SELECT did, fingerprint, score FROM scores WHERE qid=?", (qid,))}
            scores, missing, pairs = {}, [], []
            for did in doc_ids:
                doc = corpus[did]
                text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
                pair = [queries[qid], text]
                digest = hashlib.sha256(json.dumps(pair, ensure_ascii=False).encode()).hexdigest()
                if not refresh and did in cached and cached[did][0] == digest:
                    scores[did] = cached[did][1]
                else:
                    missing.append((did, digest))
                    pairs.append(pair)

            elapsed = 0.0
            if pairs:
                if not prepared:
                    scorer.prepare(pairs[0])
                    prepared = True
                values, elapsed = measure(lambda: scorer.predict(pairs, batch_size))
                if len(values) != len(pairs) or not all(math.isfinite(float(v)) for v in values):
                    raise ValueError("模型必须为每个查询-文档对返回一个有限标量")
                if not math.isfinite(elapsed) or elapsed < 0:
                    raise ValueError("计时器必须返回非负的毫秒数")
                for (did, digest), score in zip(missing, values):
                    score = float(score)
                    scores[did] = score
                    db.execute("INSERT OR REPLACE INTO scores VALUES (?, ?, ?, ?)",
                               (qid, did, digest, score))
                db.commit()  # 每条查询落盘，中断后只重做未完成的查询。
            run[qid] = dict(sorted(scores.items(), key=lambda item: (-item[1], item[0])))
            timing.append({"query_id": qid, "candidate_pairs": len(doc_ids),
                           "inferred_pairs": len(pairs), "cached_pairs": len(doc_ids) - len(pairs),
                           "inference_ms": elapsed,
                           "ms_per_inferred_pair": elapsed / len(pairs) if pairs else None,
                           "batch_size": batch_size})

    count = sum(row["inferred_pairs"] for row in timing)
    elapsed = sum(row["inference_ms"] for row in timing)
    fresh = all(row["cached_pairs"] == 0 for row in timing)
    metadata = {"dataset": dataset, "config": scorer.config, "hardware": scorer.hardware,
                "query_count": len(run), "inferred_pairs": count, "inference_ms": elapsed,
                "ms_per_inferred_pair": elapsed / count if count else None,
                "full_fresh_pass": fresh,
                "mean_full_rerank_ms": elapsed / len(run) if fresh else None,
                "timing_scope": "tokenization + model prediction + CPU scores; excludes load/warmup/cache/IO",
                "batch_size": batch_size}
    return run, timing, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["scifact", "nfcorpus", "fiqa"])
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--output-dir", default=Path("."), type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", help="建议指定 Hugging Face commit ID，固定模型版本")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--refresh", action="store_true", help="重新推理全部候选，用于正式计时")
    args = parser.parse_args()
    corpus, queries, _ = load_dataset(args.data_dir, args.split)
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    cache = args.cache or args.output_dir / "cache" / f"rerank_scores_{args.dataset}_{args.split}.sqlite3"
    scorer = CrossEncoderScorer(args.model, args.revision, args.device, args.max_length)
    run, timing, metadata = rerank(corpus, queries, candidates, cache, scorer,
                                  dataset=args.dataset, batch_size=args.batch_size,
                                  refresh=args.refresh)
    # test 文件名保持小组约定，dev/train 加后缀，避免互相覆盖。
    stem = args.dataset + ("" if args.split == "test" else f"_{args.split}")
    metadata.update(split=args.split, candidates=str(args.candidates.resolve()), cache=str(cache.resolve()))
    save_json(args.output_dir / "results" / f"{stem}_hybrid_rerank.json", run)
    save_csv(args.output_dir / "reports" / f"{stem}_timing.csv", timing)
    save_json(args.output_dir / "reports" / f"{stem}_timing.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
