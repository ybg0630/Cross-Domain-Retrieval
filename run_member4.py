"""接入 s1/s2/s3：核对候选、重排序、公共评估和交付报告。"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .analyze import analyze, write_analysis
    from .rerank import (MODEL, CrossEncoderScorer, document_text, rerank,
                         save_csv, save_json, select_candidates)
else:
    from analyze import analyze, write_analysis
    from rerank import (MODEL, CrossEncoderScorer, document_text, rerank,
                        save_csv, save_json, select_candidates)


def find_repo(path=None):
    choices = [Path(path)] if path else [Path(__file__).resolve().parent,
                                        Path(__file__).resolve().parent.parent / "Cross-Domain-Retrieval"]
    for root in choices:
        if (root / "src/utils.py").is_file() and (root / "member3/report/default_fusion.json").is_file():
            return root.resolve()
    raise FileNotFoundError("请用 --repo-root 指定含 src/utils.py 和 member3/report/default_fusion.json 的仓库")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_split(root, dataset, split, utils, download=False):
    """test 直接调用 s1；校准集复用同一 BEIR 读取器及文本拼接方式。"""
    data_dir = root / "datasets" / dataset
    needed = [data_dir / "corpus.jsonl", data_dir / "queries.jsonl", data_dir / "qrels" / f"{split}.tsv"]
    if not all(path.is_file() for path in needed):
        if not download:
            raise FileNotFoundError(f"缺少 {dataset}/{split} 数据；先下载数据或添加 --download")
        from beir import util
        util.download_and_unzip(
            f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip",
            str(root / "datasets"))
    if split == "test":
        previous = Path.cwd()
        try:
            os.chdir(root)  # s1 使用 cwd/datasets；结束后恢复，不修改其代码。
            return utils.load_dataset(dataset)
        finally:
            os.chdir(previous)
    from beir.datasets.data_loader import GenericDataLoader
    corpus, queries, qrels = GenericDataLoader(data_folder=str(data_dir)).load(split=split)
    return {did: document_text(doc) for did, doc in corpus.items()}, queries, qrels


def get_candidates(root, dataset, split, settings, hybrid):
    """test 读已交付结果；train/dev 使用成员 3 的函数与已冻结参数。"""
    config = settings["datasets"][dataset]
    if settings["top_k"] != 100 or settings["method"] not in ("wsum", "rrf"):
        raise ValueError("只支持成员 3 约定的 top-100 wsum/rrf 候选")
    if split == "test":
        source = root / config["test_candidates"]
        run = read_json(source)
        sources = [source]
    else:
        if split != config["selection_split"]:
            raise ValueError(f"{dataset} 的校准划分应为 {config['selection_split']}")
        directory = root / "member3/results" / split
        sources = [directory / f"{dataset}_bm25.json",
                   directory / f"{dataset}_dense_{settings['dense_model']}.json"]
        bm25, dense = [read_json(path) for path in sources]
        if not bm25 or bm25.keys() != dense.keys():
            raise ValueError(f"{dataset}/{split} 的 BM25 与 dense 查询 ID 不一致")
        if settings["method"] == "wsum":
            run = {qid: hybrid.weighted_sum(bm25[qid], dense[qid], config["alpha"])
                   for qid in sorted(bm25)}
        else:
            run = {qid: hybrid.rrf(bm25[qid], dense[qid], k=config.get("k", 60))
                   for qid in sorted(bm25)}
    provenance = {str(path.relative_to(root)): file_sha256(path) for path in sources}
    return run, provenance


def team_measure(utils):
    def measure(operation):
        with utils.Timer() as timer:
            value = operation()
        return value, timer.elapsed_ms
    return measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=["scifact", "nfcorpus", "fiqa"],
                        default=["scifact", "nfcorpus", "fiqa"])
    parser.add_argument("--splits", nargs="+", choices=["test", "calibration"], default=["test", "calibration"])
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="全量核验数据与基线指标，不运行模型")
    parser.add_argument("--limit", type=int, help="每个划分只测试前 N 个查询，强制隔离到 smoke_N 子目录")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.limit is not None and (args.limit < 1 or args.check_only):
        parser.error("--limit 必须为正，且不能与 --check-only 一起使用")
    root = find_repo(args.repo_root)
    sys.path.insert(0, str(root))
    from src import utils
    from member3 import hybrid

    config_path = root / "member3/report/default_fusion.json"
    settings = read_json(config_path)
    output = args.output_dir.resolve() if args.output_dir else root
    if args.limit:
        output = (output if args.output_dir else root / "member4_smoke") / f"smoke_{args.limit}"
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".mplconfig"))
    scorer = CrossEncoderScorer(args.model, args.revision, args.device, args.max_length, args.threads)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    checks, totals = [], []
    for dataset in dict.fromkeys(args.datasets):
        for selection in dict.fromkeys(args.splits):
            split = "test" if selection == "test" else settings["datasets"][dataset]["selection_split"]
            print(f"[{dataset}/{split}] loading and checking upstream inputs", flush=True)
            corpus, queries, qrels = load_split(root, dataset, split, utils, args.download)
            baseline, sources = get_candidates(root, dataset, split, settings, hybrid)
            if set(qrels) != set(queries):
                raise ValueError(f"{dataset}/{split}: queries 与 qrels 的 ID 不一致")
            if any(len(docs) != 100 for docs in baseline.values()):
                raise ValueError(f"{dataset}/{split}: 成员 3 的候选必须每条查询恰好 100 篇")
            baseline = select_candidates(corpus, queries, baseline)
            checks.append({"dataset": dataset, "split": split, "corpus_size": len(corpus),
                           "queries": len(queries), "pairs": sum(map(len, baseline.values())),
                           "baseline_metrics": utils.evaluate(qrels, baseline), "sources": sources})
            if args.check_only:
                print(json.dumps(checks[-1], ensure_ascii=False), flush=True)
                continue
            full_count = len(queries)
            if args.limit:
                ids = sorted(queries)[:args.limit]  # 固定 ID 顺序，绝不根据效果挑样本。
                queries = {q: queries[q] for q in ids}
                qrels = {q: qrels[q] for q in ids}
                baseline = {q: baseline[q] for q in ids}
            stem = dataset + ("" if split == "test" else f"_{split}")
            save_json(output / "results" / f"{stem}_rerank_candidates.json", baseline)

            def progress(done, total):
                if done == 1 or done % 25 == 0 or done == total:
                    print(f"[{dataset}/{split}] {done}/{total} queries", flush=True)

            cache_path = output / "cache" / f"rerank_scores_{dataset}_{split}.sqlite3"
            run, timing, metadata = rerank(corpus, queries, baseline, cache_path, scorer,
                                           dataset=dataset, batch_size=args.batch_size,
                                           refresh=args.refresh, measure=team_measure(utils), progress=progress)
            metadata.update(split=split, smoke_only=args.limit is not None,
                            full_split_queries=full_count, sources=sources,
                            fusion=settings, fusion_config_sha256=file_sha256(config_path),
                            timestamp_utc=stamp, timer="src.utils.Timer", evaluator="src.utils.evaluate")
            save_json(output / "results" / f"{stem}_hybrid_rerank.json", run)
            # JSON 是本次候选的快照，SQLite 是可续跑的历史分数库。
            save_json(output / "cache" / f"rerank_scores_{dataset}_{split}.json", run)
            report = output / "reports/member4" / stem
            save_csv(str(report) + "_timing.csv", timing)
            save_json(str(report) + "_timing.json", metadata)
            # 每次计时留档，读缓存重跑不会抹掉之前的完整推理记录。
            history = output / "reports/member4/history" / f"{stem}_{stamp}"
            save_csv(str(history) + "_timing.csv", timing)
            save_json(str(history) + "_timing.json", metadata)
            summary, rows = analyze(qrels, baseline, run, utils.evaluate)
            summary.update(dataset=dataset, split=split, query_count=len(queries), smoke_only=args.limit is not None)
            write_analysis(report, summary, rows, plot=True)
            totals.append({"dataset": dataset, "split": split, "queries": len(queries),
                           "smoke_only": args.limit is not None,
                           "baseline_nDCG@10": summary["baseline"]["nDCG@10"],
                           "reranked_nDCG@10": summary["reranked"]["nDCG@10"],
                           "baseline_Recall@100": summary["baseline"]["Recall@100"],
                           "reranked_Recall@100": summary["reranked"]["Recall@100"],
                           "mean_full_rerank_ms": metadata["mean_full_rerank_ms"],
                           "full_fresh_pass": metadata["full_fresh_pass"]})
            save_csv(output / "reports/member4/summary.csv", totals)
            print(json.dumps(totals[-1], ensure_ascii=False), flush=True)
    check_path = (root / "member4_checks" if args.check_only and not args.output_dir else
                  output / "reports/member4") / "input_validation.json"
    save_json(check_path, checks)
    print(f"Done: {check_path if args.check_only else output}", flush=True)


if __name__ == "__main__":
    main()
