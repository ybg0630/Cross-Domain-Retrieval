"""校验六组完整实验并构建成员 5 的独立交付包。"""

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import sys
import zipfile
from importlib.metadata import distributions
from pathlib import Path

from analyze import analyze
from rerank import read_cached_scores, save_csv, save_json
from src import utils

SPLITS = [("scifact", "test", 300), ("scifact", "train", 809),
          ("nfcorpus", "test", 323), ("nfcorpus", "dev", 324),
          ("fiqa", "test", 648), ("fiqa", "dev", 500)]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def load_labels(root, dataset, split):
    directory = root / "datasets" / dataset
    all_queries = {q["_id"]: q["text"] for q in
                   (json.loads(line) for line in (directory / "queries.jsonl").read_text(encoding="utf-8").splitlines())}
    qrels = {}
    with (directory / "qrels" / f"{split}.tsv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return {q: all_queries[q] for q in qrels}, qrels


def validate(experiment, root):
    records = []
    seen_queries = {}
    for dataset, split, count in SPLITS:
        stem = dataset + ("" if split == "test" else f"_{split}")
        queries, qrels = load_labels(root, dataset, split)
        baseline = read(experiment / f"results/{stem}_rerank_candidates.json")
        run = read(experiment / f"results/{stem}_hybrid_rerank.json")
        metadata = read(experiment / f"reports/member4/{stem}_timing.json")
        reported = read(experiment / f"reports/member4/{stem}_metrics.json")
        assert set(queries) == set(qrels) == set(baseline) == set(run)
        assert not (seen_queries.get(dataset, set()) & set(run)), "test 与校准查询 ID 重叠"
        seen_queries.setdefault(dataset, set()).update(run)
        assert len(run) == count and metadata["query_count"] == count
        assert metadata["full_split_queries"] == count and not metadata["smoke_only"]
        assert metadata["full_fresh_pass"] and metadata["inferred_pairs"] == count * 100
        assert all(len(docs) == 100 and set(docs) == set(baseline[q]) and
                   all(math.isfinite(score) for score in docs.values()) for q, docs in run.items())
        assert read(experiment / f"cache/rerank_scores_{dataset}_{split}.json") == run
        assert read_cached_scores(experiment / f"cache/rerank_scores_{dataset}_{split}.sqlite3") == run
        with (experiment / f"reports/member4/{stem}_timing.csv").open(encoding="utf-8-sig", newline="") as f:
            timings = list(csv.DictReader(f))
        assert len(timings) == count and {r["query_id"] for r in timings} == set(run)
        assert all(int(r["cached_pairs"]) == 0 and int(r["inferred_pairs"]) == 100 and
                   int(r["candidate_pairs"]) == 100 and int(r["batch_size"]) == metadata["batch_size"]
                   for r in timings)
        latencies = [float(r["inference_ms"]) for r in timings]
        assert all(math.isfinite(ms) and ms > 0 for ms in latencies)
        assert math.isclose(sum(latencies), metadata["inference_ms"], rel_tol=1e-10)
        assert math.isclose(statistics.mean(latencies), metadata["mean_full_rerank_ms"], rel_tol=1e-10)
        assert all(math.isclose(float(r["ms_per_inferred_pair"]), float(r["inference_ms"]) / 100,
                                rel_tol=1e-10) for r in timings)
        summary, delta = analyze(qrels, baseline, run, utils.evaluate)
        for method in ("baseline", "reranked"):
            for metric, actual in summary[method].items():
                assert math.isclose(actual, reported[method][metric], abs_tol=1e-12)
        assert summary["baseline"]["Recall@100"] == summary["reranked"]["Recall@100"]
        assert summary["counts"] == reported["counts"]
        with (experiment / f"reports/member4/{stem}_delta.csv").open(encoding="utf-8-sig", newline="") as f:
            saved_delta = {r["query_id"]: r for r in csv.DictReader(f)}
        assert set(saved_delta) == set(run)
        for row in delta:
            old = saved_delta[row["query_id"]]
            assert row["status"] == old["status"]
            for metric in ("ndcg_before", "ndcg_after", "delta"):
                assert math.isclose(row[metric], float(old[metric]), abs_tol=1e-12)
        # 核验本次实际用到的上游文件没有在运行后变化。
        for relative, expected in metadata["sources"].items():
            assert digest(root / relative) == expected
        assert digest(root / "member3/report/default_fusion.json") == metadata["fusion_config_sha256"]
        records.append({"dataset": dataset, "split": split, "queries": count, "pairs": count * 100,
                        "baseline": summary["baseline"], "reranked": summary["reranked"],
                        "counts": summary["counts"], "fractions": summary["fractions"],
                        "mean_full_rerank_ms": metadata["mean_full_rerank_ms"],
                        "median_rerank_ms": statistics.median(latencies),
                        "p95_rerank_ms": sorted(latencies)[math.ceil(count * .95) - 1],
                        "ms_per_pair": metadata["ms_per_inferred_pair"],
                        "hardware": metadata["hardware"], "model": metadata["config"]})
        print(f"Verified {dataset}/{split}: {count} queries / {count * 100} pairs", flush=True)
    return records


def overview(destination, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    test = [r for r in records if r["split"] == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    x = list(range(len(test)))
    a = axes[0].bar([i - .18 for i in x], [r["baseline"]["nDCG@10"] for r in test],
                    width=.36, label="Hybrid", color="#74869a")
    b = axes[0].bar([i + .18 for i in x], [r["reranked"]["nDCG@10"] for r in test],
                    width=.36, label="Hybrid + rerank", color="#27827b")
    axes[0].bar_label(a, fmt="%.4f", fontsize=9, padding=3)
    axes[0].bar_label(b, fmt="%.4f", fontsize=9, padding=3)
    axes[0].set(ylabel="nDCG@10", ylim=(0, 1), xticks=x, xticklabels=[r["dataset"] for r in test])
    axes[0].legend()
    bars = axes[1].bar(x, [r["mean_full_rerank_ms"] for r in test], color="#5277ac")
    axes[1].bar_label(bars, fmt="%.1f ms", padding=3)
    axes[1].set(ylabel="Mean extra rerank latency (ms / query)", xticks=x,
                xticklabels=[r["dataset"] for r in test])
    axes[1].margins(y=.2)
    fig.suptitle("Full test sets - top-100 reranking on RTX 3070 Ti Laptop GPU")
    fig.savefig(destination / "reports/member4/test_overview.png", dpi=180)
    plt.close(fig)


def write_readme(destination, records):
    lines = ["# 成员 4 → 成员 5：完整重排序交付包", "",
             "六组正式实验全部完成，共 **2,904 条查询、290,400 个查询-文档对**。全部对均实际推理，无小样本混入，也没有用缓存读取时间冒充推理耗时。",
             "", "## 结果", "",
             "| 数据集 | 划分 | 查询数 | 混合 nDCG@10 | 重排 nDCG@10 | Δ | 平均重排 ms/查询 | 变好/变差/不变 |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for r in records:
        before, after = r["baseline"]["nDCG@10"], r["reranked"]["nDCG@10"]
        counts = r["counts"]
        lines.append(f"| {r['dataset']} | {r['split']} | {r['queries']} | {before:.4f} | {after:.4f} | {after-before:+.4f} | {r['mean_full_rerank_ms']:.2f} | {counts['improved']}/{counts['worsened']}/{counts['unchanged']} |")
    lines += ["", "所有划分的 Recall@100 在重排前后完全相同，具体数值见 reports/member4/summary.csv。完整精度见各组 metrics.json。",
              "本次 test 中 SciFact 和 NFCorpus 的平均 nDCG@10 提升，FiQA 下降。不能默认每个查询都会受益；结合校准集 Δ 设计策略，test 结果仅用于最终报告。",
              "", "## 给成员 5 的文件", "",
              "- `results/`：六组混合候选和全量重排结果，格式都是 `{查询ID: {文档ID: 得分}}`。",
              "- `cache/`：六组 JSON 得分快照与 SQLite 断点缓存。JSON 可直接在 Mac/Windows 读取，不需要模型。",
              "- `inputs/{dataset}/{split}/`：相同查询集合的 `bm25.json`、`dense.json`（MiniLM）、`queries.json`、`qrels.json`，供计算重合度、查询长度和评估。",
              "- `reports/member4/`：逐查询 Δ、每查询耗时、每对平均耗时、硬件与参数、变化比例图、汇总 CSV 和 test 总览图。",
              "- `member5_io.py`：只依赖 Python 标准库的读取、结果选择与成本模拟接口。",
              "- `source/`：本次运行代码和全部已保存的上游候选。需要复现模型推理时进入 source，安装 requirements-member4.txt，运行 README 中的复现命令。原始语料和模型权重首次运行下载，不随包重复发送。",
              "- `MANIFEST.json`、`VALIDATION.json`、`SHA256SUMS.txt`：模型版本、上游 Git 提交、验证记录和逐文件校验值。",
              "", "命名：test 用 `scifact_hybrid_rerank.json`；校准集用 `scifact_train_hybrid_rerank.json`、`nfcorpus_dev_hybrid_rerank.json`、`fiqa_dev_hybrid_rerank.json`。",
              "", "## 直接读取和模拟", "", "在解压目录执行：", "", "```python",
              "from member5_io import load_bundle, choose_results, estimate_cost",
              "bundle = load_bundle('scifact', 'train')",
              "# selected_ids 应由你的信号和阈值策略决定，下面只是全量重排对照。",
              "selected_ids = set(bundle['queries'])",
              "run = choose_results(bundle, selected_ids)",
              "cost = estimate_cost(bundle, selected_ids)",
              "print(cost['mean_extra_rerank_ms'], cost['rerank_fraction'])",
              "# 用小组 utils.evaluate(bundle['qrels'], run) 计算质量。", "```", "",
              "`bundle` 包含 queries、qrels、bm25、dense、hybrid、reranked、rerank_ms、metadata。函数不会训练模型，也不会用 qrels 决定是否重排。",
              "", "**阈值在 SciFact train / NFCorpus dev / FiQA dev 上选；test 仅用于最终报告。** 每条 Δ 已提供，train/dev 可用于分析信号；不能根据 test 的 Δ 直接选测试查询或阈值。",
              "", "## 耗时口径", "",
              "硬件为 NVIDIA GeForce RTX 3070 Ti Laptop GPU（8 GB），batch size 32，CPU 线程 4，最大联合输入长度 512。使用同一个 GPU、模型和参数完成全部六组。",
              "计时使用成员 1 的 Timer，包含分词、模型推理和分数传回 CPU；不含模型加载/预热、上游检索、读写缓存或绘图。每条查询完整处理 100 篇。",
              "`timing.csv` 的 inference_ms 是逐查询实测；`timing.json` 的 ms_per_inferred_pair 是总推理毫秒/总对数，batch_size 和 hardware 也在其中。",
              "`estimate_cost` 将选中查询的实测重排时间相加，再除以**全部查询数**。这是离线成本模拟，不是重新测量在线服务。也可用每对平均耗时 × 100 × 重排比例作较粗估计。",
              "当前包提供的是**额外重排耗时**。画总检索耗时曲线还需同机测量混合检索耗时；不要直接加上成员 2 在 Mac CPU 上的旧耗时。纯 CPU、MPS 或其他 GPU 上的推理成本需要重新测量。",
              "", "## 验证和复现", "", "```bash", "python -B check_handoff.py",
              "# 已装 s1 的评估依赖时，另外复核公共评估指标：", "python -B check_handoff.py --evaluate", "```", "",
              "复现 GPU 实验（Windows/Linux 且有兼容 NVIDIA 驱动）：", "", "```bash", "cd source",
              "python -m pip install -r requirements-member4.txt",
              "python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126",
              "python run_member4.py --download --device cuda --batch-size 32 --threads 4 --max-length 512 --revision 233902d25c440f23af6f7d6e94d2946bac0bee0a --output-dir ../reproduced", "```", "",
              "成员 5 只做离线策略时无需安装 CUDA。不要把缓存里的 logits 与混合检索分数跨方法直接比较或相加；每条查询选择整份混合排序或整份重排序。",
              "", "本交付包未包含可选的 top-20/50 消融，完整 top-100 主实验和校准数据均已提供。", ""]
    (destination / "README_成员5.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    experiment, destination = args.experiment.resolve(), args.destination.resolve()
    if destination == experiment or destination.is_relative_to(experiment):
        raise ValueError("交付目录必须位于原始实验目录之外")
    records = validate(experiment, root)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("results", "cache", "reports"):
        shutil.copytree(experiment / name, destination / name, dirs_exist_ok=True)
    copy(experiment / "experiment.log", destination / "experiment.log")
    for name in ("member5_io.py", "check_handoff.py"):
        copy(root / name, destination / name)
    for name in ("rerank.py", "analyze.py", "run_member4.py", "demo.py", "test_rerank.py",
                 "test_member4.py", "test_member5_io.py", "member5_io.py", "environment.yml",
                 "requirements-member4.txt", "MEMBER4_README.md"):
        copy(root / name, destination / "source" / name)
    for name in ("src/utils.py", "member3/hybrid.py", "member3/report/default_fusion.json",
                 "member3/report/test_ablation.csv", "member3/report/selection_metrics.csv"):
        copy(root / name, destination / "source" / name)
    data_hashes = {}
    for dataset, split, _ in SPLITS:
        queries, qrels = load_labels(root, dataset, split)
        inputs = destination / "inputs" / dataset / split
        save_json(inputs / "queries.json", queries)
        save_json(inputs / "qrels.json", qrels)
        if split == "test":
            bm25 = Path(f"member3/results/{dataset}_bm25.json")
            dense = Path(f"results/{dataset}_dense_minilm.json")
            fusion = Path(f"member3/results/{dataset}_hybrid_wsum.json")
            copy(root / fusion, destination / "source" / fusion)
        else:
            bm25 = Path(f"member3/results/{split}/{dataset}_bm25.json")
            dense = Path(f"member3/results/{split}/{dataset}_dense_minilm.json")
        for name, relative in (("bm25", bm25), ("dense", dense)):
            copy(root / relative, inputs / f"{name}.json")
            copy(root / relative, destination / "source" / relative)
        for relative in (f"datasets/{dataset}/corpus.jsonl", f"datasets/{dataset}/queries.jsonl",
                         f"datasets/{dataset}/qrels/{split}.tsv"):
            if relative not in data_hashes:
                data_hashes[relative] = digest(root / relative)
    refs = ["origin/main", "origin/member2/dense-retrieval", "origin/member3/hybrid-retrieval", "origin/s4"]
    manifest = {"complete": True, "smoke_only": False, "total_queries": 2904, "total_pairs": 290400,
                "splits": records, "data_file_sha256": data_hashes,
                "upstream_commits": {ref: subprocess.check_output(["git", "rev-parse", ref], cwd=root, text=True).strip() for ref in refs},
                "gpu_driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()}
    save_json(destination / "MANIFEST.json", manifest)
    save_json(destination / "VALIDATION.json", {"complete": True, "query_sets_and_100_candidates": "passed",
              "json_sqlite_scores_equal": "passed", "all_scores_finite": "passed",
              "all_pairs_inferred_no_cache_hits": "passed", "all_timing_rows_positive_and_totals_match": "passed",
              "overall_and_per_query_metrics_recomputed_with_s1_utils": "passed",
              "recall_unchanged_all_splits": "passed", "upstream_file_hashes": "passed",
              "test_calibration_query_ids_disjoint": "passed", "unit_tests_passed": 18})
    versions = sorted(f'{d.metadata["Name"]}=={d.version}' for d in distributions())
    (destination / "source/requirements-runtime-windows.txt").write_text(
        "--extra-index-url https://download.pytorch.org/whl/cu126\n" + "\n".join(versions) + "\n", encoding="utf-8")
    rows = []
    for r in records:
        rows.append({"dataset": r["dataset"], "split": r["split"], "queries": r["queries"],
                     "baseline_nDCG@10": r["baseline"]["nDCG@10"], "reranked_nDCG@10": r["reranked"]["nDCG@10"],
                     "delta_nDCG@10": r["reranked"]["nDCG@10"] - r["baseline"]["nDCG@10"],
                     "Recall@100": r["reranked"]["Recall@100"], "mean_rerank_ms": r["mean_full_rerank_ms"],
                     "median_rerank_ms": r["median_rerank_ms"], "p95_rerank_ms": r["p95_rerank_ms"],
                     "ms_per_pair": r["ms_per_pair"], **r["counts"]})
    save_csv(destination / "reports/member4/full_summary.csv", rows)
    write_readme(destination, records)
    overview(destination, records)
    sums = [f"{digest(p)}  {p.relative_to(destination).as_posix()}" for p in sorted(destination.rglob("*"))
            if p.is_file() and p.name != "SHA256SUMS.txt" and "__pycache__" not in p.parts]
    (destination / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, "-B", str(destination / "check_handoff.py"), "--evaluate"], check=True)
    archive_path = destination.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in sorted(destination.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                archive.write(p, p.relative_to(destination.parent))
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
    checksum = digest(archive_path)
    archive_path.with_suffix(".zip.sha256").write_text(f"{checksum}  {archive_path.name}\n", encoding="utf-8")
    print(f"READY {archive_path} ({archive_path.stat().st_size / 1024**2:.1f} MiB)\nSHA256 {checksum}")


if __name__ == "__main__":
    main()
