# 成员 4：全量重排序

接收成员 3 的混合检索候选，对每条查询的 top-100 使用 Cross-Encoder 重新打分，输出结果、缓存、耗时和逐查询变化。三个数据集分别运行，不混合语料。

默认模型：`cross-encoder/ms-marco-MiniLM-L6-v2`。PDF 中的 `MiniLM-L-6-v2` 拼写有误，以这里的官方模型名为准。

## 文件与当前状态

| 文件 | 用途 |
|---|---|
| `rerank.py` | 数据加载、批量打分、SQLite 缓存、计时、命令行入口 |
| `analyze.py` | 调用成员 1 的评估函数，输出指标、逐查询 Δ、可选图片 |
| `demo.py` | 无需上游的小样本演示，支持假分数或真实模型 |
| `test_rerank.py` | 缓存复用、增量更新、中断续跑和接口检查 |
| `environment.yml` / `requirements.txt` | 独立环境与依赖 |
| `requirements-lock.txt` | 本机验证环境的完整版本清单（Windows / Python 3.11） |

目前没有正式上游结果，也没有成员 1 的 `utils.py`，所以没有三个数据集的正式实验指标。`demo_output/` 中只放演示产物，不能作为项目实验结果。

已验证：独立 Conda 环境依赖检查通过；10 项自动测试通过；真实模型在 CPU 上完成 6 对示例打分，两条查询的相关文档均排第一；再次读取同一缓存无需推理。分析脚本也已用评估接口桩验证 JSON、CSV 和 PNG 输出，尚未与成员 1 的真实评估实现联调。

## 环境和快速运行

本机独立 Conda 环境放在代码文件夹外，打包 `code_4` 不会把环境带进去：

```powershell
conda activate L:\ntu_class\6405\project\.envs\ee6405-member4
cd L:\ntu_class\6405\project\code_4
python -m unittest -v
python demo.py
python demo.py
```

假分数演示不需要额外依赖。首次运行计算 6 对，第二次全部读缓存，`inferred_pairs` 为 0。重复运行会覆盖本次演示的耗时记录。

若本机 `conda` 命令启动异常，可直接调用该环境的 Python：

```powershell
& 'L:\ntu_class\6405\project\.envs\ee6405-member4\python.exe' demo.py
```

其他电脑在本文件夹执行：

```powershell
conda env create -f environment.yml
conda activate ee6405-member4
```

若要复现本机的完整依赖版本，在 Python 3.11 环境运行 `python -m pip install -r requirements-lock.txt`。其他平台可先使用 `environment.yml`。

真实模型演示（首次需联网下载权重）：

```powershell
python demo.py --real-model --device cpu
```

模型依赖由 `sentence-transformers` 安装。当前先用 CPU 验证；正式使用 CUDA 前，需要安装与本机驱动相容的 PyTorch，并确认 `torch.cuda.is_available()` 为真。

## 上游需要给什么

**成员 1：数据和评估接口。** Python 内存格式如下，ID 全部为字符串：

```python
corpus = {"d1": {"title": "文档标题", "text": "文档正文"}}
queries = {"q1": "查询文本"}
qrels = {"q1": {"d1": 1}}  # 相关性标注，仅用于评估

# 小组统一接口，analyze.py 直接调用它，不重复实现指标。
def evaluate(qrels, run):
    return {"nDCG@10": ..., "Recall@100": ...}
```

接入时直接使用成员 1 的 `load_dataset(name)` 返回值即可。命令行也支持标准 BEIR 本地目录，无需安装 BEIR：

```text
datasets/scifact/
  corpus.jsonl       # 每行 {"_id":"d1","title":"...","text":"..."}
  queries.jsonl      # 每行 {"_id":"q1","text":"..."}
  qrels/test.tsv     # 表头 query-id<TAB>corpus-id<TAB>score
  qrels/train.tsv
```

`load_dataset(data_dir, split)` 按 qrels 选择该划分的查询。NFCorpus、FiQA 的 dev 读取 `qrels/dev.tsv`。SciFact 用 train：CLI 默认读取全部有标注的 train 查询；只用部分 train 时，通过 Python 接口传入大家约定的同一查询子集及其候选、qrels，不能按效果挑选。

**成员 3：固定融合方法与参数后的候选 JSON。**

```json
{
  "q1": {"d1": 0.72, "d2": 0.65},
  "q2": {"d8": 0.81, "d3": 0.52}
}
```

分数越大排名越靠前。按分数降序选前 100，同分按文档 ID 升序；小样本或候选不足 100 时保留全部。每条查询至少有一篇候选，所有文档 ID 必须存在于 corpus。查询 ID 集合必须和输入 queries **完全一致**，不允许漏掉难查询。dev/train/test 候选分文件提供。候选暂时也可用 BM25，但相关产物需单独放到 `--output-dir bm25_trial`，不得报告成混合检索结果。

## 重排序接口与运行命令

Python 接口可直接接成员 1 的数据：

```python
from rerank import CrossEncoderScorer, rerank, save_json

run, timing_rows, metadata = rerank(
    corpus, queries, candidates,
    cache_path="cache/rerank_scores_scifact_test.sqlite3",
    scorer=CrossEncoderScorer(device="cpu"),
    dataset="scifact",
    batch_size=32,
)
save_json("results/scifact_hybrid_rerank.json", run)
```

`run` 仍是 `{query_id: {doc_id: score}}`；分数是 Cross-Encoder 原始 logits，越大越相关，不是概率。文档文本为 `(title + " " + text).strip()`，查询和文档联合输入，默认最多 512 tokens，过长输入由模型分词器截断。

上面的导入方式适用于工作目录在 `code_4` 内。队友在 `project` 根目录整合时，使用 `from code_4.rerank import rerank, CrossEncoderScorer` 和 `from code_4.analyze import analyze`，无需复制代码或修改全局 Python 路径。

命令行（以下数据路径需换成成员 1、3 提供的实际路径）：

```powershell
python rerank.py --dataset scifact --data-dir ../datasets/scifact --candidates ../results/scifact_hybrid.json --device cpu
python rerank.py --dataset nfcorpus --data-dir ../datasets/nfcorpus --candidates ../results/nfcorpus_hybrid.json --device cpu
python rerank.py --dataset fiqa --data-dir ../datasets/fiqa --candidates ../results/fiqa_hybrid.json --device cpu
```

验证集示例：

```powershell
python rerank.py --dataset nfcorpus --split dev --data-dir ../datasets/nfcorpus --candidates ../results/nfcorpus_dev_hybrid.json --device cpu
python rerank.py --dataset scifact --split train --data-dir ../datasets/scifact --candidates ../results/scifact_train_hybrid.json --device cpu
```

可选参数：`--batch-size 32`、`--max-length 512`、`--model 模型名`、`--revision 模型commitID`、`--cache 缓存路径`、`--output-dir 输出目录`、`--refresh`。正式实验建议固定模型 commit ID 和依赖版本。

计时默认使用 `perf_counter`。拿到成员 1 的 `Timer` 后，可通过 `rerank(..., measure=measure_with_team_timer)` 接入；适配器必须是 `measure_with_team_timer(operation) -> (operation的返回值, 耗时毫秒)`，内部用成员 1 的 Timer 包住一次 `operation()`。由于当前不知道其 Timer 属性名，未假定具体实现，也未修改上游文件。

## 输出及下游接口

默认输出以 SciFact test 为例：

| 文件 | 消费方 / 内容 |
|---|---|
| `results/scifact_hybrid_rerank.json` | 成员 1、5、6：本次完整重排结果 |
| `cache/rerank_scores_scifact_test.sqlite3` | 成员 5：所有已计算的查询-文档得分 |
| `reports/scifact_timing.csv` | 成员 5：逐查询推理时间、实际推理对数、缓存命中数、batch size |
| `reports/scifact_timing.json` | 成员 5：硬件、模型配置、平均每对耗时、完整重排平均耗时 |

dev/train 的结果和报告前缀分别为 `scifact_dev`、`scifact_train`，缓存也按划分隔离。

成员 5 直接读取结果即可做“是否重排”的模拟：

```python
import json
from pathlib import Path

full_run = json.loads(Path("results/scifact_hybrid_rerank.json").read_text(encoding="utf-8"))
# hybrid_run 是成员 3 的候选；should_rerank 由成员 5 自己的策略决定。
adaptive_run = {
    qid: full_run[qid] if should_rerank(qid) else hybrid_run[qid]
    for qid in full_run
}
```

如需按查询 ID、文档 ID 查历史分数：

```python
from rerank import read_cached_scores
scores = read_cached_scores("cache/rerank_scores_scifact_test.sqlite3")
score = scores["q1"]["d1"]
```

SQLite 的 `scores` 表字段为 `qid, did, fingerprint, score`，主键是 `(qid, did)`。缓存保存历史候选，可能多于本次候选；**评估和自适应模拟优先读本次 results JSON**，不能直接把历史缓存当成当前候选结果。库内 `metadata` 表记录数据集和模型配置。

每条查询计算成功就提交缓存。新候选只计算缺失对；查询/文档文本发生变化时，指纹检查使相关分数失效；模型、revision、截断长度不同会拒绝混用缓存，需换缓存路径。不要同时让多个进程写同一个缓存。相同模型名下若替换了权重、修改本地模型或升级影响推理的依赖，也应换缓存或 `--refresh`；程序不会自动检测远端权重更新。

## 耗时与评估

`inference_ms` 包含分词、批量模型推理和分数回传 CPU，不包含模型加载、预热、缓存读写、结果排序或上游检索。这里不是整个应用的端到端延迟。

- `ms_per_inferred_pair`：总推理毫秒 / 真正新算的对数；全部命中缓存时为 null。
- `mean_full_rerank_ms`：只有本次所有候选都重新推理时才有值，否则为 null。
- `full_fresh_pass=true`：本次无缓存命中，可用于完整重排耗时对比。

正式计时用新输出目录并加 `--refresh`，避免覆盖之前的记录，例如在运行命令末尾加 `--output-dir benchmark_cpu --refresh`。统一硬件、batch size 和长度。成员 5 可用实测每对耗时 × 重排对数估算额外耗时；这是离线估算。若要总检索耗时，还需加上上游混合检索耗时。

拿到成员 1 的 `utils.py` 后：

```powershell
python analyze.py --data-dir ../datasets/scifact --baseline ../results/scifact_hybrid.json --reranked results/scifact_hybrid_rerank.json --utils ../utils.py --output-prefix reports/scifact --plot
```

如其指标键叫 `NDCG@10`，添加 `--ndcg-key NDCG@10`；如其返回值不是字典，先让成员 1 提供约定的字典适配接口。成员 1 的额外依赖需按其 requirements 安装。分析函数只调用统一 `evaluate`，不自行实现 nDCG 或 Recall。

输出：`scifact_metrics.json`（总体指标与变好/变差/不变比例）、`scifact_delta.csv`（`query_id, ndcg_before, ndcg_after, delta, status`）、可选 `scifact_delta.png`。Python 接口为 `analyze(qrels, baseline, reranked, evaluate, ndcg_key="nDCG@10") -> (summary, rows)`。

`delta = 重排后 nDCG@10 - 重排前 nDCG@10`，绝对值不超过 `1e-9` 记为不变。同一批 top-100 只改变顺序时，Recall@100 应保持一致。当前不包含可选的 top-20/50 深度消融。

## 参考

- [官方模型及使用示例](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- [CrossEncoder API](https://sbert.net/docs/package_reference/cross_encoder/model.html)
- [BEIR 数据与格式](https://github.com/beir-cellar/beir)
