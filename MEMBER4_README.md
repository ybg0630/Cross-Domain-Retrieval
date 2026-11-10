# 成员 4：已接入 s1 / s2 / s3 的重排序

主入口 `run_member4.py` 自动读取成员 3 冻结的配置，完成候选准备、重排序、缓存、统一评估、计时和逐查询分析。

## 目录与分支

完整仓库：`L:\ntu_class\6405\project\Cross-Domain-Retrieval`，本地集成分支 `s4-integration`。
s1 是 `main`；s2 是 `member2/dense-retrieval`；s3 是 `member3/hybrid-retrieval`，均已拉到本地。
集成分支以 s3 为基础（已经包含 s1/s2），加入 s4 模块。`project/code_4` 同步保存 s4 代码，可单独打包。
从 `code_4` 运行时会自动寻找旁边的 `Cross-Domain-Retrieval`，或用 `--repo-root` 指定仓库。

## 安装和运行

本机环境已安装，在仓库根目录执行：

```powershell
conda activate L:\ntu_class\6405\project\.envs\ee6405-member4
cd L:\ntu_class\6405\project\Cross-Domain-Retrieval
python -m unittest -v
python run_member4.py --check-only
python run_member4.py --limit 3 --device cpu
```

如果 conda 命令启动异常，可以直接调用环境内 Python：

```powershell
& 'L:\ntu_class\6405\project\.envs\ee6405-member4\python.exe' run_member4.py --check-only
```

队友在 Mac/Linux/其他 Windows 上，从仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate ee6405-member4
python run_member4.py --download --limit 3 --device cpu
```

已有环境则用 `python -m pip install -r requirements-member4.txt`；单独使用 `code_4` 时文件名是 `requirements.txt`。
使用 `pytrec-eval-terrier` 提供 `pytrec_eval`，与成员 2 保持一致，不要同时安装原版 `pytrec_eval`。
无需安装根目录 requirements 中的全部前端依赖。版本锁文件是本机 Windows/Python 3.11 的记录，Mac 优先用 environment.yml。

`--download` 下载缺失的 BEIR 数据到仓库 `datasets/`；模型首次使用时自动下载。原始数据已经在本机备好。

正式运行：

```bash
# 全部三个数据集的 test + 校准集，共六组。
python run_member4.py --device cpu

# 只跑 SciFact test。
python run_member4.py --datasets scifact --splits test --device cpu

# 给成员 5 生成阈值选择用的校准结果。
python run_member4.py --splits calibration --device cpu

# 重新推理全部候选，在独立目录测正式耗时。
python run_member4.py --refresh --output-dir benchmark_cpu --device cpu
```

可选 `--batch-size 32`、`--threads 4`、`--max-length 512`、`--revision 模型commitID`、`--repo-root 仓库路径`。
设备支持 cpu/cuda/mps；CUDA 需相容的 PyTorch，Apple Silicon 可尝试 mps。已实际验证 Windows CPU 和 CUDA，尚未实测 Mac。

`--limit N` 按查询 ID 取前 N 条，仅供联调；强制输出到 `smoke_N` 子目录，报告标记 `smoke_only=true`，避免覆盖全量结果。
`--check-only` 核验完整候选、文档 ID 和基线指标，不运行重排序模型。

## 上游接口

成员 1 的实际格式和接口：

```python
from src import utils
corpus, queries, qrels = utils.load_dataset("scifact")  # 固定读取 test
# corpus = {"文档ID": "拼好的标题和正文"}
# queries = {"查询ID": "查询文本"}
# qrels = {"查询ID": {"文档ID": 相关性等级}}
metrics = utils.evaluate(qrels, run)  # nDCG@10、Recall@100
with utils.Timer() as timer:
    ...
elapsed_ms = timer.elapsed_ms
```

s4 同时兼容字符串文档与 BEIR 的 title/text 字典。test 直接使用 s1 加载函数；train/dev 使用相同 BEIR 读取器和文本拼接规则。
集成入口的全部指标和计时分别调用 `src.utils.evaluate`、`src.utils.Timer`，不修改 s1 源码。

成员 3 的配置入口：`member3/report/default_fusion.json`。当前设置：

| 数据集 | MiniLM 权重 alpha | 校准划分 | test 查询数 | 校准查询数 |
|---|---:|---|---:|---:|
| SciFact | 0.6 | train | 300 | 809 |
| NFCorpus | 0.5 | dev | 323 | 324 |
| FiQA | 0.7 | dev | 648 | 500 |

- test：读取配置指定的 `member3/results/{dataset}_hybrid_wsum.json`。
- 校准：读取 `member3/results/train/` 或 `dev/` 下的 BM25 和 MiniLM 结果，直接调用 s3 的 `weighted_sum`，应用已选 alpha，不再调参；也支持配置选择 RRF。
- 候选格式 `{query_id: {doc_id: score}}`，每条 100 篇，分数降序、同分按文档 ID 升序。
- 查询 ID 必须与该划分 qrels 完全一致，候选文档必须存在。先检查完整输入，再截取联调样本。

## 下游产物

默认输出到仓库根目录，或 `--output-dir` 指定目录。以 SciFact test 为例：

| 路径 | 内容 |
|---|---|
| `results/scifact_hybrid_rerank.json` | 本次重排结果，原始模型分数越大越相关 |
| `results/scifact_rerank_candidates.json` | 本次混合候选，供前后对照 |
| `cache/rerank_scores_scifact_test.json` | 本次候选得分快照，成员 5 可直接读取 |
| `cache/rerank_scores_scifact_test.sqlite3` | 可续跑、含历史候选的缓存 |
| `reports/member4/scifact_metrics.json` | 总体前后指标与变好/变差/不变比例 |
| `reports/member4/scifact_delta.csv` | query_id、ndcg_before、ndcg_after、delta、status |
| `reports/member4/scifact_delta.png` | Δ 分布和变化比例图 |
| `reports/member4/scifact_timing.csv` | 每查询推理耗时、新算/缓存对数、batch size |
| `reports/member4/scifact_timing.json` | 硬件、模型参数、输入文件哈希、平均耗时 |
| `reports/member4/history/` | 每次调用的计时归档 |
| `reports/member4/summary.csv` | 本次命令的各数据集/划分汇总 |

校准集的结果/报告前缀加实际划分，例如 `scifact_train_hybrid_rerank.json`、`nfcorpus_dev_hybrid_rerank.json`；缓存名称始终包含 split。

成员 5 可直接使用 JSON：

```python
import json
from pathlib import Path

full_run = json.loads(Path("results/scifact_hybrid_rerank.json").read_text(encoding="utf-8"))
hybrid_run = json.loads(Path("results/scifact_rerank_candidates.json").read_text(encoding="utf-8"))
# should_rerank 由成员 5 的策略实现，阈值在 train/dev 上选择。
adaptive_run = {q: full_run[q] if should_rerank(q) else hybrid_run[q] for q in full_run}
```

读取历史 SQLite：`from rerank import read_cached_scores`，再调用 `read_cached_scores(path)` 得到同格式字典。
历史缓存可能多于当前候选，评估和自适应模拟优先读本次结果 JSON。

Python 函数接口：

```python
from rerank import CrossEncoderScorer, rerank
from run_member4 import team_measure
from src import utils

run, timing_rows, metadata = rerank(
    corpus, queries, candidates, "cache/custom.sqlite3",
    CrossEncoderScorer(device="cpu"), dataset="scifact",
    batch_size=32, measure=team_measure(utils),
)
# analyze(qrels, baseline, run, utils.evaluate) -> (summary, per_query_rows)
```

从独立 `code_4` 的父目录调用，改为 `from code_4.rerank import ...`。重排序函数不接收 qrels，标注仅用于评估。
原有 `rerank.py`、`analyze.py` 独立 CLI 仍可用；组内正式运行优先用 `run_member4.py`，自动接入统一计时器和全部输出。

## 缓存与耗时口径

模型 `cross-encoder/ms-marco-MiniLM-L6-v2`，输出原始 logits，默认最大联合长度 512 tokens。
每条查询成功后提交缓存；新候选只补算新对，文本变化重算相关对，模型配置变化需换缓存文件。不要多个进程同时写同一缓存。

`inference_ms` 包含分词、模型推理和分数回传 CPU，排除模型加载/预热、上游检索、缓存读写和绘图。
`ms_per_inferred_pair` 只除以实际新算对数；只要有缓存命中，`mean_full_rerank_ms=null`。
正式计时用 `--refresh`，统一硬件、batch size、线程数和长度。s2 的 Apple M1 数据不能与本机 Windows CPU 耗时直接拼成同机对比。

最新计时在 `_timing.*`，历史记录在 `history/`，缓存重跑不会抹掉历史计时。仅使用 `full_fresh_pass=true` 且输入来源/参数匹配的记录比较完整推理时间。
`summary.csv` 只汇总本次命令；分次运行的各组明细仍保留。Recall@100 在同一批 top-100 仅重排时应不变。本版未做可选的 top-20/50 消融。

## 验证状态

2026-11-10：六组完整重排序已完成，共 2,904 条查询、290,400 对候选，全部实际推理。硬件为 RTX 3070 Ti Laptop GPU（8 GB），PyTorch 2.14.1+cu126，batch size 32、CPU 线程 4、最大长度 512，模型 revision 固定为 233902d25c440f23af6f7d6e94d2946bac0bee0a。

18 项自动测试通过；三个 test / 校准集 ID 均不重叠。实验原始产物位于本机 project/member4_full_20261110；成员 5 交付包为 project/member5_handoff_20261110.zip，包内 README 和 check_handoff.py 提供读取方法及完整性检查。所有校准集和 test 的逐查询 Δ、缓存、耗时均提供。

三个 test 集 nDCG@10：SciFact 0.6815 → 0.6876，NFCorpus 0.3256 → 0.3453，FiQA 0.3820 → 0.3730。FiQA 全量重排出现下降，保留真实结果，不根据 test 标签改策略。

本机独立环境已经切换到 CUDA 版 PyTorch。若重建 GPU 环境，先安装 requirements-member4.txt，再运行 `python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126`。Mac 仍使用 environment.yml，读取交付包不需要 CUDA 或模型。
