# 成员2：Dense Retrieval

使用 MiniLM 和 BGE 在 SciFact、NFCorpus、FiQA 上进行精确稠密检索，复用成员1的 `src/utils.py`，不修改已有 baseline。

## 安装与运行

推荐 Python 3.12。从仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-member2.txt
python src/dense.py --datasets scifact --models minilm --device cpu
# 完整测试：两个模型、三个数据集
python src/dense.py --datasets scifact nfcorpus fiqa --models minilm bge --device cpu
```

首次运行需联网下载数据和模型。使用本依赖文件代替原 requirements.txt；pytrec-eval-terrier 提供 pytrec_eval 接口，避免同时安装两个同名实现。运行生成的 datasets/、cache/、.venv/ 请保留本地，不上传 GitHub。

## 方法

- MiniLM：sentence-transformers/all-MiniLM-L6-v2，最大长度256。
- BGE：BAAI/bge-small-en-v1.5，最大长度512，仅查询添加 `Represent this sentence for searching relevant passages: `。
- 模型 revision 固定在代码中；文档使用标题加正文。
- 向量 L2 归一化，以点积计算余弦相似度；遍历完整语料，输出 top-100，使用公共 utils.evaluate 评估。
- 缓存包含文档ID、向量和配置/文本指纹，重跑时验证后复用。

## 提交结果与下游使用

- `results/{dataset}_dense_{minilm|bge}.json`：六个官方 test 检索结果。格式为 `{query_id: {document_id: score}}`，每条查询100个文档，保留原始ID，可直接交给混合检索模块。融合时按 score 排序，不依赖 JSON 插入次序。
- `results/dense_metrics.csv`：六组质量和耗时汇总，包含模型revision、环境和参数。实验采用 Apple M1 CPU、4线程、batch size 32。
- 运行还会生成每组 `_metrics.json`、`_per_query.csv` 及本地向量缓存；这些可按需再生成，本次不随精简提交上传。

nDCG@10 / Recall@100：

| 数据集 | MiniLM | BGE |
|---|---|---|
| SciFact | 0.6451 / 0.9250 | 0.7127 / 0.9417 |
| NFCorpus | 0.3159 / 0.3115 | 0.3430 / 0.3110 |
| FiQA | 0.3687 / 0.7061 | 0.4035 / 0.6963 |

离线语料编码、批量查询摊销和在线单查询延迟分别记录。在线延迟为模型及文档向量加载后的查询编码、精确检索和top-100耗时，排除下载、加载、评估和写盘；每条test查询测一次，属于本机测量。

下游调参可通过 `--splits train`（SciFact）或 `--splits dev`（NFCorpus、FiQA）另行生成开发结果。请用开发集选择融合参数，test仅用于最终评估。
