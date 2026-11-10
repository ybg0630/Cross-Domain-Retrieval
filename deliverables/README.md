# 成员 5 交付入口

下载 [member5_handoff_20261110.zip](member5_handoff_20261110.zip)（44.7 MiB）。GitHub 文件页点击 **Download raw file** 即可下载；解压后先看包内 `README_成员5.md`。

完整 test + 校准集共六组：SciFact test/train、NFCorpus test/dev、FiQA test/dev，共 2,904 条查询和 290,400 个候选对。包内已经包含离线策略所需的数据，Windows 和 Mac 均可读取，无需运行重排序模型。

包含：

- 六组混合候选、重排序结果、JSON/SQLite 得分缓存。
- 对齐查询 ID 的 BM25、MiniLM 结果、查询文本和 qrels。
- 逐查询 Δ、变好/变差/不变统计、逐查询及每对实测耗时、硬件参数和图表。
- `member5_io.py` 读取与成本模拟接口、完整源代码快照、环境和模型版本信息。
- 文件校验值与验证记录。

解压后运行：

```bash
python -B check_handoff.py
# 已有小组评估依赖时，可再复核所有总体指标：
python -B check_handoff.py --evaluate
```

```python
from member5_io import load_bundle, choose_results, estimate_cost

bundle = load_bundle("scifact", "train")
selected_ids = set(bundle["queries"])  # 示例：全量重排对照；替换为你的策略所选 ID。
run = choose_results(bundle, selected_ids)
cost = estimate_cost(bundle, selected_ids)
```

阈值在 train/dev 上选择，test 只用于最终报告。`mean_extra_rerank_ms` 的分母是全部查询数；这是基于逐查询 GPU 实测时间的离线模拟，不含上游检索耗时。不要直接与其他硬件的旧耗时相加。

仓库内可直接查看 [六组结果汇总](../reports/member4/full_summary.csv)、[test 图表](../reports/member4/test_overview.png)、[完整指标与逐查询文件](../reports/member4/) 和 [结果 JSON](../results/)。

压缩包 SHA-256 校验值见 [校验文件](member5_handoff_20261110.zip.sha256)。包内全部文件校验、六组公共评估复算、JSON/SQLite 一致性、计时记录及 test/校准查询不重叠检查均已通过。
