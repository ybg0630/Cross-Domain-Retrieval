# Member 5 — Adaptive Reranking

This module studies whether query-level selection can reduce Cross-Encoder reranking work without substantially hurting retrieval quality. It uses the Hybrid top-100 candidates from Member 3 and the saved full reranking results and per-query timings from Member 4. The experiments are **offline**: they reuse genuine model outputs rather than running the Cross-Encoder again.

## Method

For each query, we compute three inexpensive signals before using any reranker scores:

- **Top-10 overlap:** fraction of document IDs shared by the BM25 and Dense top-10 lists.
- **Normalized Hybrid score gap:** difference between the top two Hybrid scores divided by the range of scores in its candidate list.
- **Query length:** number of whitespace-separated words in the query.

A threshold rule decides whether to use the full Cross-Encoder ranking or keep the Hybrid ranking for the entire query. We test both threshold directions. The rule is selected on **SciFact train / NFCorpus dev / FiQA dev**, with a maximum reranking budget of 50% of queries, and then kept fixed for each dataset's held-out **test** split.

We compare **No rerank**, **Full rerank**, **Adaptive rerank**, and a **Random rerank** baseline that selects exactly the same number of queries as Adaptive (20 seeds, 42–61). We report nDCG@10, Recall@100, reranked-query fraction, and the additional Cross-Encoder inference time measured by Member 4.

## Held-out test results

| Dataset  | Hybrid nDCG@10 | Full rerank | Adaptive | Random (20-seed mean) | Adaptive reranked queries |
| -------- | -------------: | ----------: | -------: | --------------------: | ------------------------: |
| SciFact  |         0.6815 |      0.6876 |   0.6835 |                0.6837 |           142/300 (47.3%) |
| NFCorpus |         0.3256 |      0.3453 |   0.3344 |                0.3338 |           138/323 (42.7%) |
| FiQA     |         0.3820 |      0.3730 |   0.3823 |                0.3786 |           259/648 (40.0%) |

Adaptive reduces the number of queries sent to the Cross-Encoder, while preserving reasonably competitive ranking quality. However, its advantage over matched-budget random query selection is **not consistent across datasets**. Full reranking itself lowers nDCG@10 on FiQA. Recall@100 is unchanged because all methods use the same top-100 candidate documents per query.

The reported timing is **additional Cross-Encoder inference time only**, not end-to-end search latency. The selection rules were tuned separately by dataset; cross-domain transfer was not established.

## Files

- `adaptive.py` — signals, threshold decisions, and selection of Hybrid or Full rankings.
- `evaluator.py` — uses the existing `src/utils.py` evaluation function for saved results.
- `run_experiments.py` — calibration, fixed-policy test evaluation, random baselines, CSV outputs, and diagnostic plots.
- `plot_final_summary.py` — regenerates the three concise final comparison plots from `outputs/combined_test_summary.csv`.
- `test_adaptive.py` — unit tests for decision logic and an adapter test (requires the S5-owned `src/adaptive.py`).
- `outputs/` — aggregate, per-query, threshold-sweep, random-seed, and provenance results.
- `figures/` — generated plots for analysis and presentation.

## Reproducing the experiments

Run commands from the repository root. Install the analysis dependencies in your Python environment:

```bash
python -m pip install -r member5/requirements.txt
```

The default input location is `member5/inputs/s4_handoff_20261110/`. This directory is ignored by Git. Its contents come from Member 4's handoff archive at `origin/s4:deliverables/member5_handoff_20261110.zip`; it must contain `inputs/`, `results/`, `reports/member4/`, `MANIFEST.json`, and `SHA256SUMS.txt` directly under that location. The S3 output files under `member3/results/` and the team's `src/utils.py` must also be present. If the handoff is not available, the experiment cannot be rerun from the summary CSV alone.

With those inputs present:

```bash
python -m member5.run_experiments --dataset all
python member5/plot_final_summary.py
```

To run unit tests, make sure the S5-owned `src/adaptive.py` adapter is present:

```bash
python -m unittest -v member5.test_adaptive
```

The calibration runs produce `threshold_sweep.csv`; the test results are reported in `outputs/combined_test_summary.csv`. Files named `test_curve_exploratory_only.csv` and the corresponding threshold-sweep plots are **exploratory**. Test-set threshold sweeps were not used to select the final policy or to report its held-out performance.

The `*_provenance.json` files record the selection rule, input sources, evaluation hash, and timing scope. Paths in these metadata files are relative to the repository so they do not expose machine-specific user directories.