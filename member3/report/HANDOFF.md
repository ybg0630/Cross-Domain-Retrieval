# Member 3 fusion handoff

Use **Weighted sum with MiniLM** for the top 100 candidates per query.
Mean train/dev nDCG@10: RRF 0.4251, weighted sum 0.4739.
SciFact uses train. NFCorpus and FiQA use dev. Test labels are used for final reporting.

| Dataset | Selection split | Parameter | Test candidates |
|---|---|---|---|
| SciFact | train | alpha = 0.6 | [scifact_hybrid_wsum.json](../results/scifact_hybrid_wsum.json) |
| NFCorpus | dev | alpha = 0.5 | [nfcorpus_hybrid_wsum.json](../results/nfcorpus_hybrid_wsum.json) |
| FiQA | dev | alpha = 0.7 | [fiqa_hybrid_wsum.json](../results/fiqa_hybrid_wsum.json) |

For weighted sum, alpha is the MiniLM weight. Min-max normalize each retriever's per-query top-100 scores before fusion. Missing scores contribute zero. At alpha 0 or 1, use the original single-retriever ranking.

Sort candidate JSON scores in descending order and break ties by ascending document ID. For reranker calibration, apply the same parameters to the saved train/dev BM25 and MiniLM runs.

See the [selection table](selection_metrics.csv), [test ablation table](test_ablation.csv), [alpha curve](alpha_ndcg.png), and [test ablation chart](test_ablation.png).

Reproduce from the repository root with python -m member3.report (Matplotlib required).
