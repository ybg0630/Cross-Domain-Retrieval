# Cross-Domain Hybrid Retrieval Project

## s4: completed full reranking experiments

All six test/calibration runs are complete: **2,904 queries / 290,400 query-document pairs**.
Inference used an RTX 3070 Ti Laptop GPU, batch size 32, max length 512 and the frozen Member 3 fusion settings.

- **[Member 5 handoff ZIP (44.7 MiB)](deliverables/member5_handoff_20261110.zip)**: scores, SQLite/JSON caches, queries/qrels, BM25/MiniLM inputs, reports and reproducible source.
- [Download and usage guide](deliverables/README.md)
- [Full experiment summary](reports/member4/full_summary.csv) / [test overview chart](reports/member4/test_overview.png)
- [Member 4 interfaces and reproduction](MEMBER4_README.md)

| Test dataset | Hybrid nDCG@10 | Reranked nDCG@10 | Mean extra rerank latency |
|---|---:|---:|---:|
| SciFact | 0.6815 | 0.6876 | 239.0 ms/query |
| NFCorpus | 0.3256 | 0.3453 | 237.3 ms/query |
| FiQA | 0.3820 | 0.3730 | 177.1 ms/query |

Recall@100 is unchanged in all six runs. FiQA quality decreased; the measured results are preserved as-is.
Latency is additional GPU reranking time, excluding upstream retrieval. It should not be directly combined with the earlier Mac CPU measurements.

For s4, install `requirements-member4.txt` (rather than the generic requirements below).
Member 5 can use the exported results without downloading the model or installing CUDA.

---

This repository contains the implementation for the Cross Domain Retrieval Project.
This project evaluates sparse retrieval, dense retrieval, hybrid fusion and full reranking across 3 BEIR datasets (SciFact, NFCorpus, FiQA).
An adaptive strategy to optimize retrieval quality versus processing latency is developed.
Random seed of 42 is used throughout the project.
Quality metrics including nDCG@10 and Recall@100 are handled using utils.py.


Environment Setup
1. Close the repository and navigate to root directory.
2. Install dependencies from the requirements.txt
pip install -r requirements.txt

## Member 4: integrated reranking

See [MEMBER4_README.md](MEMBER4_README.md) for the s1/s2/s3 interfaces, setup and outputs.
Install `requirements-member4.txt` in a separate environment, then run
`python run_member4.py --download --limit 3 --device cpu` for an integration smoke test.
Omit `--limit` to run all test and calibration queries.

