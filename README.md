# Cross-Domain Hybrid Retrieval Project

This repository contains the implementation for the Cross Domain Retrieval Project.
This project evaluates sparse retrieval, dense retrieval, hybrid fusion and full reranking across 3 BEIR datasets (SciFact, NFCorpus, FiQA).
An adaptive strategy to optimize retrieval quality versus processing latency is developed.
Random seed of 42 is used throughout the project.
Quality metrics including nDCG@10 and Recall@100 are handled using utils.py.


Environment Setup
1. Close the repository and navigate to root directory.
2. Install dependencies from the requirements.txt
pip install -r requirements.txt

