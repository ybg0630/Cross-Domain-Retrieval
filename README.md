# Cross-Domain Hybrid Retrieval

Retrieval Studio is an English-language Streamlit app for searching SciFact, NFCorpus, and FiQA. It supports BM25, MiniLM semantic search, hybrid retrieval, full reranking, and adaptive reranking.

## Run the demo

Use Python 3.11. From the repository root on the `ui-demo` branch, run:

```bash
python -m pip install -r requirements-demo.txt
python prepare_demo.py
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. Setup downloads the BEIR corpora and model weights and builds the document indexes. To prepare only SciFact, run `python prepare_demo.py --datasets scifact` in place of the second command. After setup, the app runs locally without an API key.

- **Search Explorer** shows ranked documents and the reranking decision. Use **Generate Answer from Top 3 Documents** to see an answer and its source passages.
- **Method Comparison** compares five methods on the same query.
- **Experiment Dashboard** shows held-out test results and lets you download the selected dataset's summary as CSV.

The [adaptive reranking experiments](member5/README.md) include reproduction steps and explain the evaluation and timing setup.
