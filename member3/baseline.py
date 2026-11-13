import argparse
import os
import json
import numpy as np
from rank_bm25 import BM25Okapi
from beir.datasets.data_loader import GenericDataLoader
from src import utils

available_datasets = ["scifact", "nfcorpus", "fiqa"]
parser = argparse.ArgumentParser(description="Run the BM25 retrieval baseline.")
parser.add_argument("--datasets", nargs="+", choices=available_datasets, default=available_datasets)
parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
args = parser.parse_args()
output_dir = "member3/results" if args.split == "test" else f"member3/results/{args.split}"
os.makedirs(output_dir, exist_ok=True)

for dataset in args.datasets:
    corpus, queries, qrels = utils.load_dataset(dataset)
    if args.split != "test":
        _, queries, qrels = GenericDataLoader(data_folder=f"datasets/{dataset}").load(split=args.split)
    doc_ids = list(corpus.keys())
    doc_texts = list(corpus.values())

    def tokenize(text):
        return text.lower().split()


    tokenized_corpus = [tokenize(doc) for doc in doc_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    run_results = {}

    timer = utils.Timer()
    timer.start()

    for qid, query_text in queries.items():
        tokenized_query = tokenize(query_text)
        scores = bm25.get_scores(tokenized_query)

        # Get the indices of the top 100 scores
        top_100_idx = np.argsort(scores)[-100:][::-1]

        run_results[qid] = {}
        for idx in top_100_idx:
            doc_id = doc_ids[idx]
            score = float(scores[idx])
            # Store the score in the format: run = {query_id: {doc_id: score}}
            run_results[qid][doc_id] = score

    elapsed_time = timer.stop()

    # Efficiency metric: Avg latency per query
    avg_latency = elapsed_time / len(queries)

    output_file = f"{output_dir}/{dataset}_bm25.json"
    with open(output_file, "w") as f:
        json.dump(run_results, f, indent=4)


    metrics = utils.evaluate(qrels, run_results)
    print(f"{dataset} ({args.split}): {metrics}", flush=True)
    print(f"Efficiency (Avg Latency per query): {avg_latency:.2f} ms\n")
