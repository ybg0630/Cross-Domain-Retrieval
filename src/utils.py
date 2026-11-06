import os
import random
import time
import numpy as np
import pytrec_eval
from beir import util
from beir.datasets.data_loader import GenericDataLoader

random.seed(42)
np.random.seed(42)

def load_dataset(name: str):
    """
    Downloads and loads the BEIR dataset, returning corpus, queries, and qrels.
    Name must be one of: 'scifact', 'nfcorpus', 'fiqa'.
    """
    out_dir = os.path.join(os.getcwd(), "datasets")
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    
    # Aggregates title and text for the documents
    processed_corpus = {}
    for doc_id, doc in corpus.items():
        title = doc.get("title", "").strip()
        text = doc.get("text", "").strip()
        
        if title:
            processed_corpus[doc_id] = f"{title} {text}"
        else:
            processed_corpus[doc_id] = text
            
    return processed_corpus, queries, qrels

def evaluate(qrels, run):
    """
    Evaluates the retrieval run against qrels.
    Returns a dictionary containing the project's required metrics: nDCG@10 and Recall@100.
    """
    # pytrec_eval expects qrels and run formats to be {query_id: {doc_id: score}}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut_10', 'recall_100'})
    results = evaluator.evaluate(run)
    
    ndcg_10 = 0.0
    recall_100 = 0.0
    
    for query_id in results.keys():
        ndcg_10 += results[query_id]["ndcg_cut_10"]
        recall_100 += results[query_id]["recall_100"]
        
    num_queries = len(results)
    
    return {
        "nDCG@10": ndcg_10 / num_queries if num_queries > 0 else 0.0,
        "Recall@100": recall_100 / num_queries if num_queries > 0 else 0.0
    }

class Timer:
    """
    Used as a context manager (with Timer() as t:) or manually (t.start(), t.stop()).
    """
    def __init__(self):
        self.start_time = None
        self.elapsed_ms = None

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        if self.start_time is None:
            raise RuntimeError("Timer must be started before calling stop().")
        end_time = time.perf_counter()
        self.elapsed_ms = (end_time - self.start_time) * 1000
        return self.elapsed_ms

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
