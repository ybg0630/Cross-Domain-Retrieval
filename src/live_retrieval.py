"""Local BEIR search backend for the Streamlit presentation."""

import csv
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock

import numpy as np
from rank_bm25 import BM25Okapi

from member3.hybrid import weighted_sum
from rerank import CrossEncoderScorer, document_text
from src.adaptive import search as adaptive_search

ROOT = Path(__file__).resolve().parents[1]
# The demo uses PyTorch models in an environment that also has TensorFlow/Keras.
os.environ["USE_TF"] = "0"
DATASETS = ("scifact", "nfcorpus", "fiqa")
TOP_K = 100
MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
    "cross-encoder": ("cross-encoder/ms-marco-MiniLM-L6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a"),
    "flan-t5-small": ("google/flan-t5-small", "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"),
}
BENCHMARK = ROOT / "member5" / "outputs" / "combined_test_summary.csv"
FUSION = ROOT / "member3" / "report" / "default_fusion.json"


@lru_cache(maxsize=1)
def default_weights():
    config = json.loads(FUSION.read_text(encoding="utf-8"))
    return {name: values["alpha"] for name, values in config["datasets"].items()}


@lru_cache(maxsize=1)
def calibrated_policies():
    names = {"overlap": "top10_overlap", "gap": "normalized_score_gap", "query_length": "query_length"}
    with BENCHMARK.open(encoding="utf-8", newline="") as handle:
        return {
            row["dataset"]: (names[row["selected_signal"]], float(row["threshold"]), row["direction"])
            for row in csv.DictReader(handle) if row["method"] == "Adaptive rerank"
        }


class LocalBackend:
    def __init__(self, embedding_device="cpu"):
        self.embedding_device = embedding_device
        self.lock = RLock()
        self.indexes = {}
        self.encoder = None
        self.scorer = CrossEncoderScorer(model=str(ROOT / "cache" / "cross-encoder"), device="cpu")

    def _encode(self, texts, progress=False):
        import torch
        from sentence_transformers import SentenceTransformer

        if self.encoder is None:
            torch.set_num_threads(4)
            self.encoder = SentenceTransformer(str(ROOT / "cache" / "minilm"),
                                               device=self.embedding_device, local_files_only=True)
        return self.encoder.encode(texts, prompt="", normalize_embeddings=True, convert_to_numpy=True,
                                   batch_size=64, show_progress_bar=progress)

    def _vectors(self, corpus_path, texts, progress):
        with corpus_path.open("rb") as corpus:
            digest = hashlib.file_digest(corpus, "sha256").hexdigest()[:16]
        revision = MODELS["minilm"][1][:12]
        path = ROOT / "cache" / "indexes" / f"{corpus_path.parent.name}-{revision}-{digest}.npy"
        if path.is_file():
            vectors = np.load(path, mmap_mode="r", allow_pickle=False)
            if vectors.shape == (len(texts), 384) and vectors.dtype == np.float32:
                return vectors
            raise ValueError(f"Invalid vector cache format. Rebuild {path.name}.")
        path.parent.mkdir(parents=True, exist_ok=True)
        vectors = self._encode(texts, progress)
        with NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as handle:
            temp = Path(handle.name)
        try:
            np.save(temp, vectors, allow_pickle=False)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def prepare_dataset(self, dataset, progress=False):
        with self.lock:
            if dataset in self.indexes:
                return
            if dataset not in DATASETS:
                raise ValueError(f"Unknown dataset: {dataset}")
            path = ROOT / "datasets" / dataset / "corpus.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"The {dataset} corpus is missing. Run prepare_demo.py to download it.")
            with path.open(encoding="utf-8") as handle:
                corpus = {str(doc["_id"]): doc for doc in map(json.loads, handle)}
            ids = np.array(list(corpus))
            texts = [document_text(doc) for doc in corpus.values()]
            vectors = self._vectors(path, texts, progress)
            bm25 = BM25Okapi([text.lower().split() for text in texts])
            documents = {did: {"title": doc.get("title") or f"{dataset.upper()} · {did}",
                               "abstract": doc.get("text", "")} for did, doc in corpus.items()}
            self.indexes[dataset] = (ids, dict(zip(ids, texts, strict=True)), documents, bm25, vectors)

    def warm_models(self, rerank=False):
        with self.lock:
            if self.encoder is None:
                self._encode(["Warmup query"])
            if rerank:
                self.scorer.prepare(("Warmup query", "Warmup document"))

    def retrieve_single(self, query, dataset, mode):
        ids, _, documents, bm25, vectors = self.indexes[dataset]
        if mode == "bm25":
            scores = bm25.get_scores(query.lower().split())
            top = np.argsort(scores)[-TOP_K:][::-1]
        else:
            with self.lock:
                query_vector = self._encode([query])[0]
            scores = vectors @ query_vector
            top = np.lexsort((ids, -scores))[:TOP_K]
        return {str(ids[i]): float(scores[i]) for i in top}, documents

    def retrieve(self, query, dataset, alpha):
        bm25, documents = self.retrieve_single(query, dataset, "bm25")
        dense, _ = self.retrieve_single(query, dataset, "dense")
        return {"bm25": bm25, "dense": dense,
                "hybrid": weighted_sum(bm25, dense, alpha, top_k=TOP_K), "documents": documents}

    def rerank(self, query, dataset, doc_ids):
        _, texts_by_id, _, _, _ = self.indexes[dataset]
        pairs = [(query, texts_by_id[doc_id]) for doc_id in doc_ids]
        with self.lock:
            scores = self.scorer.predict(pairs, batch_size=32)
        return dict(zip(doc_ids, scores, strict=True))

    def search(self, query, dataset, mode, alpha, threshold):
        self.prepare_dataset(dataset)
        if mode != "bm25":
            self.warm_models(rerank=mode in {"hybrid_rerank", "adaptive"})
        policy = calibrated_policies()[dataset] if mode == "adaptive" and threshold is None else None
        return adaptive_search(query, dataset, mode, alpha, threshold, backend=self, policy=policy)
