"""Reproducible exact dense retrieval. Run from the repository root.

The upstream utils.py and baseline.py are deliberately kept unchanged.
Embeddings are cached by content and model configuration, not by filename alone.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import tempfile
from datetime import datetime, timezone

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader

try:
    from . import utils
except ImportError:
    import utils

ROOT = Path(__file__).resolve().parents[1]
MODELS = {'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
          'bge': 'BAAI/bge-small-en-v1.5'}
MODEL_REVISIONS = {'minilm': '1110a243fdf4706b3f48f1d95db1a4f5529b4d41', 'bge': '5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'}
BGE_PREFIX = 'Represent this sentence for searching relevant passages: '


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False, suffix='.tmp') as f:
        json.dump(obj, f, ensure_ascii=False, allow_nan=False, indent=2)
        temp = f.name
    os.replace(temp, path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False, suffix='.tmp', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
        temp = f.name
    os.replace(temp, path)


def fingerprint(ids, texts):
    digest = hashlib.sha256()
    for identifier, text in zip(ids, texts, strict=True):
        digest.update(json.dumps([identifier, text], ensure_ascii=False).encode())
        digest.update(b'\n')
    return digest.hexdigest()


def load_data(dataset, split='test'):
    """Reuse the public loader. Read extra official splits without editing it."""
    corpus, queries, qrels = utils.load_dataset(dataset)
    if split != 'test':
        folder = Path.cwd() / 'datasets' / dataset
        if not (folder / 'qrels' / f'{split}.tsv').exists():
            raise ValueError(f'{dataset} has no official {split} split')
        _, queries, qrels = GenericDataLoader(data_folder=str(folder)).load(split=split)
    # Evaluate every labelled query, never silently omit queries with bad results.
    missing = set(qrels) - set(queries)
    if missing:
        raise ValueError(f'Queries missing text: {sorted(missing)[:5]}')
    queries = {qid: queries[qid] for qid in sorted(qrels)}
    return corpus, queries, qrels


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
    elif str(device) == 'mps':
        torch.mps.synchronize()


def timed(fn, device='cpu'):
    synchronize(device)
    timer = utils.Timer()
    timer.start()
    value = fn()
    synchronize(device)
    return value, timer.stop()


def versions():
    return {name: importlib.metadata.version(name) for name in
            ['numpy', 'torch', 'sentence-transformers', 'transformers', 'beir', 'pytrec-eval-terrier']}


def seed_all(seed=42, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def validate_embeddings(vectors, count):
    if vectors.ndim != 2 or vectors.shape[0] != count or not np.isfinite(vectors).all():
        raise ValueError('Invalid embedding shape or non-finite embeddings')
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=2e-4):
        raise ValueError('Embeddings must be nonzero and L2 normalized')


def cached_encode(model, ids, texts, directory, kind, config, batch_size, device):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    key = {**config, 'content_sha256': fingerprint(ids, texts), 'kind': kind,
           'count': len(ids)}
    prefix = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:20]
    data_path = directory / f'{kind}_{prefix}.npz'
    meta_path = directory / f'{kind}_{prefix}.json'
    if data_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['key'] != key:
            raise ValueError('Cache metadata mismatch')
        with np.load(data_path, allow_pickle=False) as data:
            if data['ids'].tolist() != ids:
                raise ValueError('Cache ID order mismatch')
            vectors = data['embeddings']
        validate_embeddings(vectors, len(ids))
        return vectors, meta, True, str(data_path.relative_to(ROOT))
    # Disable any model-provided default prompt. Input text is prepared explicitly.
    vectors, elapsed = timed(lambda: model.encode(
        texts, batch_size=batch_size, prompt='', normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True), device)
    vectors = np.asarray(vectors, dtype=np.float32)
    validate_embeddings(vectors, len(ids))
    # Count exact tokenizer truncation separately, outside embedding timing.
    truncated = 0
    for start in range(0, len(texts), 256):
        lengths = model.tokenizer(texts[start:start+256], truncation=False,
                                  padding=False, return_length=True, verbose=False)['length']
        truncated += sum(n > model.max_seq_length for n in lengths)
    meta = {'key': key, 'encoding_ms': elapsed, 'batch_size': batch_size,
            'encoding_device': str(device), 'encoding_hardware': platform.platform()+' '+platform.processor(),
            'dimension': vectors.shape[1],
            'truncated_count': truncated, 'truncated_fraction': truncated / len(ids),
            'created_utc': datetime.now(timezone.utc).isoformat()}
    with tempfile.NamedTemporaryFile('wb', dir=directory, delete=False, suffix='.tmp') as f:
        np.savez(f, embeddings=vectors, ids=np.asarray(ids))
        temp = f.name
    os.replace(temp, data_path)
    atomic_json(meta_path, meta)
    return vectors, meta, False, str(data_path.relative_to(ROOT))


def select_top(scores, ids, k=100):
    """Partial selection with explicit handling of ties at the cutoff.

    IDs are strings; lexicographic ascending IDs break tied scores deterministically.
    """
    scores = np.asarray(scores)
    ids = np.asarray(ids)
    valid = np.flatnonzero(np.isfinite(scores))
    k = min(k, len(valid))
    if not k:
        return np.empty(0, dtype=np.int64)
    if len(valid) > k:
        threshold = np.partition(scores[valid], len(valid)-k)[len(valid)-k]
        above = valid[scores[valid] > threshold]
        tied = valid[scores[valid] == threshold]
        tied = tied[np.argsort(ids[tied], kind='stable')[:k-len(above)]]
        chosen = np.concatenate([above, tied])
    else:
        chosen = valid
    return chosen[np.lexsort((ids[chosen], -scores[chosen]))]


def exact_search(query_vectors, doc_vectors, doc_ids, query_ids,
                 top_k=100, doc_chunk_size=20000, query_chunk_size=32):
    """CPU float32 exact cosine search; bounded score matrices and stable ties."""
    doc_ids = np.asarray(doc_ids)
    run = {}
    for qstart in range(0, len(query_ids), query_chunk_size):
        qbatch = query_vectors[qstart:qstart+query_chunk_size]
        candidates = [[] for _ in qbatch]
        for dstart in range(0, len(doc_ids), doc_chunk_size):
            scores = qbatch @ doc_vectors[dstart:dstart+doc_chunk_size].T
            local_ids = doc_ids[dstart:dstart+doc_chunk_size]
            for row in range(len(qbatch)):
                selected = select_top(scores[row], local_ids, top_k)
                candidates[row].extend((str(local_ids[j]), float(scores[row, j])) for j in selected)
        for row, pairs in enumerate(candidates):
            pairs.sort(key=lambda x: (-x[1], x[0]))
            run[query_ids[qstart+row]] = dict(pairs[:top_k])
    return run


def validate_run(run, query_ids, doc_ids, top_k=100):
    if set(run) != set(query_ids):
        raise ValueError('Run query coverage mismatch')
    valid_docs = set(doc_ids)
    for qid, values in run.items():
        if len(values) != min(top_k, len(doc_ids)) or not set(values) <= valid_docs:
            raise ValueError(f'Invalid candidate count or IDs for {qid}')
        if not all(isinstance(v, float) and np.isfinite(v) for v in values.values()):
            raise ValueError(f'Invalid scores for {qid}')


def per_query_metrics(qrels, run):
    return {qid: utils.evaluate({qid: qrels[qid]}, {qid: run[qid]}) for qid in qrels}


def run_experiment(args, dataset, alias, split):
    corpus, queries, qrels = load_data(dataset, split)
    doc_ids = sorted(corpus)
    qids = list(queries)
    model_name = MODELS[alias]
    revision = MODEL_REVISIONS[alias] if args.revision is None else args.revision
    model = SentenceTransformer(model_name, revision=revision, device=args.device,
                                cache_folder=str(ROOT / 'cache' / 'models'),
                                model_kwargs={'attn_implementation': 'sdpa'})
    model.eval()
    device = str(model.device)
    query_prefix = BGE_PREFIX if alias == 'bge' else ''
    doc_texts = [corpus[x] for x in doc_ids]
    query_texts = [query_prefix + queries[x] for x in qids]
    # A warm model is used for both offline encoding and online measurement.
    model.encode(['Warmup document', 'Warmup query'], prompt='', normalize_embeddings=True)
    synchronize(device)
    common = {'model': model_name, 'revision': revision,
              'max_seq_length': model.max_seq_length, 'normalize': True,
              'dtype': 'float32', 'preprocessing': 'upstream title+text, unchanged',
              'versions': versions(), 'attention': 'sdpa'}
    directory = ROOT / 'cache' / dataset / alias
    docs, doc_meta, doc_hit, doc_path = cached_encode(
        model, doc_ids, doc_texts, directory, 'documents', common, args.batch_size, device)
    qvectors, qmeta, qhit, qpath = cached_encode(
        model, qids, query_texts, directory, f'queries_{split}',
        {**common, 'query_prefix': query_prefix, 'split': split}, args.batch_size, device)
    search = lambda q, ids: exact_search(q, docs, doc_ids, ids,
                      doc_chunk_size=args.doc_chunk_size, query_chunk_size=args.query_chunk_size)
    run, search_ms = timed(lambda: search(qvectors, qids))
    validate_run(run, qids, doc_ids)
    # test files stay at the agreed root; validation results have separate directories.
    out = ROOT / 'results' if split == 'test' else ROOT / 'results' / split
    atomic_json(out / f'{dataset}_dense_{alias}.json', run)
    metrics = utils.evaluate(qrels, run)
    individual = per_query_metrics(qrels, run)
    observations = []
    # Query cache is deliberately bypassed here. Search uses the same CPU path.
    for repeat in range(args.latency_repeats):
        for qid, text in zip(qids, query_texts, strict=True):
            vector, encode_ms = timed(lambda: model.encode(
                [text], prompt='', batch_size=1, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False), device)
            _, retrieve_ms = timed(lambda: search(vector, [qid]))
            observations.append({'dataset': dataset, 'split': split, 'model': alias,
                'query_id': qid, 'repeat': repeat, 'query_encode_ms': encode_ms,
                'search_ms': retrieve_ms, 'online_ms': encode_ms+retrieve_ms,
                **individual[qid]})
    write_csv(out / f'{dataset}_dense_{alias}_per_query.csv', observations)
    latency = [x['online_ms'] for x in observations]
    info = {'dataset': dataset, 'split': split, 'model': alias, 'model_name': model_name,
        'revision': revision, 'queries': len(qids), 'documents': len(doc_ids), **metrics,
        'offline_corpus_encode_ms': doc_meta['encoding_ms'],
        'offline_encoding_batch_size': doc_meta['batch_size'],
        'batch_query_encoding_batch_size': qmeta['batch_size'],
        'offline_encoding_device': doc_meta['encoding_device'],
        'batch_query_encode_ms': qmeta['encoding_ms'], 'batch_search_ms': search_ms,
        'batch_amortized_ms_per_query': (qmeta['encoding_ms']+search_ms)/len(qids),
        'online_mean_ms': float(np.mean(latency)) if latency else None,
        'online_p50_ms': float(np.percentile(latency, 50)) if latency else None,
        'online_p95_ms': float(np.percentile(latency, 95)) if latency else None,
        'latency_repeats': args.latency_repeats, 'batch_size': args.batch_size,
        'query_chunk_size': args.query_chunk_size, 'doc_chunk_size': args.doc_chunk_size,
        'encode_device': device, 'search_device': 'cpu numpy float32',
        'hardware': platform.platform() + ' ' + platform.processor(),
        'torch_threads': torch.get_num_threads(),
        'doc_cache_hit': doc_hit, 'query_cache_hit': qhit,
        'doc_cache_path': doc_path, 'query_cache_path': qpath,
        'query_prefix': query_prefix, 'max_seq_length': model.max_seq_length,
        'doc_truncated_fraction': doc_meta['truncated_fraction'],
        'query_truncated_fraction': qmeta['truncated_fraction'],
        'versions': versions(), 'seed': 42, 'warmup_texts': 2, 'attention': 'sdpa',
        'self_match_policy': 'keep all corpus documents, matching upstream BM25',
        'timing_note': 'Cache encode costs reference original measurement; search is measured now. '
                       'Online bypasses query cache; excludes loading/download/write/evaluation.',
        'created_utc': datetime.now(timezone.utc).isoformat()}
    atomic_json(out / f'{dataset}_dense_{alias}_metrics.json', info)
    print(json.dumps({k: info[k] for k in ['dataset','split','model','nDCG@10','Recall@100','online_mean_ms']}, ensure_ascii=False), flush=True)
    return info


def refresh_summary():
    rows = []
    for path in sorted((ROOT / 'results').glob('*_dense_*_metrics.json')):
        row = json.loads(path.read_text())
        row['versions'] = json.dumps(row['versions'], sort_keys=True)
        rows.append(row)
    write_csv(ROOT / 'results' / 'dense_metrics.csv', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=['scifact','fiqa','nfcorpus'], choices=['scifact','fiqa','nfcorpus'])
    parser.add_argument('--models', nargs='+', default=['minilm','bge'], choices=MODELS)
    parser.add_argument('--splits', nargs='+', default=['test'], choices=['test','dev','train'])
    parser.add_argument('--device', default='cpu', choices=['cpu','mps','cuda'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--doc-chunk-size', type=int, default=20000)
    parser.add_argument('--query-chunk-size', type=int, default=32)
    parser.add_argument('--latency-repeats', type=int, default=1)
    parser.add_argument('--revision', default=None)
    args = parser.parse_args()
    if min(args.batch_size, args.threads, args.doc_chunk_size, args.query_chunk_size) <= 0 or args.latency_repeats < 0:
        parser.error('Batch, chunk and thread sizes must be positive; repeats nonnegative')
    if Path.cwd().resolve() != ROOT:
        parser.error('Run from the repository root so the upstream dataset paths stay consistent')
    seed_all(threads=args.threads)
    requested_revision = args.revision
    for alias in args.models:
        # Resolve once per model so all datasets use exactly the same weights.
        args.revision = requested_revision or MODEL_REVISIONS[alias]
        for dataset in args.datasets:
            for split in args.splits:
                run_experiment(args, dataset, alias, split)
                refresh_summary()



class DenseRetriever:
    """Small reusable adapter for the demo, using an existing document cache.

    Example: DenseRetriever('results/scifact_dense_minilm_metrics.json').search('...')
    Loading belongs in the frontend resource cache, outside request timing.
    """
    def __init__(self, metrics_path, device='cpu'):
        meta = json.loads(Path(metrics_path).read_text())
        self.dataset = meta['dataset']
        self.query_prefix = meta['query_prefix']
        self.model = SentenceTransformer(meta['model_name'], revision=meta['revision'],
            device=device, cache_folder=str(ROOT/'cache'/'models'),
            model_kwargs={'attn_implementation': 'sdpa'})
        self.model.eval()
        cache_path = ROOT/meta['doc_cache_path']
        cache_meta = json.loads(cache_path.with_suffix('.json').read_text())
        if (cache_meta['key']['model'] != meta['model_name'] or
                cache_meta['key']['revision'] != meta['revision']):
            raise ValueError('Demo model and document cache configuration differ')
        with np.load(cache_path, allow_pickle=False) as data:
            self.doc_ids = data['ids'].tolist()
            self.doc_vectors = data['embeddings']
        if len(set(self.doc_ids)) != len(self.doc_ids):
            raise ValueError('Duplicate cached document IDs')
        validate_embeddings(self.doc_vectors, len(self.doc_ids))
        self.doc_chunk_size = meta['doc_chunk_size']

    def search(self, query, top_k=100):
        if not query.strip():
            return {'scores': {}, 'latency_ms': 0.0}
        if top_k <= 0:
            raise ValueError('top_k must be positive')
        def retrieve():
            vector = self.model.encode([self.query_prefix+query], prompt='',
                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            synchronize(str(self.model.device))
            return exact_search(vector, self.doc_vectors, self.doc_ids, ['interactive'],
                top_k=top_k, doc_chunk_size=self.doc_chunk_size)['interactive']
        scores, elapsed = timed(retrieve, str(self.model.device))
        return {'scores': scores, 'latency_ms': elapsed}


if __name__ == '__main__':
    main()
