"""Download the pinned local models and prepare the three BEIR indexes."""

import argparse
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
from zipfile import ZipFile

from src.live_retrieval import DATASETS, MODELS, ROOT, LocalBackend

BEIR_BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


def prepare_corpus(dataset):
    directory = ROOT / "datasets"
    target = directory / dataset / "corpus.jsonl"
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as work:
        archive = Path(work) / "dataset.zip"
        urlretrieve(f"{BEIR_BASE}/{dataset}.zip", archive)
        with ZipFile(archive) as source:
            path = Path(work) / target.name
            with source.open(f"{dataset}/corpus.jsonl") as compressed, path.open("wb") as dest:
                shutil.copyfileobj(compressed, dest)
            path.replace(target)


def prepare_models():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    for name, (model, revision) in MODELS.items():
        directory = ROOT / "cache" / name
        snapshot_download(repo_id=model, revision=revision, local_dir=directory,
                          allow_patterns=["*.json", "*.txt", "model.safetensors", "spiece.model"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu",
                        help="Device used to build document vectors. The UI runs on CPU.")
    args = parser.parse_args()
    prepare_models()
    engine = LocalBackend(embedding_device=args.device)
    for dataset in args.datasets:
        print(f"Preparing {dataset} …", flush=True)
        prepare_corpus(dataset)
        engine.prepare_dataset(dataset, progress=True)
        print(f"Ready: {dataset}", flush=True)


if __name__ == "__main__":
    main()
