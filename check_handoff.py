"""在交付包根目录运行：python -B check_handoff.py [--evaluate]。"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from member5_io import choose_results, estimate_cost, load_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true", help="另外使用随包 s1 utils 复核总体指标")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        file = (root / relative).resolve()
        if not file.is_relative_to(root) or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise ValueError(f"文件缺失或已变化: {relative}")
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    if args.evaluate:
        sys.path.insert(0, str(root / "source"))
        from src.utils import evaluate
    for item in manifest["splits"]:
        bundle = load_bundle(item["dataset"], item["split"], root)
        assert len(bundle["queries"]) == item["queries"]
        assert all(len(docs) == 100 for docs in bundle["reranked"].values())
        assert choose_results(bundle, []) == bundle["hybrid"]
        assert choose_results(bundle, bundle["queries"]) == bundle["reranked"]
        assert estimate_cost(bundle, [])["mean_extra_rerank_ms"] == 0
        assert math.isclose(estimate_cost(bundle, bundle["queries"])["mean_extra_rerank_ms"],
                            item["mean_full_rerank_ms"], rel_tol=1e-10)
        if args.evaluate:
            for key, expected in (("hybrid", item["baseline"]), ("reranked", item["reranked"])):
                actual = evaluate(bundle["qrels"], bundle[key])
                assert all(math.isclose(actual[k], v, abs_tol=1e-12) for k, v in expected.items())
        print(f"PASS {item['dataset']}/{item['split']}: {item['queries']} queries")
    print("All file checksums, six splits and member 5 interfaces passed.")


if __name__ == "__main__":
    main()
