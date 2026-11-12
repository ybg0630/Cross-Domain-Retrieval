"""Load the unchanged S1 evaluate function without S1's unused BEIR imports.

The local analysis uses saved JSON, so loading all of src.utils would require
Torch just for its BEIR loader. This compiles only the original evaluate AST
node and supplies the same pytrec_eval module. No metric code is copied here.
"""

import ast
import hashlib
from pathlib import Path

import pytrec_eval


def team_evaluator():
    source = Path(__file__).resolve().parents[1] / "src" / "utils.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate"]
    if len(nodes) != 1:
        raise RuntimeError("Expected one src.utils.evaluate function")
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec")
    namespace = {"pytrec_eval": pytrec_eval}
    exec(code, namespace)
    return namespace["evaluate"], hashlib.sha256(source.read_bytes()).hexdigest()
