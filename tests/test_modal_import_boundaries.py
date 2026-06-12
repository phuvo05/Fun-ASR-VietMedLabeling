import ast
from pathlib import Path

TREE = ast.parse(Path("modal_funasr_infer.py").read_text(encoding="utf-8"))


def test_modal_worker_module_does_not_import_local_batch_helpers_at_top_level():
    top_level_imports = [
        node
        for node in TREE.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    assert all(
        not (isinstance(node, ast.ImportFrom) and node.module == "asr_batch")
        for node in top_level_imports
    )
