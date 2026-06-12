import ast
from pathlib import Path


SOURCE = Path("modal_funasr_infer.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def test_funasr_model_and_language_are_vietnamese_multilingual():
    assignments = {
        node.targets[0].id: node.value.value
        for node in TREE.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }

    assert assignments["MODEL_ID"] == "FunAudioLLM/Fun-ASR-MLT-Nano-2512"
    assert assignments["LANGUAGE"] == "越南语"


def test_generate_passes_configured_language():
    generate_calls = [
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "generate"
    ]

    assert generate_calls, "Expected a FunASR model.generate(...) call"
    language_keywords = [
        keyword
        for call in generate_calls
        for keyword in call.keywords
        if keyword.arg == "language"
    ]

    assert language_keywords, "model.generate(...) must pass language=LANGUAGE"
    assert isinstance(language_keywords[0].value, ast.Name)
    assert language_keywords[0].value.id == "LANGUAGE"
