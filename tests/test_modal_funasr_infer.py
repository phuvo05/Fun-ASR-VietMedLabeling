import ast
from pathlib import Path

TREE = ast.parse(Path("modal_funasr_infer.py").read_text(encoding="utf-8"))


def _constant_assignments() -> dict[str, object]:
    return {
        node.targets[0].id: node.value.value
        for node in TREE.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }


def test_funasr_model_and_language_are_vietnamese_multilingual():
    assignments = _constant_assignments()

    assert assignments["MODEL_ID"] == "FunAudioLLM/Fun-ASR-MLT-Nano-2512"
    assert assignments["LANGUAGE"] == "越南语"


def test_confidence_wrapper_passes_configured_language():
    wrapper_calls = [
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "transcribe_with_token_confidence"
    ]

    assert wrapper_calls
    assert len(wrapper_calls[0].args) >= 3
    assert isinstance(wrapper_calls[0].args[2], ast.Name)
    assert wrapper_calls[0].args[2].id == "LANGUAGE"


def test_modal_image_includes_logits_helper_module():
    image_assignments = [
        node
        for node in TREE.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "image"
    ]
    assert len(image_assignments) == 1

    local_source_calls = [
        node
        for node in ast.walk(image_assignments[0].value)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_local_python_source"
    ]

    assert local_source_calls
    assert any(
        call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "funasr_logits_infer"
        for call in local_source_calls
    )


def test_worker_uses_logits_confidence_wrapper():
    transcribe_functions = [
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.FunctionDef) and node.name == "transcribe"
    ]
    assert len(transcribe_functions) == 1
    transcribe = transcribe_functions[0]

    imports_wrapper = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "funasr_logits_infer"
        and any(alias.name == "transcribe_with_token_confidence" for alias in node.names)
        for node in ast.walk(transcribe)
    )
    calls_wrapper = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "transcribe_with_token_confidence"
        for node in ast.walk(transcribe)
    )

    assert imports_wrapper
    assert calls_wrapper


def test_default_batch_size_is_five_audio_files():
    assignments = _constant_assignments()

    assert assignments["BATCH_SIZE"] == 5


def test_local_entrypoint_accepts_batch_size_parameter():
    main_functions = [
        node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]

    assert len(main_functions) == 1
    args = main_functions[0].args.args
    assert [arg.arg for arg in args] == ["data_dir", "batch_size"]


def test_worker_transcribes_single_audio_because_funasr_vad_requires_batch_one():
    methods = [
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.FunctionDef)
    ]

    assert any(node.name == "transcribe" for node in methods)
    assert not any(node.name == "transcribe_batch" for node in methods)


def test_worker_delegates_batch_one_generation_to_confidence_wrapper():
    wrapper_calls = [
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "transcribe_with_token_confidence"
    ]
    direct_generate_calls = [
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "generate"
    ]

    assert wrapper_calls
    assert not direct_generate_calls


def test_local_entrypoint_runs_up_to_five_single_audio_remote_calls_concurrently():
    remote_calls = [
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "remote"
        and isinstance(node.func.value, ast.Attribute)
    ]
    executor_calls = [
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ThreadPoolExecutor"
    ]

    assert any(call.func.value.attr == "transcribe" for call in remote_calls)
    assert not any(call.func.value.attr == "transcribe_batch" for call in remote_calls)
    assert executor_calls
    assert any(
        keyword.arg == "max_workers"
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "batch_size"
        for call in executor_calls
        for keyword in call.keywords
    )
