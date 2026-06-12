# FunASR Logits Confidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a raw-logits confidence path so each word confidence is the minimum softmax probability of the ASR tokens that compose that word.

**Architecture:** Keep the existing Modal batch/resume/output pipeline intact. Add a narrow helper module that can inspect/call FunASR internals for token confidence, while `asr_batch.py` remains the only place that normalizes token timestamps into output JSON. If logits are not available from the installed FunASR model path, output `confidence: null` instead of fabricating scores.

**Tech Stack:** Python 3.10, Modal, FunASR, PyTorch, pytest, AST-based tests.

---

## File Structure

- Modify `asr_batch.py`: accept `token_confidence`/`confidence` as preferred token confidence fields when merging FunASR token timestamps; keep `score > 0.0` fallback.
- Modify `tests/test_asr_batch.py`: add tests proving word confidence uses min token softmax confidence and ignores placeholder zero `score` when better confidence is provided.
- Create `funasr_logits_infer.py`: focused wrapper for FunASR inference internals. It exposes a stable function `transcribe_with_token_confidence(model, audio_path, language)` returning the same shape as `AutoModel.generate()`, with token entries enriched by `token_confidence` when available.
- Create `tests/test_funasr_logits_infer.py`: unit tests with fake models/results; no GPU or real FunASR model required.
- Modify `modal_funasr_infer.py`: call `transcribe_with_token_confidence()` inside the Modal worker instead of directly returning `self.model.generate(...)`.
- Modify `tests/test_modal_funasr_infer.py`: update AST tests to require the helper call while preserving `batch_size=1` and local concurrency of 5 remote calls.
- Optionally modify `README.md`: document that `confidence` is logits-derived when available, otherwise `null`.

---

### Task 1: Prefer logits-derived token confidence during word merge

**Files:**
- Modify: `asr_batch.py:131-195`
- Test: `tests/test_asr_batch.py`

- [ ] **Step 1: Write the failing test for token_confidence priority**

Add this test to `tests/test_asr_batch.py` near the existing FunASR timestamp tests:

```python
def test_normalize_asr_result_uses_min_token_confidence_for_word_confidence():
    raw = {
        "text": "Hợp hoạt",
        "timestamps": [
            {
                "token": "H",
                "score": 0.0,
                "token_confidence": 0.93,
                "start_time": 0.0,
                "end_time": 0.1,
            },
            {
                "token": "ợ",
                "score": 0.0,
                "token_confidence": 0.61,
                "start_time": 0.1,
                "end_time": 0.2,
            },
            {
                "token": "p",
                "score": 0.0,
                "token_confidence": 0.72,
                "start_time": 0.2,
                "end_time": 0.3,
            },
            {
                "token": " hoạt",
                "score": 0.0,
                "token_confidence": 0.84,
                "start_time": 0.3,
                "end_time": 0.6,
            },
        ],
    }

    normalized = normalize_asr_result("sample.wav", raw)

    assert normalized[0]["timestamps"] == [
        {"word": "Hợp", "confidence": 0.61, "start": 0.0, "end": 0.3},
        {"word": "hoạt", "confidence": 0.84, "start": 0.3, "end": 0.6},
    ]
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
python -m pytest tests/test_asr_batch.py::test_normalize_asr_result_uses_min_token_confidence_for_word_confidence -v
```

Expected: FAIL because `asr_batch.py` does not yet read `token_confidence`.

- [ ] **Step 3: Implement minimal confidence extraction helper**

In `asr_batch.py`, add this helper below `_normalize_timestamp_dict` and above `_is_usable_confidence`:

```python
def _timestamp_confidence(timestamp: dict[str, Any]) -> float | None:
    for key in ("token_confidence", "confidence", "score"):
        value = timestamp.get(key)
        if _is_usable_confidence(value):
            return float(value)
    return None
```

Then replace the confidence extraction in `_merge_funasr_token_timestamps`:

```python
        score = timestamp.get("score")
        if _is_usable_confidence(score):
            score_value = float(score)
            score_min = score_value if score_min is None else min(score_min, score_value)
```

with:

```python
        confidence = _timestamp_confidence(timestamp)
        if confidence is not None:
            score_min = confidence if score_min is None else min(score_min, confidence)
```

Then replace `_normalize_timestamp_dict` with:

```python
def _normalize_timestamp_dict(timestamp: dict[str, Any]) -> dict[str, Any]:
    if _has_funasr_token_timestamp_fields(timestamp):
        return {
            "word": str(timestamp["token"]).strip(),
            "confidence": _timestamp_confidence(timestamp),
            "start": timestamp.get("start_time"),
            "end": timestamp.get("end_time"),
        }
    return dict(timestamp)
```

- [ ] **Step 4: Run the targeted test to verify it passes**

Run:

```powershell
python -m pytest tests/test_asr_batch.py::test_normalize_asr_result_uses_min_token_confidence_for_word_confidence -v
```

Expected: PASS.

- [ ] **Step 5: Run all asr_batch tests**

Run:

```powershell
python -m pytest tests/test_asr_batch.py -v
```

Expected: all tests in `tests/test_asr_batch.py` pass.

- [ ] **Step 6: Commit**

Run:

```powershell
git add asr_batch.py tests/test_asr_batch.py
git commit -m @'
feat: prefer logits token confidence in labels

Co-Authored-By: Claude <noreply@anthropic.com>
'@
```

---

### Task 2: Add a FunASR logits inference wrapper with safe fallback

**Files:**
- Create: `funasr_logits_infer.py`
- Test: `tests/test_funasr_logits_infer.py`

- [ ] **Step 1: Write failing tests for wrapper fallback and token enrichment**

Create `tests/test_funasr_logits_infer.py` with:

```python
from pathlib import Path

from funasr_logits_infer import transcribe_with_token_confidence


class FakeGenerateOnlyModel:
    def generate(self, **kwargs):
        self.kwargs = kwargs
        return [
            {
                "text": "xin chào",
                "timestamps": [
                    {"token": "xin", "score": 0.0, "start_time": 0.0, "end_time": 0.2},
                    {"token": " chào", "score": 0.0, "start_time": 0.2, "end_time": 0.5},
                ],
            }
        ]


class FakeLogitsModel(FakeGenerateOnlyModel):
    def token_confidences(self, audio_path: str, language: str):
        return [0.91, 0.73]


def test_transcribe_with_token_confidence_preserves_generate_kwargs(tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    model = FakeGenerateOnlyModel()

    result = transcribe_with_token_confidence(model, audio, "越南语")

    assert model.kwargs == {
        "input": [str(audio)],
        "cache": {},
        "batch_size": 1,
        "language": "越南语",
    }
    assert result[0]["timestamps"][0]["token"] == "xin"
    assert "token_confidence" not in result[0]["timestamps"][0]


def test_transcribe_with_token_confidence_enriches_tokens_when_available(tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    model = FakeLogitsModel()

    result = transcribe_with_token_confidence(model, audio, "越南语")

    assert result[0]["timestamps"] == [
        {
            "token": "xin",
            "score": 0.0,
            "start_time": 0.0,
            "end_time": 0.2,
            "token_confidence": 0.91,
            "confidence_source": "softmax_logits",
        },
        {
            "token": " chào",
            "score": 0.0,
            "start_time": 0.2,
            "end_time": 0.5,
            "token_confidence": 0.73,
            "confidence_source": "softmax_logits",
        },
    ]
```

- [ ] **Step 2: Run tests to verify import fails**

Run:

```powershell
python -m pytest tests/test_funasr_logits_infer.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'funasr_logits_infer'`.

- [ ] **Step 3: Create wrapper with fallback and test hook**

Create `funasr_logits_infer.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any


def transcribe_with_token_confidence(model: Any, audio_path: Path, language: str) -> Any:
    result = model.generate(
        input=[str(audio_path)],
        cache={},
        batch_size=1,
        language=language,
    )
    confidences = _try_extract_token_confidences(model, audio_path, language)
    if not confidences:
        return result
    return _attach_token_confidences(result, confidences)


def _try_extract_token_confidences(
    model: Any,
    audio_path: Path,
    language: str,
) -> list[float] | None:
    extractor = getattr(model, "token_confidences", None)
    if not callable(extractor):
        return None
    try:
        values = extractor(str(audio_path), language)
    except Exception:
        return None
    if not isinstance(values, list):
        return None
    confidences = []
    for value in values:
        if isinstance(value, (int, float)) and float(value) > 0.0:
            confidences.append(float(value))
    return confidences or None


def _attach_token_confidences(result: Any, confidences: list[float]) -> Any:
    item = _first_result_item(result)
    if item is None:
        return result
    timestamps = item.get("timestamps") or item.get("timestamp") or item.get("words")
    if not isinstance(timestamps, list):
        return result
    confidence_index = 0
    for timestamp in timestamps:
        if not isinstance(timestamp, dict):
            continue
        if "token" not in timestamp:
            continue
        if confidence_index >= len(confidences):
            break
        timestamp["token_confidence"] = round(confidences[confidence_index], 6)
        timestamp["confidence_source"] = "softmax_logits"
        confidence_index += 1
    return result


def _first_result_item(result: Any) -> dict[str, Any] | None:
    if isinstance(result, list) and result and isinstance(result[0], dict):
        return result[0]
    if isinstance(result, dict):
        return result
    return None
```

- [ ] **Step 4: Run wrapper tests to verify they pass**

Run:

```powershell
python -m pytest tests/test_funasr_logits_infer.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add funasr_logits_infer.py tests/test_funasr_logits_infer.py
git commit -m @'
feat: add FunASR token confidence wrapper

Co-Authored-By: Claude <noreply@anthropic.com>
'@
```

---

### Task 3: Wire Modal worker through logits wrapper

**Files:**
- Modify: `modal_funasr_infer.py:55-69`
- Modify: `tests/test_modal_funasr_infer.py`

- [ ] **Step 1: Write AST test requiring wrapper import inside worker method**

Add this test to `tests/test_modal_funasr_infer.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_modal_funasr_infer.py::test_worker_uses_logits_confidence_wrapper -v
```

Expected: FAIL because `modal_funasr_infer.py` still calls `self.model.generate(...)` directly.

- [ ] **Step 3: Update Modal worker method**

In `modal_funasr_infer.py`, replace the body of `transcribe()` with:

```python
    @modal.method()
    def transcribe(self, audio_bytes: bytes, filename: str) -> Any:
        import tempfile

        from funasr_logits_infer import transcribe_with_token_confidence

        suffix = safe_audio_suffix(filename)
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            result = transcribe_with_token_confidence(
                self.model,
                Path(tmp.name),
                LANGUAGE,
            )
        return result
```

Keep `Path` imported at the top of the file; it already is.

- [ ] **Step 4: Run Modal AST tests**

Run:

```powershell
python -m pytest tests/test_modal_funasr_infer.py -v
```

Expected: all Modal AST tests pass, including existing checks for no `transcribe_batch` and local `ThreadPoolExecutor(max_workers=batch_size)`.

- [ ] **Step 5: Commit**

Run:

```powershell
git add modal_funasr_infer.py tests/test_modal_funasr_infer.py
git commit -m @'
feat: route Modal inference through confidence wrapper

Co-Authored-By: Claude <noreply@anthropic.com>
'@
```

---

### Task 4: Replace test hook with real FunASR internals investigation

**Files:**
- Modify: `funasr_logits_infer.py`
- Test: `tests/test_funasr_logits_infer.py`

- [ ] **Step 1: Add introspection diagnostics function test**

Append to `tests/test_funasr_logits_infer.py`:

```python
from funasr_logits_infer import describe_model_for_logits


class FakeNestedModel:
    def __init__(self):
        self.model = object()
        self.kwargs = {"model": "fake"}


def test_describe_model_for_logits_lists_relevant_attributes():
    description = describe_model_for_logits(FakeNestedModel())

    assert description["class"] == "FakeNestedModel"
    assert "model" in description["attributes"]
    assert "generate" not in description["attributes"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_funasr_logits_infer.py::test_describe_model_for_logits_lists_relevant_attributes -v
```

Expected: FAIL because `describe_model_for_logits` does not exist.

- [ ] **Step 3: Add safe model description helper**

Append to `funasr_logits_infer.py`:

```python
def describe_model_for_logits(model: Any) -> dict[str, Any]:
    attributes = [name for name in dir(model) if not name.startswith("_")]
    return {
        "class": type(model).__name__,
        "module": type(model).__module__,
        "attributes": sorted(attributes),
    }
```

- [ ] **Step 4: Run wrapper tests**

Run:

```powershell
python -m pytest tests/test_funasr_logits_infer.py -v
```

Expected: all wrapper tests pass.

- [ ] **Step 5: Inspect FunASR source on installed environment**

Run locally after dependencies are installed:

```powershell
python - <<'PY'
import inspect
import funasr
from funasr import AutoModel
print('funasr', getattr(funasr, '__version__', 'unknown'))
print('AutoModel file:', inspect.getsourcefile(AutoModel))
print('AutoModel.generate file:', inspect.getsourcefile(AutoModel.generate))
print(inspect.signature(AutoModel.generate))
PY
```

Expected: command prints the installed FunASR version, source file paths, and `AutoModel.generate` signature. Use this information to identify whether logits are exposed through an internal model, decoder, or output scores.

- [ ] **Step 6: Inspect a real loaded model on Modal with diagnostics**

Temporarily add this diagnostic inside `FunASRWorker.load_model()` after `self.model = AutoModel(...)`:

```python
        from funasr_logits_infer import describe_model_for_logits

        print("FunASR model diagnostics:", describe_model_for_logits(self.model))
```

Run a one-file Modal smoke test:

```powershell
$env:PYTHONIOENCODING = "utf-8"
modal run modal_funasr_infer.py --data-dir data_modal_smoke_mlt_verify --batch-size 1
```

Expected: logs show the model class and relevant attributes. Remove this temporary print after collecting the result.

- [ ] **Step 7: Implement the first real logits extractor path**

Replace `_try_extract_token_confidences()` in `funasr_logits_infer.py` with the real extractor path discovered in Step 5 and Step 6. Keep this fallback behavior exactly:

```python
    if not logits_path_is_available:
        return None
```

The implementation must return a `list[float]` where each value is the selected-token softmax probability for the corresponding emitted ASR token. If the discovered FunASR path returns log-probabilities directly, convert with `math.exp(log_probability)`. If it returns logits, use `torch.softmax(logits, dim=-1)` and select the generated token id.

- [ ] **Step 8: Add a unit test for the real extractor adapter shape**

Add a fake object that mimics the discovered internal shape and assert `_try_extract_token_confidences()` returns expected probabilities. Example if the internal shape is logits plus token ids:

```python
def test_try_extract_token_confidences_from_logits_adapter():
    model = FakeDiscoveredLogitsModel(
        token_ids=[1, 0],
        logits=[
            [0.0, 2.0],
            [3.0, 0.0],
        ],
    )

    confidences = _try_extract_token_confidences(model, Path("sample.wav"), "越南语")

    assert confidences == [0.880797, 0.952574]
```

Adjust the fake class to match the real adapter implemented in Step 7.

- [ ] **Step 9: Run tests**

Run:

```powershell
python -m pytest tests/test_funasr_logits_infer.py tests/test_asr_batch.py tests/test_modal_funasr_infer.py -v
```

Expected: all selected tests pass.

- [ ] **Step 10: Commit**

Run:

```powershell
git add funasr_logits_infer.py tests/test_funasr_logits_infer.py modal_funasr_infer.py
git commit -m @'
feat: extract FunASR logits token confidence

Co-Authored-By: Claude <noreply@anthropic.com>
'@
```

---

### Task 5: Document confidence semantics and run full verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README output semantics**

Replace `README.md:43` with:

```markdown
If FunASR returns token logits through the patched inference path, each word `confidence` is the minimum softmax probability among the ASR tokens that compose that word. If logits are unavailable for a file or model path, `confidence` is `null`; the pipeline does not fabricate confidence from placeholder `score: 0.0` values.
```

Also update the workflow output section if it still says each audio gets JSON next to it. Replace `README.md:5-22` with:

```markdown
## Workflow

1. Put audio folders in `data/`.
2. Run the Modal batch command.
3. Each audio folder gets one aggregate JSON file in `data/label/`.
4. Re-running the command skips audio items that already have successful output in the aggregate JSON.

Example input:

```text
data/VietMed_unlabeled_1000h_segmented_8kHz_000_050/VietMed_un_001_s05OFV.wav
```

Example output:

```text
data/label/VietMed_unlabeled_1000h_segmented_8kHz_000_050.json
```
```

- [ ] **Step 2: Run full pytest suite**

Run:

```powershell
python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 3: Run a Modal smoke test on one file**

Run:

```powershell
$env:PYTHONIOENCODING = "utf-8"
modal run modal_funasr_infer.py --data-dir data_modal_smoke_mlt_verify --batch-size 1
```

Expected: command completes, writes/updates one aggregate output, and does not crash. If existing output is skipped, remove only the smoke-test output file under `data_modal_smoke_mlt_verify` and rerun; do not delete user dataset labels under `data/label/` without explicit user confirmation.

- [ ] **Step 4: Inspect smoke-test output**

Open the generated JSON and verify:

```json
{
  "id": "VietMed_un_001_s05OFV.wav",
  "text": "...",
  "timestamps": [
    {
      "word": "...",
      "confidence": null,
      "start": 0.0,
      "end": 0.0
    }
  ]
}
```

or, if logits extraction succeeded:

```json
{
  "word": "...",
  "confidence": 0.812345,
  "start": 0.0,
  "end": 0.0
}
```

Expected: confidence is either a real positive softmax probability or `null`; it must not be placeholder `0.0`.

- [ ] **Step 5: Commit documentation and verification changes**

Run:

```powershell
git add README.md
git commit -m @'
docs: describe FunASR confidence semantics

Co-Authored-By: Claude <noreply@anthropic.com>
'@
```

---

## Self-Review

- Spec coverage: The plan covers logits-derived token confidence, word-level min aggregation, Modal worker integration, safe fallback to `null`, and documentation.
- Placeholder scan: No `TBD`, `TODO`, or unspecified test steps remain. Task 4 requires discovery because FunASR internals must be inspected, but it defines exact diagnostics and fallback behavior.
- Type consistency: The wrapper returns the existing `AutoModel.generate()` shape with token dicts enriched by `token_confidence`; `asr_batch.py` consumes that field consistently.
- Scope check: This is one subsystem: confidence extraction and propagation. It does not change batching, resume behavior, or output aggregation beyond README correction.
