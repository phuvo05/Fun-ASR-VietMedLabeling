# FunASR Modal Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local batch pseudo-labeling CLI that sends audio files from `data/` to FunASR running on Modal GPU and writes resumable per-audio JSON outputs.

**Architecture:** Keep local orchestration and pure helper logic testable in `asr_batch.py`, and keep Modal-specific remote execution in `modal_funasr_infer.py`. The local entrypoint scans `data/`, skips successful outputs, invokes the Modal function, writes JSON atomically, and logs per-file errors without stopping the batch.

**Tech Stack:** Python 3.10+, Modal, FunASR, PyTorch, pytest, JSON/JSONL, local filesystem.

---

## File structure

- Create: `data/.gitkeep` — ensures the user has a committed `data/` folder to drop audio files into.
- Create: `asr_batch.py` — pure local helpers: audio discovery, output path mapping, success detection, normalization, atomic writes, error logging.
- Create: `modal_funasr_infer.py` — Modal app definition, FunASR remote function, and local batch entrypoint.
- Create: `requirements.txt` — local runtime/test dependencies.
- Create: `README.md` — setup and usage instructions.
- Create: `tests/test_asr_batch.py` — tests for local helper behavior.

---

### Task 1: Add testable local batch helpers

**Files:**
- Create: `asr_batch.py`
- Create: `tests/test_asr_batch.py`
- Create: `requirements.txt`

- [ ] **Step 1: Create initial failing tests**

Create `tests/test_asr_batch.py` with:

```python
import json
from pathlib import Path

from asr_batch import (
    AUDIO_EXTENSIONS,
    atomic_write_json,
    discover_audio_files,
    error_record,
    is_success_output,
    output_path_for_audio,
    normalize_asr_result,
)


def test_audio_extensions_include_wav():
    assert ".wav" in AUDIO_EXTENSIONS


def test_output_path_keeps_original_audio_filename(tmp_path):
    audio = tmp_path / "VietMed_un_001_s05OFV.wav"
    assert output_path_for_audio(audio) == tmp_path / "VietMed_un_001_s05OFV.wav.json"


def test_discover_audio_files_recursively_and_ignores_json(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"RIFF")
    (tmp_path / "b.json").write_text("{}", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.WAV").write_bytes(b"RIFF")

    found = discover_audio_files(tmp_path)

    assert found == [tmp_path / "a.wav", nested / "c.WAV"]


def test_success_output_requires_matching_id_text_and_timestamps_key(tmp_path):
    output = tmp_path / "sample.wav.json"
    output.write_text(
        json.dumps([
            {"id": "sample.wav", "text": "xin chào", "timestamps": []}
        ], ensure_ascii=False),
        encoding="utf-8",
    )

    assert is_success_output(output, "sample.wav") is True
    assert is_success_output(output, "other.wav") is False


def test_invalid_or_empty_output_is_not_success(tmp_path):
    invalid = tmp_path / "bad.wav.json"
    invalid.write_text("not json", encoding="utf-8")
    empty_text = tmp_path / "empty.wav.json"
    empty_text.write_text(
        json.dumps([{"id": "empty.wav", "text": "", "timestamps": []}]),
        encoding="utf-8",
    )

    assert is_success_output(invalid, "bad.wav") is False
    assert is_success_output(empty_text, "empty.wav") is False


def test_normalize_asr_result_from_text_only_result():
    normalized = normalize_asr_result("sample.wav", {"text": "nội dung mẫu"})

    assert normalized == [
        {"id": "sample.wav", "text": "nội dung mẫu", "timestamps": []}
    ]


def test_normalize_asr_result_preserves_word_timestamps():
    raw = {
        "text": "áp ứng",
        "timestamps": [
            {"word": "áp", "confidence": 0.46, "start": 0.0, "end": 0.16},
            {"word": "ứng", "confidence": 0.99, "start": 0.16, "end": 0.32},
        ],
    }

    normalized = normalize_asr_result("sample.wav", raw)

    assert normalized[0]["timestamps"] == raw["timestamps"]


def test_normalize_asr_result_accepts_funasr_list_result():
    normalized = normalize_asr_result("sample.wav", [{"text": "xin chào"}])

    assert normalized == [
        {"id": "sample.wav", "text": "xin chào", "timestamps": []}
    ]


def test_atomic_write_json_writes_final_file_without_tmp(tmp_path):
    output = tmp_path / "sample.wav.json"
    payload = [{"id": "sample.wav", "text": "xin chào", "timestamps": []}]

    atomic_write_json(output, payload)

    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert not output.with_suffix(output.suffix + ".tmp").exists()


def test_error_record_has_required_fields():
    record = error_record(Path("data/sample.wav"), "inference failed", "modal")

    assert record["audio_path"] == "data/sample.wav"
    assert record["filename"] == "sample.wav"
    assert record["error"] == "inference failed"
    assert record["stage"] == "modal"
    assert "timestamp" in record
```

- [ ] **Step 2: Create minimal requirements for tests**

Create `requirements.txt` with:

```text
modal>=1.0.0
pytest>=8.0.0
```

- [ ] **Step 3: Run tests and verify they fail because helpers do not exist yet**

Run:

```powershell
python -m pytest tests/test_asr_batch.py -v
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'asr_batch'` or missing helper names.

- [ ] **Step 4: Implement local helper module**

Create `asr_batch.py` with:

```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def discover_audio_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    files = [
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return sorted(files, key=lambda path: str(path).lower())


def output_path_for_audio(audio_path: Path) -> Path:
    return audio_path.with_name(f"{audio_path.name}.json")


def is_success_output(output_path: Path, expected_audio_id: str) -> bool:
    if not output_path.exists():
        return False
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, list) or not payload:
        return False
    item = payload[0]
    if not isinstance(item, dict):
        return False
    if item.get("id") != expected_audio_id:
        return False
    if not isinstance(item.get("text"), str) or not item["text"].strip():
        return False
    if "timestamps" not in item or not isinstance(item["timestamps"], list):
        return False
    return True


def normalize_asr_result(audio_id: str, raw_result: Any) -> list[dict[str, Any]]:
    item = _first_result_item(raw_result)
    text = str(item.get("text", "")).strip()
    timestamps = _extract_timestamps(item)
    return [{"id": audio_id, "text": text, "timestamps": timestamps}]


def _first_result_item(raw_result: Any) -> dict[str, Any]:
    if isinstance(raw_result, list) and raw_result:
        first = raw_result[0]
        return first if isinstance(first, dict) else {"text": str(first)}
    if isinstance(raw_result, dict):
        return raw_result
    return {"text": str(raw_result)}


def _extract_timestamps(item: dict[str, Any]) -> list[dict[str, Any]]:
    timestamps = item.get("timestamps") or item.get("timestamp") or item.get("words")
    if isinstance(timestamps, list):
        return [_normalize_timestamp(ts) for ts in timestamps if isinstance(ts, dict)]
    return []


def _normalize_timestamp(ts: dict[str, Any]) -> dict[str, Any]:
    return {
        "word": str(ts.get("word") or ts.get("text") or ""),
        "confidence": ts.get("confidence"),
        "start": ts.get("start"),
        "end": ts.get("end"),
    }


def atomic_write_json(output_path: Path, payload: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(output_path)


def error_record(audio_path: Path, error: str, stage: str) -> dict[str, str]:
    return {
        "audio_path": str(audio_path),
        "filename": audio_path.name,
        "error": error,
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
```

- [ ] **Step 5: Run helper tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_asr_batch.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit local helpers**

Run:

```powershell
git add asr_batch.py tests/test_asr_batch.py requirements.txt
git commit -m @'
Add local ASR batch helpers

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
'@
```

---

### Task 2: Add Modal FunASR inference app and local batch entrypoint

**Files:**
- Create: `modal_funasr_infer.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add Modal/FunASR dependencies to requirements**

Replace `requirements.txt` with:

```text
modal>=1.0.0
pytest>=8.0.0
```

Keep heavy remote dependencies in the Modal image instead of local requirements so local setup stays lightweight.

- [ ] **Step 2: Create Modal inference script**

Create `modal_funasr_infer.py` with:

```python
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import modal

from asr_batch import (
    append_jsonl,
    atomic_write_json,
    discover_audio_files,
    error_record,
    is_success_output,
    normalize_asr_result,
    output_path_for_audio,
)

APP_NAME = "vietmed-funasr-labeling"
DATA_DIR = Path("data")
ERROR_LOG = DATA_DIR / "_asr_errors.jsonl"
MODEL_ID = "FunAudioLLM/Fun-ASR-Nano-2512"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch",
        "torchaudio",
        "funasr>=1.3.3",
        "modelscope",
        "huggingface_hub",
        "soundfile",
        "librosa",
    )
)

app = modal.App(APP_NAME, image=image)


@app.cls(gpu="T4", timeout=60 * 30, scaledown_window=60 * 5)
class FunASRWorker:
    @modal.enter()
    def load_model(self) -> None:
        from funasr import AutoModel

        self.model = AutoModel(
            model=MODEL_ID,
            trust_remote_code=True,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0",
            hub="hf",
        )

    @modal.method()
    def transcribe(self, audio_bytes: bytes, filename: str) -> Any:
        import tempfile
        from pathlib import Path

        suffix = Path(filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            result = self.model.generate(
                input=[tmp.name],
                cache={},
                batch_size=1,
            )
        return result


@app.local_entrypoint()
def main(data_dir: str = str(DATA_DIR)) -> None:
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    audio_files = discover_audio_files(root)
    if not audio_files:
        print(f"No audio files found in {root}. Add .wav files and run again.")
        return

    worker = FunASRWorker()
    skipped = 0
    succeeded = 0
    failed = 0

    for audio_path in audio_files:
        output_path = output_path_for_audio(audio_path)
        if is_success_output(output_path, audio_path.name):
            skipped += 1
            print(f"SKIP {audio_path}")
            continue

        print(f"ASR  {audio_path}")
        try:
            audio_bytes = audio_path.read_bytes()
            raw_result = worker.transcribe.remote(audio_bytes, audio_path.name)
            payload = normalize_asr_result(audio_path.name, raw_result)
            if not payload[0]["text"]:
                raise ValueError("FunASR returned empty text")
            atomic_write_json(output_path, payload)
            succeeded += 1
            print(f"OK   {output_path}")
        except Exception as exc:
            failed += 1
            append_jsonl(
                ERROR_LOG,
                error_record(
                    audio_path,
                    f"{exc}\n{traceback.format_exc()}",
                    "modal_inference",
                ),
            )
            print(f"FAIL {audio_path}: {exc}")

    print("\nSummary")
    print(f"  total:     {len(audio_files)}")
    print(f"  skipped:   {skipped}")
    print(f"  succeeded: {succeeded}")
    print(f"  failed:    {failed}")
    print(f"  data_dir:  {root}")
```

- [ ] **Step 3: Run local tests to ensure Modal script did not break helper behavior**

Run:

```powershell
python -m pytest tests/test_asr_batch.py -v
```

Expected: all tests PASS.

- [ ] **Step 4: Run no-audio smoke check**

Run:

```powershell
modal run modal_funasr_infer.py
```

Expected if `data/` has no audio: prints `No audio files found in data. Add .wav files and run again.` and exits without GPU inference.

If Modal is not logged in, expected failure mentions Modal authentication. Tell the user to run:

```powershell
! modal setup
```

- [ ] **Step 5: Commit Modal app**

Run:

```powershell
git add modal_funasr_infer.py requirements.txt
git commit -m @'
Add Modal FunASR inference app

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
'@
```

---

### Task 3: Add data folder and usage documentation

**Files:**
- Create: `data/.gitkeep`
- Create: `README.md`

- [ ] **Step 1: Create data folder placeholder**

Create `data/.gitkeep` as an empty file.

- [ ] **Step 2: Write README**

Create `README.md` with:

```markdown
# VietMed FunASR Modal Labeling

This repository runs pseudo-labeling for VietMed audio files with FunASR on Modal.com.

## Workflow

1. Put audio files in `data/`.
2. Run the Modal batch command.
3. Each successful audio file gets a JSON output next to it.
4. Re-running the command skips files that already have successful output.

Example input:

```text
data/VietMed_un_001_s05OFV.wav
```

Example output:

```text
data/VietMed_un_001_s05OFV.wav.json
```

Output JSON format:

```json
[
  {
    "id": "VietMed_un_001_s05OFV.wav",
    "text": "...",
    "timestamps": [
      {
        "word": "áp",
        "confidence": 0.4609244167804718,
        "start": 0.0,
        "end": 0.16
      }
    ]
  }
]
```

If FunASR does not return word-level timestamps or confidence for a file, `timestamps` is an empty list. The script does not fabricate timestamps or confidence values.

## Setup

Install local dependencies:

```powershell
python -m pip install -r requirements.txt
```

Log in to Modal if needed:

```powershell
modal setup
```

## Run

```powershell
modal run modal_funasr_infer.py
```

Use a custom data folder:

```powershell
modal run modal_funasr_infer.py --data-dir "D:\path\to\audio_data"
```

## Resume behavior

A file is skipped when its output JSON exists, parses correctly, has matching `id`, has non-empty `text`, and contains a `timestamps` list.

Invalid or partial output files are re-run.

Failures are appended to:

```text
data/_asr_errors.jsonl
```

The batch continues after individual file failures.

## Tests

```powershell
python -m pytest -v
```
```

- [ ] **Step 3: Run tests**

Run:

```powershell
python -m pytest -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit docs and data folder**

Run:

```powershell
git add README.md data/.gitkeep
git commit -m @'
Document FunASR Modal labeling workflow

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
'@
```

---

### Task 4: Final verification

**Files:**
- No new files expected.

- [ ] **Step 1: Run full test suite**

Run:

```powershell
python -m pytest -v
```

Expected: all tests PASS.

- [ ] **Step 2: Check repository status**

Run:

```powershell
git status --short
```

Expected: clean working tree, unless the user has added audio files under `data/`.

- [ ] **Step 3: Optional Modal smoke check**

Run only if Modal is authenticated:

```powershell
modal run modal_funasr_infer.py
```

Expected with empty `data/`: clean no-audio message. Expected with audio files: creates `data/<audio filename>.json` for each successful audio and skips those files on a second run.

- [ ] **Step 4: Report results**

Tell the user:

- which files were created,
- test command and result,
- whether Modal smoke check was run or skipped,
- how to add audio and run inference.

---

## Self-review

Spec coverage:

- Local `data/` folder: covered by Task 3.
- One JSON output per audio file: covered by Tasks 1 and 2.
- Resume/skip successful outputs: covered by Task 1 tests and Task 2 entrypoint.
- Modal GPU FunASR inference: covered by Task 2.
- Error JSONL logging: covered by Task 1 helper and Task 2 exception handling.
- README usage instructions: covered by Task 3.
- No fabricated confidence/timestamps: covered by Task 1 normalization and README.

Placeholder scan: no TBD/TODO/fill-in placeholders remain.

Type consistency: helper names used by tests and Modal script match the definitions in Task 1.
