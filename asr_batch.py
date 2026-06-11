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
    if isinstance(raw_result, list):
        if not raw_result:
            return {"text": ""}
        first = raw_result[0]
        return first if isinstance(first, dict) else {"text": str(first)}
    if isinstance(raw_result, dict):
        return raw_result
    return {"text": str(raw_result)}


def _extract_timestamps(item: dict[str, Any]) -> list[dict[str, Any]]:
    timestamps = item.get("timestamps") or item.get("timestamp") or item.get("words")
    if not isinstance(timestamps, list):
        return []

    normalized = []
    for timestamp in timestamps:
        if isinstance(timestamp, dict):
            normalized.append(dict(timestamp))
        elif _is_timestamp_pair(timestamp):
            normalized.append({"start": timestamp[0], "end": timestamp[1]})
    return normalized


def _is_timestamp_pair(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) >= 2


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
        "audio_path": audio_path.as_posix(),
        "filename": audio_path.name,
        "error": error,
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
