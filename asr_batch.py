from __future__ import annotations

import json
import uuid
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


def chunked(items: list[Any], batch_size: int) -> list[list[Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def aggregate_output_path_for_audio(audio_path: Path, label_dir: Path) -> Path:
    return label_dir / f"{audio_path.parent.name}.json"


def is_success_output(output_path: Path, expected_audio_id: str) -> bool:
    if not output_path.exists():
        return False
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, list) or not payload:
        return False
    return any(
        _is_success_item(item, expected_audio_id)
        for item in payload
        if isinstance(item, dict)
    )


def _is_success_item(item: dict[str, Any], expected_audio_id: str) -> bool:
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


def merge_result_into_aggregate(
    output_path: Path,
    result: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing = _read_aggregate_output(output_path)
    if not result:
        return existing

    result_id = result[0].get("id")
    merged = []
    replaced = False
    for item in existing:
        if item.get("id") == result_id:
            merged.extend(result)
            replaced = True
        else:
            merged.append(item)
    if not replaced:
        merged.extend(result)
    return merged


def _read_aggregate_output(output_path: Path) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


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

    if all(
        isinstance(timestamp, dict) and _has_funasr_token_timestamp_fields(timestamp)
        for timestamp in timestamps
    ):
        return _merge_funasr_token_timestamps(timestamps)

    normalized = []
    for timestamp in timestamps:
        if isinstance(timestamp, dict):
            normalized.append(_normalize_timestamp_dict(timestamp))
        elif _is_timestamp_pair(timestamp):
            normalized.append({"start": timestamp[0], "end": timestamp[1]})
    return normalized


def _merge_funasr_token_timestamps(
    timestamps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    words = []
    current: dict[str, Any] | None = None
    score_min: float | None = None

    def flush_current() -> None:
        nonlocal current, score_min
        if current is None:
            return
        if score_min is not None:
            current["confidence"] = round(score_min, 6)
        words.append(current)
        current = None
        score_min = None

    for timestamp in timestamps:
        token = str(timestamp["token"])
        word_piece = token.strip()
        if not word_piece:
            continue

        starts_new_word = token[:1].isspace()
        is_punctuation = _is_punctuation_token(word_piece)
        if starts_new_word or is_punctuation:
            flush_current()

        if current is None:
            current = {
                "word": word_piece,
                "confidence": None,
                "start": timestamp.get("start_time"),
                "end": timestamp.get("end_time"),
            }
        else:
            current["word"] += word_piece
            current["end"] = timestamp.get("end_time")

        confidence = _timestamp_confidence(timestamp)
        if confidence is not None:
            score_min = confidence if score_min is None else min(score_min, confidence)

        if is_punctuation:
            flush_current()

    flush_current()
    return words


def _normalize_timestamp_dict(timestamp: dict[str, Any]) -> dict[str, Any]:
    if _has_funasr_token_timestamp_fields(timestamp):
        return {
            "word": str(timestamp["token"]).strip(),
            "confidence": _timestamp_confidence(timestamp),
            "start": timestamp.get("start_time"),
            "end": timestamp.get("end_time"),
        }
    return dict(timestamp)


def _timestamp_confidence(timestamp: dict[str, Any]) -> float | None:
    for key in ("token_confidence", "confidence", "score"):
        value = timestamp.get(key)
        if _is_usable_confidence(value):
            return float(value)
    return None


def _is_usable_confidence(score: Any) -> bool:
    return isinstance(score, (int, float)) and float(score) > 0.0


def _is_punctuation_token(token: str) -> bool:
    return token in {".", ",", "?", "!", ":", ";", "…", "。", "，", "？", "！", "：", "；"}


def _has_funasr_token_timestamp_fields(timestamp: dict[str, Any]) -> bool:
    return "token" in timestamp and "start_time" in timestamp and "end_time" in timestamp


def _is_timestamp_pair(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) >= 2


def atomic_write_json(output_path: Path, payload: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.{uuid.uuid4().hex}.tmp")
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
