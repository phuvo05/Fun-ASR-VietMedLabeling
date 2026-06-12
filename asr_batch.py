from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
PUNCTUATION_TOKENS = set(".,!?;:，。！？；：、…")

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
    timestamps = item.get("timestamps")
    if not isinstance(timestamps, list):
        return False

    return _looks_like_word_level_timestamps(timestamps, item["text"])

def _looks_like_word_level_timestamps(timestamps: list[Any], text: str) -> bool:
    if not timestamps:
        return False

    # JSON cũ có field token => chắc chắn là token-level, không được skip.
    if any(isinstance(timestamp, dict) and "token" in timestamp for timestamp in timestamps):
        return False

    text_word_count = len(text.split())

    # Nếu timestamp nhiều hơn số word quá nhiều thì khả năng vẫn là token-level.
    if text_word_count and len(timestamps) > max(text_word_count + 5, int(text_word_count * 1.25)):
        return False

    for timestamp in timestamps:
        if not isinstance(timestamp, dict):
            return False
        if "word" not in timestamp:
            return False
        if "start" not in timestamp or "end" not in timestamp:
            return False

    return True


def normalize_asr_result(audio_id: str, raw_result: Any) -> list[dict[str, Any]]:
    item = _first_result_item(raw_result)
    text = str(item.get("text", "")).strip()
    timestamps = _extract_timestamps(item, text=text)
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


def _extract_timestamps(item: dict[str, Any], text: str = "") -> list[dict[str, Any]]:
    timestamps = item.get("timestamps") or item.get("timestamp") or item.get("words")
    if not isinstance(timestamps, list):
        return []

    # FunASR trả timestamp dạng token/subword.
    # Cần gộp token trước khi lưu JSON.
    if timestamps and all(
        isinstance(timestamp, dict) and _has_funasr_token_timestamp_fields(timestamp)
        for timestamp in timestamps
    ):
        return _merge_funasr_token_timestamps(timestamps, text)

    normalized = []
    for timestamp in timestamps:
        if isinstance(timestamp, dict):
            normalized.append(_normalize_timestamp_dict(timestamp))
        elif _is_timestamp_pair(timestamp):
            normalized.append({"start": timestamp[0], "end": timestamp[1]})
    return normalized

def _merge_funasr_token_timestamps(
    token_timestamps: list[dict[str, Any]],
    transcript: str = "",
) -> list[dict[str, Any]]:
    token_groups = _group_token_timestamps_by_word(token_timestamps)
    transcript_words = transcript.split() if transcript else []

    use_transcript_words = len(transcript_words) == len(token_groups)
    word_timestamps: list[dict[str, Any]] = []

    for index, group in enumerate(token_groups):
        word = transcript_words[index] if use_transcript_words else _tokens_to_text(group)
        word = word.strip()
        if not word:
            continue

        scores = [
            _to_float(token.get("score"))
            for token in group
            if _to_float(token.get("score")) is not None
        ]
        confidence = round(sum(scores) / len(scores), 6) if scores else None

        word_timestamps.append(
            {
                "word": word,
                "start": _first_non_null(token.get("start_time") for token in group),
                "end": _last_non_null(token.get("end_time") for token in group),
                "confidence": confidence,
            }
        )

    return word_timestamps


def _group_token_timestamps_by_word(
    token_timestamps: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for timestamp in token_timestamps:
        token = str(timestamp.get("token", ""))
        stripped = token.strip()
        if not stripped:
            continue

        starts_new_word = token[:1].isspace() and not _is_punctuation_token(stripped)

        if current and starts_new_word:
            groups.append(current)
            current = []

        current.append(timestamp)

    if current:
        groups.append(current)

    return groups


def _tokens_to_text(group: list[dict[str, Any]]) -> str:
    return "".join(str(token.get("token", "")) for token in group).strip()


def _normalize_timestamp_dict(timestamp: dict[str, Any]) -> dict[str, Any]:
    if _has_funasr_token_timestamp_fields(timestamp):
        return {
            "word": str(timestamp["token"]).strip(),
            "confidence": timestamp.get("score"),
            "start": timestamp.get("start_time"),
            "end": timestamp.get("end_time"),
        }
    return dict(timestamp)


def _has_funasr_token_timestamp_fields(timestamp: dict[str, Any]) -> bool:
    return "token" in timestamp and "start_time" in timestamp and "end_time" in timestamp


def _is_punctuation_token(value: str) -> bool:
    return bool(value) and all(char in PUNCTUATION_TOKENS for char in value)


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_non_null(values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _last_non_null(values: Any) -> Any:
    last = None
    for value in values:
        if value is not None:
            last = value
    return last

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
