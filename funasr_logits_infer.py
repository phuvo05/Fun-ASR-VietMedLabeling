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
