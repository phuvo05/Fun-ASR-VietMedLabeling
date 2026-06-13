from __future__ import annotations

import math
import re
import types
import inspect
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
    if callable(extractor):
        try:
            values = extractor(str(audio_path), language)
        except Exception:
            values = None
        confidences = _valid_confidences(values)
        if confidences:
            return confidences
    return _try_extract_funasr_ctc_confidences(model, audio_path, language)


def _valid_confidences(values: Any) -> list[float] | None:
    if not isinstance(values, list):
        return None
    confidences = []
    for value in values:
        if value is None:
            confidences.append(None)
        elif isinstance(value, (int, float)) and float(value) > 0.0:
            confidences.append(float(value))
    return confidences or None


def _try_extract_funasr_ctc_confidences(
    model: Any,
    audio_path: Path,
    language: str,
) -> list[float] | None:
    inner_model = getattr(model, "model", None)
    if inner_model is None or not all(
        hasattr(inner_model, name)
        for name in ("inference_prepare", "ctc_decoder", "ctc", "ctc_tokenizer")
    ):
        return None
    result_holder: dict[str, Any] = {}
    original_inference_llm = getattr(inner_model, "inference_llm", None)
    if not callable(original_inference_llm):
        return None

    def wrapped_inference_llm(self: Any, data_in: Any, data_lengths: Any = None, key: list | None = None, tokenizer: Any = None, frontend: Any = None, **kwargs: Any) -> Any:
        result_holder["tokenizer"] = tokenizer
        original_llm_generate = getattr(getattr(self, "llm", None), "generate", None)

        def wrapped_llm_generate(*generate_args: Any, **generate_kwargs: Any) -> Any:
            generate_kwargs = dict(generate_kwargs)
            generate_kwargs["output_scores"] = True
            generate_kwargs["return_dict_in_generate"] = True
            output = original_llm_generate(*generate_args, **generate_kwargs)
            sequences = getattr(output, "sequences", None)
            scores = getattr(output, "scores", None)
            if sequences is not None and scores is not None:
                result_holder["llm_token_ids"], result_holder["llm_token_confidences"] = (
                    _generated_token_confidences(sequences, list(scores))
                )
                return sequences
            return output

        try:
            result_holder["ctc_log_probs"] = _extract_ctc_log_probs_from_inference(
                self,
                data_in,
                data_lengths=data_lengths,
                key=key,
                tokenizer=tokenizer,
                frontend=frontend,
                **kwargs,
            )
        except Exception:
            result_holder["ctc_log_probs"] = None

        if callable(original_llm_generate):
            self.llm.generate = wrapped_llm_generate
        try:
            return original_inference_llm(
                data_in,
                data_lengths=data_lengths,
                key=key,
                tokenizer=tokenizer,
                frontend=frontend,
                **kwargs,
            )
        finally:
            if callable(original_llm_generate):
                self.llm.generate = original_llm_generate

    try:
        inner_model.inference_llm = types.MethodType(wrapped_inference_llm, inner_model)
        result = model.generate(
            input=[str(audio_path)],
            cache={},
            batch_size=1,
            language=language,
        )
    except Exception:
        return None
    finally:
        try:
            inner_model.inference_llm = original_inference_llm
        except Exception:
            pass

    item = _first_result_item(result)
    if item is None:
        return None
    timestamps = item.get("timestamps") or item.get("timestamp") or item.get("words")
    if not isinstance(timestamps, list):
        return None
    llm_token_ids = result_holder.get("llm_token_ids")
    llm_token_confidences = result_holder.get("llm_token_confidences")
    tokenizer = result_holder.get("tokenizer")
    if tokenizer is not None and llm_token_ids and llm_token_confidences:
        confidences = _timestamp_confidences_from_generated_tokens(
            tokenizer,
            timestamps,
            llm_token_ids,
            llm_token_confidences,
        )
        if confidences:
            return confidences

    ctc_log_probs = result_holder.get("ctc_log_probs")
    if ctc_log_probs is None:
        return None
    return _token_confidences_from_ctc_log_probs(inner_model, ctc_log_probs, timestamps)


def _extract_ctc_log_probs_from_inference(inner_model: Any, data_in: Any, **kwargs: Any) -> Any:
    inputs_embeds, contents, batch, source_ids, meta_data = inner_model.inference_prepare(data_in, **kwargs)
    del inputs_embeds, contents, batch, source_ids
    if inner_model.ctc_decoder is None:
        return None
    encoder_out = meta_data["encoder_out"]
    encoder_out_lens = meta_data["encoder_out_lens"]
    decoder_out, _decoder_out_lens = inner_model.ctc_decoder(encoder_out, encoder_out_lens)
    ctc_logits = inner_model.ctc.log_softmax(decoder_out)
    return ctc_logits[0, : encoder_out_lens[0].item(), :]


def _generated_token_confidences(sequences: Any, scores: list[Any]) -> tuple[list[int], list[float]]:
    if not scores:
        return [], []
    generated_count = len(scores)
    sequence = sequences[0]
    generated_ids = sequence[-generated_count:]
    token_ids = []
    confidences = []
    for token_id_tensor, score in zip(generated_ids, scores):
        token_id = int(token_id_tensor.item())
        probability = score[0].softmax(dim=-1)[token_id]
        token_ids.append(token_id)
        confidences.append(round(float(probability.item()), 6))
    return token_ids, confidences



def _timestamp_confidences_from_generated_tokens(
    tokenizer: Any,
    timestamps: list[dict[str, Any]],
    token_ids: list[int],
    token_confidences: list[float],
) -> list[float] | None:
    pieces = []
    for token_id, confidence in zip(token_ids, token_confidences):
        try:
            text = tokenizer.decode([token_id], skip_special_tokens=True)
        except TypeError:
            text = tokenizer.decode([token_id])
        except Exception:
            text = ""
        if text:
            pieces.append((text, float(confidence)))

    cursor = 0
    confidences = []
    for timestamp in timestamps:
        if not isinstance(timestamp, dict) or "token" not in timestamp:
            continue
        target = _clean_piece_for_ctc(str(timestamp["token"])).replace(" ", "")
        if not target:
            confidences.append(None)
            continue
        combined = ""
        span_scores = []
        while cursor < len(pieces) and target not in combined.replace(" ", ""):
            piece, confidence = pieces[cursor]
            combined += piece
            span_scores.append(confidence)
            cursor += 1
        if target in combined.replace(" ", "") and span_scores:
            confidences.append(round(min(span_scores), 6))
        else:
            confidences.append(None)

    fallback = _timestamp_confidences_by_generated_word_order(timestamps, pieces)
    if any("�" in str(timestamp.get("token", "")) for timestamp in timestamps if isinstance(timestamp, dict)):
        return fallback
    if any(confidence is not None for confidence in confidences):
        return confidences
    return fallback



def _timestamp_confidences_by_generated_word_order(
    timestamps: list[dict[str, Any]],
    pieces: list[tuple[str, float]],
) -> list[float] | None:
    word_confidences = []
    current_scores = []
    for piece, confidence in pieces:
        if piece[:1].isspace() and current_scores:
            word_confidences.append(round(min(current_scores), 6))
            current_scores = []
        stripped = piece.strip()
        if stripped and not _is_punctuation_text(stripped):
            current_scores.append(confidence)
    if current_scores:
        word_confidences.append(round(min(current_scores), 6))

    word_index = 0
    confidences = []
    for timestamp in timestamps:
        if not isinstance(timestamp, dict) or "token" not in timestamp:
            continue
        token = str(timestamp["token"])
        stripped = token.strip()
        if not stripped or ("�" not in stripped and _is_punctuation_text(stripped)):
            confidences.append(None)
            continue
        if token[:1].isspace() and confidences:
            word_index += 1
        confidence = word_confidences[word_index] if word_index < len(word_confidences) else None
        confidences.append(confidence)
    return confidences if any(confidence is not None for confidence in confidences) else None



def _is_punctuation_text(text: str) -> bool:
    return bool(re.fullmatch(r"[^\w\s]+", text, flags=re.UNICODE))



def _token_confidences_from_ctc_log_probs(
    inner_model: Any,
    ctc_log_probs: Any,
    timestamps: list[dict[str, Any]],
) -> list[float] | None:
    token_ids = ctc_log_probs.argmax(dim=-1)
    unique_ids = _unique_nonblank_ids(token_ids, getattr(inner_model, "blank_id", 0))
    frame_scores = ctc_log_probs.exp()
    window_confidences = _timestamp_window_confidences(frame_scores, timestamps)
    cursor = 0
    confidences = []
    for timestamp_index, timestamp in enumerate(timestamps):
        if not isinstance(timestamp, dict) or "token" not in timestamp:
            continue
        piece_ids = _encode_ctc_piece(inner_model, str(timestamp["token"]))
        if not piece_ids:
            confidences.append(window_confidences[timestamp_index])
            continue
        piece_scores = []
        next_cursor = cursor
        matched = True
        for piece_id in piece_ids:
            match_index = _next_matching_id_index(unique_ids, piece_id, next_cursor)
            if match_index is None:
                matched = False
                break
            frame_index = unique_ids[match_index][0]
            piece_scores.append(float(frame_scores[frame_index, piece_id].item()))
            next_cursor = match_index + 1
        if not matched or not piece_scores:
            confidences.append(window_confidences[timestamp_index])
            continue
        cursor = next_cursor
        confidences.append(round(min(piece_scores), 6))
    return confidences if any(confidence is not None for confidence in confidences) else None


def _timestamp_window_confidences(
    frame_scores: Any,
    timestamps: list[dict[str, Any]],
) -> list[float | None]:
    total_frames = frame_scores.shape[0]
    time_bounds = [
        (
            timestamp.get("start_time"),
            timestamp.get("end_time"),
        )
        for timestamp in timestamps
        if isinstance(timestamp, dict)
    ]
    numeric_ends = [end for _start, end in time_bounds if isinstance(end, (int, float))]
    max_end = max(numeric_ends) if numeric_ends else None
    confidences = []
    for timestamp in timestamps:
        if not isinstance(timestamp, dict):
            confidences.append(None)
            continue
        start = timestamp.get("start_time")
        end = timestamp.get("end_time")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or not max_end:
            confidences.append(None)
            continue
        start_index = max(0, min(total_frames - 1, int((float(start) / float(max_end)) * total_frames)))
        end_index = max(start_index + 1, min(total_frames, math.ceil((float(end) / float(max_end)) * total_frames)))
        window_scores = frame_scores[start_index:end_index]
        if window_scores.numel() == 0:
            confidences.append(None)
            continue
        confidences.append(round(float(window_scores.max(dim=-1).values.max().item()), 6))
    return confidences



def _unique_nonblank_ids(token_ids: Any, blank_id: int) -> list[tuple[int, int]]:
    unique = []
    previous = None
    for frame_index, token_id_tensor in enumerate(token_ids):
        token_id = int(token_id_tensor.item())
        if token_id == previous:
            continue
        previous = token_id
        if token_id == blank_id:
            continue
        unique.append((frame_index, token_id))
    return unique


def _encode_ctc_piece(inner_model: Any, piece: str) -> list[int]:
    text = _clean_piece_for_ctc(piece)
    if not text:
        return []
    try:
        ids = inner_model.ctc_tokenizer.encode(text)
    except Exception:
        return []
    return [int(token_id) for token_id in ids if int(token_id) != getattr(inner_model, "blank_id", 0)]


def _clean_piece_for_ctc(piece: str) -> str:
    return re.sub(r"\s+", " ", piece).strip()


def _next_matching_id_index(
    unique_ids: list[tuple[int, int]],
    target_id: int,
    start_index: int,
) -> int | None:
    for index in range(start_index, len(unique_ids)):
        if unique_ids[index][1] == target_id:
            return index
    return None


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
        confidence = confidences[confidence_index]
        if confidence is not None:
            timestamp["token_confidence"] = round(confidence, 6)
            timestamp["confidence_source"] = "ctc_log_probs"
        confidence_index += 1
    return result


def _first_result_item(result: Any) -> dict[str, Any] | None:
    if isinstance(result, list) and result and isinstance(result[0], dict):
        return result[0]
    if isinstance(result, dict):
        return result
    return None


def describe_model_for_logits(model: Any) -> dict[str, Any]:
    attributes = [name for name in dir(model) if not name.startswith("_")]
    return {
        "class": type(model).__name__,
        "module": type(model).__module__,
        "attributes": sorted(attributes),
    }


def describe_nested_model_for_logits(model: Any) -> dict[str, Any]:
    inner_model = getattr(model, "model", None)
    kwargs = getattr(model, "kwargs", {})
    return {
        "outer": describe_model_for_logits(model),
        "inner": describe_model_for_logits(inner_model) if inner_model is not None else None,
        "inner_signatures": _callable_signatures(
            inner_model,
            ["forward", "inference", "inference_prepare", "inference_llm", "encode", "decode", "ctc", "ctc_decoder"],
        ),
        "inner_source": _callable_source_excerpts(
            inner_model,
            ["inference", "inference_prepare", "inference_llm", "encode"],
        ),
        "component_types": _component_types(
            inner_model,
            ["audio_encoder", "audio_adaptor", "ctc", "ctc_decoder", "ctc_tokenizer", "llm"],
        ),
        "kwargs_keys": sorted(kwargs.keys()) if isinstance(kwargs, dict) else [],
    }


def _component_types(obj: Any, names: list[str]) -> dict[str, str]:
    components = {}
    if obj is None:
        return components
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            components[name] = f"{type(value).__module__}.{type(value).__name__}"
    return components


def _callable_source_excerpts(obj: Any, names: list[str]) -> dict[str, str]:
    excerpts = {}
    if obj is None:
        return excerpts
    for name in names:
        value = getattr(obj, name, None)
        if not callable(value):
            continue
        try:
            source = inspect.getsource(value)
        except (OSError, TypeError):
            excerpts[name] = "<source unavailable>"
            continue
        lines = source.splitlines()
        excerpts[name] = "\n".join(lines[:120])
    return excerpts


def _callable_signatures(obj: Any, names: list[str]) -> dict[str, str]:
    signatures = {}
    if obj is None:
        return signatures
    for name in names:
        value = getattr(obj, name, None)
        if not callable(value):
            continue
        try:
            signatures[name] = str(inspect.signature(value))
        except (TypeError, ValueError):
            signatures[name] = "<signature unavailable>"
    return signatures
