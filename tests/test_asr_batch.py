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


def test_success_output_requires_matching_id_text_and_word_level_timestamps(tmp_path):
    output = tmp_path / "sample.wav.json"
    output.write_text(
        json.dumps(
            [
                {
                    "id": "sample.wav",
                    "text": "xin chào",
                    "timestamps": [
                        {"word": "xin", "start": 0.0, "end": 0.2, "confidence": 0.9},
                        {"word": "chào", "start": 0.2, "end": 0.5, "confidence": 0.8},
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert is_success_output(output, "sample.wav") is True
    assert is_success_output(output, "other.wav") is False


def test_success_output_rejects_empty_timestamps(tmp_path):
    output = tmp_path / "sample.wav.json"
    output.write_text(
        json.dumps(
            [{"id": "sample.wav", "text": "xin chào", "timestamps": []}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert is_success_output(output, "sample.wav") is False


def test_success_output_rejects_old_funasr_token_level_timestamps(tmp_path):
    output = tmp_path / "sample.wav.json"
    output.write_text(
        json.dumps(
            [
                {
                    "id": "sample.wav",
                    "text": "xin chào",
                    "timestamps": [
                        {"token": "xin", "score": 0.9, "start_time": 0.0, "end_time": 0.2},
                        {"token": " chào", "score": 0.8, "start_time": 0.2, "end_time": 0.5},
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert is_success_output(output, "sample.wav") is False


def test_invalid_or_empty_output_is_not_success(tmp_path):
    invalid = tmp_path / "bad.wav.json"
    invalid.write_text("not json", encoding="utf-8")

    empty_text = tmp_path / "empty.wav.json"
    empty_text.write_text(
        json.dumps(
            [
                {
                    "id": "empty.wav",
                    "text": "",
                    "timestamps": [
                        {"word": "xin", "start": 0.0, "end": 0.2},
                    ],
                }
            ],
            ensure_ascii=False,
        ),
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


def test_normalize_asr_result_empty_list_has_empty_text():
    normalized = normalize_asr_result("sample.wav", [])

    assert normalized == [{"id": "sample.wav", "text": "", "timestamps": []}]


def test_normalize_asr_result_preserves_timestamp_pairs_without_confidence():
    raw = {"text": "xin chào", "timestamp": [[0, 500], [500, 900]]}

    normalized = normalize_asr_result("sample.wav", raw)

    assert normalized[0]["timestamps"] == [
        {"start": 0, "end": 500},
        {"start": 500, "end": 900},
    ]


def test_normalize_asr_result_merges_funasr_token_timestamps_to_word_level():
    raw = {
        "text": "xin chào",
        "timestamps": [
            {"token": "xin", "score": 0.9, "start_time": 0.0, "end_time": 0.2},
            {"token": " chào", "score": 0.8, "start_time": 0.2, "end_time": 0.5},
        ],
    }

    normalized = normalize_asr_result("sample.wav", raw)

    assert normalized[0]["timestamps"] == [
        {"word": "xin", "start": 0.0, "end": 0.2, "confidence": 0.9},
        {"word": "chào", "start": 0.2, "end": 0.5, "confidence": 0.8},
    ]


def test_normalize_asr_result_merges_subword_tokens_and_uses_transcript_words():
    raw = {
        "text": "Ứng miễn dịch",
        "timestamps": [
            {"token": "�", "score": 0.0, "start_time": 0.0, "end_time": 0.06},
            {"token": "�", "score": 0.0, "start_time": 0.06, "end_time": 0.12},
            {"token": "ng", "score": 0.0, "start_time": 0.12, "end_time": 0.18},
            {"token": " mi", "score": 0.7, "start_time": 0.18, "end_time": 0.24},
            {"token": "ễ", "score": 0.8, "start_time": 0.24, "end_time": 0.3},
            {"token": "n", "score": 0.9, "start_time": 0.3, "end_time": 0.36},
            {"token": " d", "score": 0.6, "start_time": 0.36, "end_time": 0.42},
            {"token": "ịch", "score": 0.8, "start_time": 0.42, "end_time": 0.48},
        ],
    }

    normalized = normalize_asr_result("sample.wav", raw)

    assert normalized[0]["timestamps"] == [
        {"word": "Ứng", "start": 0.0, "end": 0.18, "confidence": 0.0},
        {"word": "miễn", "start": 0.18, "end": 0.36, "confidence": 0.8},
        {"word": "dịch", "start": 0.36, "end": 0.48, "confidence": 0.7},
    ]


def test_normalize_asr_result_attaches_punctuation_to_previous_word():
    raw = {
        "text": "xin chào,",
        "timestamps": [
            {"token": "xin", "score": 0.9, "start_time": 0.0, "end_time": 0.2},
            {"token": " chào", "score": 0.8, "start_time": 0.2, "end_time": 0.5},
            {"token": ",", "score": 0.7, "start_time": 0.5, "end_time": 0.55},
        ],
    }

    normalized = normalize_asr_result("sample.wav", raw)

    assert normalized[0]["timestamps"] == [
        {"word": "xin", "start": 0.0, "end": 0.2, "confidence": 0.9},
        {"word": "chào,", "start": 0.2, "end": 0.55, "confidence": 0.75},
    ]


def test_atomic_write_json_writes_final_file_without_tmp(tmp_path):
    output = tmp_path / "sample.wav.json"
    payload = [{"id": "sample.wav", "text": "xin chào", "timestamps": []}]

    atomic_write_json(output, payload)

    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert not list(tmp_path.glob("sample.wav.json.*.tmp"))


def test_error_record_has_required_fields():
    record = error_record(Path("data/sample.wav"), "inference failed", "modal")

    assert record["audio_path"] == "data/sample.wav"
    assert record["filename"] == "sample.wav"
    assert record["error"] == "inference failed"
    assert record["stage"] == "modal"
    assert "timestamp" in record