import json
from pathlib import Path

from asr_batch import (
    AUDIO_EXTENSIONS,
    aggregate_output_path_for_audio,
    atomic_write_json,
    discover_audio_files,
    chunked,
    error_record,
    is_success_output,
    merge_result_into_aggregate,
    normalize_asr_result,
)


def test_audio_extensions_include_wav():
    assert ".wav" in AUDIO_EXTENSIONS


def test_aggregate_output_path_uses_label_folder_and_audio_parent_name(tmp_path):
    audio_dir = tmp_path / "data" / "VietMed_unlabeled_1000h_segmented_8kHz_000_050"
    audio = audio_dir / "VietMed_un_001_s05OFV.wav"

    assert aggregate_output_path_for_audio(audio, tmp_path / "label") == (
        tmp_path / "label" / "VietMed_unlabeled_1000h_segmented_8kHz_000_050.json"
    )


def test_chunked_groups_items_by_batch_size():
    assert list(chunked([1, 2, 3, 4, 5, 6], 5)) == [[1, 2, 3, 4, 5], [6]]



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


def test_success_output_finds_matching_audio_inside_folder_aggregate(tmp_path):
    output = tmp_path / "label" / "folder.json"
    output.parent.mkdir()
    output.write_text(
        json.dumps(
            [
                {"id": "first.wav", "text": "một", "timestamps": []},
                {"id": "second.wav", "text": "hai", "timestamps": []},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert is_success_output(output, "second.wav") is True
    assert is_success_output(output, "missing.wav") is False


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


def test_normalize_asr_result_maps_funasr_token_timestamps_to_required_fields():
    raw = {
        "text": "xin chào",
        "timestamps": [
            {"token": "xin", "score": 0.9, "start_time": 0.0, "end_time": 0.2},
            {"token": " chào", "score": 0.8, "start_time": 0.2, "end_time": 0.5},
        ],
    }

    normalized = normalize_asr_result("sample.wav", raw)

    assert normalized[0]["timestamps"] == [
        {"word": "xin", "confidence": 0.9, "start": 0.0, "end": 0.2},
        {"word": "chào", "confidence": 0.8, "start": 0.2, "end": 0.5},
    ]


def test_normalize_asr_result_uses_none_for_placeholder_zero_token_scores():
    raw = {
        "text": "xin chào",
        "timestamps": [
            {"token": "xin", "score": 0.0, "start_time": 0.0, "end_time": 0.2},
            {"token": " chào", "score": 0.0, "start_time": 0.2, "end_time": 0.5},
        ],
    }

    normalized = normalize_asr_result("sample.wav", raw)

    assert normalized[0]["timestamps"] == [
        {"word": "xin", "confidence": None, "start": 0.0, "end": 0.2},
        {"word": "chào", "confidence": None, "start": 0.2, "end": 0.5},
    ]



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



def test_normalize_asr_result_merges_funasr_subword_tokens_to_words():
    raw = {
        "text": "Hợp hoạt động.",
        "timestamps": [
            {"token": "H", "score": 0.8, "start_time": 0.0, "end_time": 0.1},
            {"token": "ợ", "score": 0.6, "start_time": 0.1, "end_time": 0.2},
            {"token": "p", "score": 0.7, "start_time": 0.2, "end_time": 0.3},
            {"token": " ho", "score": 0.9, "start_time": 0.3, "end_time": 0.4},
            {"token": "ạt", "score": 0.7, "start_time": 0.4, "end_time": 0.5},
            {"token": " động", "score": 0.5, "start_time": 0.5, "end_time": 0.8},
            {"token": ".", "score": 1.0, "start_time": 0.8, "end_time": 0.9},
        ],
    }

    normalized = normalize_asr_result("sample.wav", raw)

    assert normalized[0]["timestamps"] == [
        {"word": "Hợp", "confidence": 0.6, "start": 0.0, "end": 0.3},
        {"word": "hoạt", "confidence": 0.7, "start": 0.3, "end": 0.5},
        {"word": "động", "confidence": 0.5, "start": 0.5, "end": 0.8},
        {"word": ".", "confidence": 1.0, "start": 0.8, "end": 0.9},
    ]


def test_merge_result_into_aggregate_appends_new_result(tmp_path):
    output = tmp_path / "label" / "folder.json"
    output.parent.mkdir()
    output.write_text(
        json.dumps([{"id": "first.wav", "text": "một", "timestamps": []}], ensure_ascii=False),
        encoding="utf-8",
    )
    result = [{"id": "second.wav", "text": "hai", "timestamps": []}]

    merged = merge_result_into_aggregate(output, result)

    assert merged == [
        {"id": "first.wav", "text": "một", "timestamps": []},
        {"id": "second.wav", "text": "hai", "timestamps": []},
    ]


def test_merge_result_into_aggregate_replaces_existing_result(tmp_path):
    output = tmp_path / "label" / "folder.json"
    output.parent.mkdir()
    output.write_text(
        json.dumps(
            [
                {"id": "first.wav", "text": "old", "timestamps": []},
                {"id": "second.wav", "text": "hai", "timestamps": []},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = [{"id": "first.wav", "text": "new", "timestamps": []}]

    merged = merge_result_into_aggregate(output, result)

    assert merged == [
        {"id": "first.wav", "text": "new", "timestamps": []},
        {"id": "second.wav", "text": "hai", "timestamps": []},
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
