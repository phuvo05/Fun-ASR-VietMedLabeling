import math
from pathlib import Path

import torch

from funasr_logits_infer import (
    _token_confidences_from_ctc_log_probs,
    describe_model_for_logits,
    describe_nested_model_for_logits,
    transcribe_with_token_confidence,
)


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


class FakeCtcTokenizer:
    def encode(self, text: str):
        return {"xin": [1], "chào": [2], "xin chào": [1, 2]}[text]


class FakeCtcInnerModel:
    blank_id = 0
    ctc_tokenizer = FakeCtcTokenizer()


class FakeInnerModel:
    def inference(self):
        return None

    def forward(self):
        return None


class FakeNestedModel:
    def __init__(self):
        self.model = FakeInnerModel()
        self.kwargs = {"model": "fake"}


def test_describe_model_for_logits_lists_relevant_attributes():
    description = describe_model_for_logits(FakeNestedModel())

    assert description["class"] == "FakeNestedModel"
    assert "model" in description["attributes"]
    assert "generate" not in description["attributes"]


def test_describe_nested_model_for_logits_includes_inner_model_and_signatures():
    description = describe_nested_model_for_logits(FakeNestedModel())

    assert description["outer"]["class"] == "FakeNestedModel"
    assert description["inner"]["class"] == "FakeInnerModel"
    assert "forward" in description["inner_signatures"]
    assert "inference" in description["inner_signatures"]
    assert description["kwargs_keys"] == ["model"]


def test_token_confidences_from_ctc_log_probs_aligns_timestamp_tokens():
    log_probs = torch.log(
        torch.tensor(
            [
                [0.05, 0.90, 0.05],
                [0.80, 0.10, 0.10],
                [0.05, 0.10, 0.85],
            ]
        )
    )
    timestamps = [{"token": "xin"}, {"token": " chào"}]

    confidences = _token_confidences_from_ctc_log_probs(
        FakeCtcInnerModel(),
        log_probs,
        timestamps,
    )

    assert confidences == [0.9, 0.85]



def test_token_confidences_from_ctc_log_probs_keeps_positions_for_unmatched_tokens():
    log_probs = torch.log(
        torch.tensor(
            [
                [0.05, 0.90, 0.05],
                [0.80, 0.10, 0.10],
                [0.05, 0.10, 0.85],
            ]
        )
    )
    timestamps = [{"token": "<?>"}, {"token": "xin"}, {"token": " chào"}]

    confidences = _token_confidences_from_ctc_log_probs(
        FakeCtcInnerModel(),
        log_probs,
        timestamps,
    )

    assert confidences == [None, 0.9, 0.85]



def test_token_confidences_from_ctc_log_probs_falls_back_to_timestamp_windows():
    log_probs = torch.log(
        torch.tensor(
            [
                [0.05, 0.90, 0.05],
                [0.05, 0.80, 0.15],
                [0.05, 0.10, 0.85],
                [0.05, 0.15, 0.80],
            ]
        )
    )
    timestamps = [
        {"token": "<?>" , "start_time": 0.0, "end_time": 0.5},
        {"token": " ???", "start_time": 0.5, "end_time": 1.0},
    ]

    confidences = _token_confidences_from_ctc_log_probs(
        FakeCtcInnerModel(),
        log_probs,
        timestamps,
    )

    assert confidences == [0.9, 0.85]



def test_token_confidences_from_ctc_log_probs_uses_min_for_multi_token_piece():
    log_probs = torch.log(
        torch.tensor(
            [
                [0.05, 0.80, 0.10, 0.05],
                [0.05, 0.10, 0.70, 0.15],
            ]
        )
    )
    timestamps = [{"token": "xin chào"}]

    confidences = _token_confidences_from_ctc_log_probs(
        FakeCtcInnerModel(),
        log_probs,
        timestamps,
    )

    assert confidences == [0.7]



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
            "confidence_source": "ctc_log_probs",
        },
        {
            "token": " chào",
            "score": 0.0,
            "start_time": 0.2,
            "end_time": 0.5,
            "token_confidence": 0.73,
            "confidence_source": "ctc_log_probs",
        },
    ]
