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
