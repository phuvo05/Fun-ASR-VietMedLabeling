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

    error_log = root / "_asr_errors.jsonl"
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
                error_log,
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
