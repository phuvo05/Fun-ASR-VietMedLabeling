from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import modal

APP_NAME = "vietmed-funasr-labeling"
DATA_DIR = Path("data")
ERROR_LOG = DATA_DIR / "_asr_errors.jsonl"
MODEL_ID = "FunAudioLLM/Fun-ASR-MLT-Nano-2512"
LANGUAGE = "越南语"
BATCH_SIZE = 5
ALLOWED_REMOTE_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def safe_audio_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return suffix if suffix in ALLOWED_REMOTE_SUFFIXES else ".wav"


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

        from funasr_logits_infer import transcribe_with_token_confidence

        suffix = safe_audio_suffix(filename)
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            result = transcribe_with_token_confidence(
                self.model,
                Path(tmp.name),
                LANGUAGE,
            )
        return result


@app.local_entrypoint()
def main(data_dir: str = str(DATA_DIR), batch_size: int = BATCH_SIZE) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from asr_batch import (
        aggregate_output_path_for_audio,
        append_jsonl,
        atomic_write_json,
        chunked,
        discover_audio_files,
        error_record,
        is_success_output,
        merge_result_into_aggregate,
        normalize_asr_result,
    )

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    audio_files = discover_audio_files(root)
    if not audio_files:
        print(f"No audio files found in {root}. Add .wav files and run again.")
        return

    label_dir = root / "label"
    error_log = root / "_asr_errors.jsonl"
    worker = FunASRWorker()
    skipped = 0
    succeeded = 0
    failed = 0

    for batch_index, audio_batch in enumerate(chunked(audio_files, batch_size), start=1):
        print(f"\nBatch {batch_index}: {len(audio_batch)} audio files")
        pending_batch = []
        for audio_path in audio_batch:
            output_path = aggregate_output_path_for_audio(audio_path, label_dir)
            if is_success_output(output_path, audio_path.name):
                skipped += 1
                print(f"SKIP {audio_path}")
                continue
            pending_batch.append(audio_path)

        if not pending_batch:
            continue

        for audio_path in pending_batch:
            print(f"ASR  {audio_path}")

        def transcribe_one(audio_path: Path) -> tuple[Path, Any]:
            audio_bytes = audio_path.read_bytes()
            raw_result = worker.transcribe.remote(audio_bytes, audio_path.name)
            return audio_path, raw_result

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {
                executor.submit(transcribe_one, audio_path): audio_path
                for audio_path in pending_batch
            }
            for future in as_completed(futures):
                audio_path = futures[future]
                try:
                    _, raw_result = future.result()
                    output_path = aggregate_output_path_for_audio(audio_path, label_dir)
                    payload = normalize_asr_result(audio_path.name, raw_result)
                    if not payload[0]["text"]:
                        raise ValueError("FunASR returned empty text")
                    aggregate_payload = merge_result_into_aggregate(output_path, payload)
                    atomic_write_json(output_path, aggregate_payload)
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
