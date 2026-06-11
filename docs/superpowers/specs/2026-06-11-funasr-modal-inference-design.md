# FunASR Modal Inference Design

## Goal

Build a repeatable pseudo-labeling pipeline for VietMed ASR audio files, matching the existing Week 3 pseudo-labeling workflow style but replacing the Qwen3 ASR stack with FunASR on Modal.com.

The user will place audio files in a local `data/` directory. A local CLI entrypoint will send each unprocessed file to a Modal GPU function, receive the ASR result, and write one JSON output per audio file. Re-running the command must skip files that already have successful ASR output so long jobs can resume without starting from zero.

## Project layout

```text
D:\Fun-ASR-VietMedLabeling\
  data\
    *.wav
    <audio filename>.json
    _asr_errors.jsonl
  modal_funasr_infer.py
  requirements.txt
  README.md
```

`data/` is created in the repository so the user can add audio files later. The code scans this folder recursively for supported audio extensions, with `.wav` as the primary expected format.

## Recommended approach

Use a local batch script plus a Modal remote inference function.

Why this approach:

- It matches the user's desired local `data/` workflow.
- It avoids requiring a Modal Volume upload/sync step.
- It supports checkpoint/resume naturally by checking local JSON outputs.
- It keeps the public output format simple: one JSON file per audio file.

Alternatives considered:

1. Modal Volume batch processing: better for very large datasets, but setup and sync are more complex.
2. Modal FastAPI endpoint: closer to the old FastAPI notebook architecture, but unnecessary for a local batch labeling workflow and adds deployment/API management overhead.

## Model configuration

The Modal function uses FunASR with:

```python
MODEL_ID = "FunAudioLLM/Fun-ASR-Nano-2512"
DEVICE = "cuda:0"
HUB = "hf"
VAD_MODEL = "fsmn-vad"
```

The model is loaded with `funasr.AutoModel`. VAD is enabled to make longer audio files more robust. The function calls `model.generate(...)` with batch size 1 for reliability.

FunASR documentation for this model documents transcript generation and sentence-level `sentence_info` for diarization-style output. It does not clearly guarantee word-level confidence for this model. Therefore, output normalization is best-effort:

- If FunASR returns word-level timestamps/confidence, preserve them.
- If it returns sentence-level spans, keep transcript and expose no fabricated word confidence.
- If only transcript is available, output `timestamps: []`.

The pipeline must not invent confidence values or fake word timings, because downstream quality scoring depends on those fields being trustworthy.

## Output schema

For each input audio file, write:

```text
data/<original audio filename>.json
```

For example:

```text
data/VietMed_un_001_s05OFV.wav.json
```

The JSON content is an array with one object:

```json
[
  {
    "id": "VietMed_un_001_s05OFV.wav",
    "text": "...",
    "timestamps": [
      {
        "word": "áp",
        "confidence": 0.4609244167804718,
        "start": 0.0,
        "end": 0.16
      }
    ]
  }
]
```

The public per-audio JSON stays close to the user's required format. Operational failure details are written separately to `data/_asr_errors.jsonl` rather than polluting successful output files.

## Resume and checkpoint behavior

A file is considered successfully processed when its output JSON:

- exists,
- parses as JSON,
- is a non-empty list,
- first item has `id` matching the input filename,
- first item has non-empty string `text`,
- first item has a `timestamps` key, even if it is an empty list.

If these checks pass, the local entrypoint skips that audio file.

If output is missing, invalid JSON, has empty text, or is structurally incomplete, the script runs inference again.

Writes are atomic:

1. Write `data/<audio>.json.tmp`.
2. Flush the full JSON.
3. Replace/rename to `data/<audio>.json`.

This prevents interrupted runs from leaving a partial file that looks successful.

## Error handling

The batch run should continue when one file fails.

For each failed file, append a JSON line to:

```text
data/_asr_errors.jsonl
```

Each error record includes:

- audio path,
- filename,
- error message,
- stage if known,
- timestamp in ISO format.

The CLI prints a final summary:

- total discovered audio files,
- skipped existing successes,
- newly successful files,
- failed files,
- output directory.

## Testing and verification

Implementation should include tests for pure local logic where practical:

- success-output detection,
- output filename mapping,
- FunASR result normalization,
- atomic JSON writing.

Manual verification requires at least one audio file in `data/` and a configured Modal account. The expected command is:

```powershell
modal run modal_funasr_infer.py
```

If no audio files exist, the command should exit cleanly and tell the user to add audio files to `data/`.

## Scope boundaries

In scope:

- Create local `data/` folder.
- Add Modal FunASR batch inference script.
- Add minimal Python requirements.
- Add README instructions.
- Implement resume/skip behavior.
- Produce one JSON output per audio file.

Out of scope for the initial implementation:

- Modal Volume upload/sync workflow.
- Long-running deployed API endpoint.
- Dataset-wide scoring report.
- Forced alignment with a separate aligner.
- Fabricated word-level timestamps/confidence when the model does not return them.

These can be added later after confirming FunASR's actual output fields on representative VietMed audio.
