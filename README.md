# VietMed FunASR Modal Labeling

**Repository:** [https://github.com/phuvo05/Fun-ASR-VietMedLabeling](https://github.com/phuvo05/Fun-ASR-VietMedLabeling)

This repository runs pseudo-labeling for VietMed audio files with FunASR on Modal.com.

## Workflow

1. Put audio files in `data/`.
2. Run the Modal batch command.
3. Each successful audio file gets a JSON output next to it.
4. Re-running the command skips files that already have successful output.

Example input:

```text
data/VietMed_un_001_s05OFV.wav
```

Example output:

```text
data/VietMed_un_001_s05OFV.wav.json
```

Output JSON format:

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

If FunASR does not return timestamp entries for a file, `timestamps` is an empty list. If FunASR returns timing entries without confidence, the script preserves the timing fields and does not fabricate confidence values.

## Setup

Install local dependencies:

```powershell
python -m pip install -r requirements.txt
```

Log in to Modal if needed:

```powershell
modal setup
```

## Run

```powershell
modal run modal_funasr_infer.py
```

Use a custom data folder:

```powershell
modal run modal_funasr_infer.py --data-dir "D:\path\to\audio_data"
```

Use a custom batch size:

```powershell
modal run modal_funasr_infer.py --batch-size 5
```

## Resume behavior

A file is skipped when its output JSON exists, parses correctly, has matching `id`, has non-empty `text`, and contains a `timestamps` list.

Invalid or partial output files are re-run.

Failures are appended to `_asr_errors.jsonl` inside the selected data folder. With the default folder, that path is:

```text
data/_asr_errors.jsonl
```

With `--data-dir "D:\path\to\audio_data"`, failures are written to:

```text
D:\path\to\audio_data\_asr_errors.jsonl
```

The batch continues after individual file failures.

## Tests

```powershell
python -m pytest -v
```
