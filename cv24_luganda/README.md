# Common Voice 24.0 — Luganda Processor

Dedicated processor for **Mozilla Common Voice Scripted Speech 24.0 — Luganda (`lg`)**.

Source: https://datacollective.mozillafoundation.org/datasets/cmj8u3pcu00elnxxby6wyhysl

---

## Dataset Facts

| Property | Value |
|---|---|
| Version | 24.0 |
| Language | Luganda (`lg`) |
| Archive | `mcv-scripted-lg-v24.0.tar.gz` (11.03 GB) |
| Total clips | 348,763 |
| Total hours | 560.35 h |
| Validated hours | 436.83 h |
| Speakers | 665 |
| Format | MP3 in a flat `clips/` directory |
| Metadata | TSV files per split |
| Licence | **CC0-1.0** (public domain) |
| Train split | 71,087 clips |
| Dev split | 13,352 clips |
| Test split | 13,388 clips |

---

## Files

```
cv24_luganda/
├── common_voice_24_luganda.py   ← Main processor (raw archive → HF Dataset)
├── common_voice_24_hf_loader.py ← Alternative: pull from HF Hub directly
├── requirements.txt
└── tests/
    └── test_cv24_processor.py
```

---

## Quick Start

### Option A — Process from raw archive (recommended for CV 24.0)

```bash
pip install -r requirements.txt

# Get your signed download URL from the Mozilla Data Collective website, then:
export CV24_DOWNLOAD_URL="https://your-signed-url-here..."
export HF_TOKEN="hf_xxxxx"          # only needed for --push-to-hub

# Full run: download → extract → filter → process → save
python common_voice_24_luganda.py

# Skip the download if you already have the archive:
python common_voice_24_luganda.py --skip-download

# Process only the validated split with stricter filters:
python common_voice_24_luganda.py \
    --splits validated \
    --min-up-votes 3 \
    --min-snr 18.0 \
    --push-to-hub your-org/cv24-luganda
```

### Option B — Pull from HuggingFace Hub

```bash
# Requires HF account with Common Voice access approved
export HF_TOKEN="hf_xxxxx"

python common_voice_24_hf_loader.py --splits train,validation

# Stream without full download (slower per-example but no disk pre-req):
python common_voice_24_hf_loader.py --streaming
```

---

## CLI Reference

### `common_voice_24_luganda.py`

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | `data/cv24_luganda` | Local working directory |
| `--output-dir` | `data/cv24_luganda/processed` | Output dataset path |
| `--download-url` | env `CV24_DOWNLOAD_URL` | Signed URL for the tar.gz archive |
| `--skip-download` | false | Skip download (archive already present) |
| `--skip-extract` | false | Skip extraction (already extracted) |
| `--splits` | `train,dev,test` | Comma-separated TSV splits |
| `--validated-only` | true | Accept only clips with `up_votes > down_votes` |
| `--min-up-votes` | `2` | Minimum community up-votes |
| `--min-duration` | `0.5` | Minimum clip duration (seconds) |
| `--max-duration` | `30.0` | Maximum clip duration (seconds) |
| `--min-snr` | `15.0` | Minimum estimated SNR (dB) |
| `--min-text-len` | `2` | Minimum transcript character count |
| `--max-text-len` | `500` | Maximum transcript character count |
| `--target-sr` | `16000` | Output sample rate (Hz) |
| `--num-workers` | `cpu_count - 1` | Parallel worker processes |
| `--push-to-hub` | false | Push result to HuggingFace Hub |
| `--hub-repo-id` | | HF Hub repo ID |
| `--dry-run` | false | Print stats only, no output saved |

---

## Processing Pipeline

```
[1] Download      Resumable HTTPS download of mcv-scripted-lg-v24.0.tar.gz (11 GB)
[2] Extract       Safe tar.gz extraction with path-traversal protection
[3] Load TSVs     Parse train / dev / test / validated TSV splits
                  Fields: client_id, path, text, up_votes, down_votes, age, gender, accent
[4] Meta-filter   Remove: missing text/audio, down-voted clips, short transcripts,
                  clips whose MP3 file doesn't exist on disk
[5] Audio proc    Per-clip in parallel workers:
                    - torchaudio.load (MP3 → float32 tensor)
                    - Mono conversion
                    - Resample → 16 000 Hz
                    - Duration gate  [0.5 s – 30 s]
                    - Energy-based SNR estimate  ≥ 15 dB
                    - Peak normalise to -1 dBFS
                    - Encode to 16-bit PCM WAV bytes
[6] Assemble      HuggingFace Dataset:  id | audio_lug | text_lug
                  Cast audio_lug to Audio(sampling_rate=16000)
                  Save to disk (Arrow format) + optional Hub push
```

---

## Output Schema

| Column | Type | Description |
|---|---|---|
| `id` | `string` | `cv24_lg_{index:07d}` |
| `audio_lug` | `Audio(16kHz)` | 16-bit mono WAV bytes |
| `text_lug` | `string` | NFC-normalised Luganda transcript |

This output is ready to be consumed by **Stage 4 — Translation** of the
main Luganda–English speech pipeline.

---

## Quality Filters Applied

| Filter | Default | Rationale |
|---|---|---|
| `up_votes > down_votes` | enabled | Community validation — ensures transcript matches audio |
| `up_votes >= 2` | enabled | At least 2 independent listeners confirmed the clip |
| Duration 0.5–30 s | enabled | Removes near-silent clips and very long recordings |
| SNR ≥ 15 dB | enabled | Rejects noisy background recordings |
| Text length 2–500 chars | enabled | Removes obviously malformed transcripts |
| Unicode NFC + whitespace | always | Standardises text encoding |

---

## Expected Yield

Starting from the 97,827 clips in train+dev+test splits:

| Stage | Approx. remaining |
|---|---|
| After validation filter | ~87,000 (89%) |
| After min up-votes ≥ 2 | ~72,000 (74%) |
| After duration gate | ~68,000 (70%) |
| After SNR gate | ~60,000–65,000 (65%) |
| **Final estimated yield** | **~60,000–65,000 clips / ~85–95 h** |

To maximise yield, run with `--splits validated` to include all 436h of validated clips
rather than only the official train/dev/test splits.

---

## Licence

This processor code is released under the **MIT Licence**.

The Common Voice 24.0 dataset is released under **CC0-1.0 (Public Domain)**.
You may use, modify, and distribute it freely.

> ⚠ Forbidden by Mozilla's terms: attempting to identify speakers or re-hosting the raw dataset.
