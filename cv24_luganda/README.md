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
├── ../scripts/run_cv24_gpu_fast.sh ← GPU-optimized launcher for rented instances
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
    --push-to-hub \
    --hub-repo-id your-org/cv24-luganda

# Build full bilingual schema (audio_lug + text_lug + text_eng + audio_eng)
# and push it to a DIFFERENT Hugging Face repo:
python common_voice_24_luganda.py \
    --splits validated \
    --run-general-steps \
    --push-paired-to-hub \
    --paired-hub-repo-id your-org/cv24-lug-eng
```

### Option B — Pull from HuggingFace Hub

```bash
# Requires HF account with Common Voice access approved
export HF_TOKEN="hf_xxxxx"

python common_voice_24_hf_loader.py --splits train,validation

# Stream without full download (slower per-example but no disk pre-req):
python common_voice_24_hf_loader.py --streaming
```

### Option C — Rented GPU (fastest path)

```bash
# 1) Prepare env
export CV24_DOWNLOAD_URL="https://your-mozilla-signed-url"
export HF_TOKEN="hf_xxx"
export PAIRED_REPO_ID="your-org/cv24-lug-eng"     # bilingual final dataset
# optional
export LUGANDA_REPO_ID="your-org/cv24-lug-only"    # lug-only intermediate dataset

# 2) Run with GPU-optimized defaults
bash scripts/run_cv24_gpu_fast.sh
```

By default this script:
- uses `splits=validated` (max usable data),
- runs `--run-general-steps` (translation + TTS + final assembly),
- pushes paired output to `PAIRED_REPO_ID`,
- places HF caches on fast local disk (`/mnt/nvme` or `/local` if present),
- defaults to `PROFILE=balanced`,
- prints an overall end-to-end terminal progress bar (plus per-step bars).

Profiles:

```bash
# Fastest throughput, lower MT quality than the fine-tuned model
PROFILE=fast bash scripts/run_cv24_gpu_fast.sh

# Recommended default on A100-class GPUs
PROFILE=balanced bash scripts/run_cv24_gpu_fast.sh

# Slower, stricter filtering, strongest translation quality
PROFILE=accurate bash scripts/run_cv24_gpu_fast.sh
```

To use a different config entirely:

```bash
export PIPELINE_CONFIG="/path/to/your_config.yaml"
bash scripts/run_cv24_gpu_fast.sh
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
| `--run-general-steps` | false | Run shared Stage 4/5/6 (translation, TTS, assembly) after CV24 build |
| `--pipeline-config` | `config/config.yaml` | Main pipeline YAML for translation/TTS settings |
| `--paired-output-dir` | `<output-dir>/paired_general_steps` | Local output root for bilingual artifacts |
| `--push-paired-to-hub` | false | Push bilingual final dataset to HuggingFace Hub |
| `--paired-hub-repo-id` | | HF Hub repo ID for bilingual dataset |
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

Optional after step 6 (`--run-general-steps`):
[7] Translation   text_lug -> text_eng
[8] TTS           text_eng -> audio_eng
[9] Assembly      Final bilingual schema:
                  id | audio_eng | audio_lug | text_eng | text_lug
                  Save + optional push to a separate HF repo via --paired-hub-repo-id
```

---

## Output Schemas

### Base CV24 output
| Column | Type | Description |
|---|---|---|
| `id` | `string` | `cv24_lg_{index:07d}` |
| `audio_lug` | `Audio(16kHz)` | 16-bit mono WAV bytes |
| `text_lug` | `string` | NFC-normalised Luganda transcript |

### Optional bilingual output (`--run-general-steps`)
| Column | Type | Description |
|---|---|---|
| `id` | `string` | Generated final ID (`lug_eng_{index:07d}`) |
| `audio_eng` | `Audio(16kHz)` | English TTS audio |
| `audio_lug` | `Audio(16kHz)` | Luganda audio |
| `text_eng` | `string` | English translation |
| `text_lug` | `string` | Luganda transcript |

This bilingual output is what gets pushed when you use `--push-paired-to-hub`.

---

## GPU Throughput Tuning (Rented Machines)

For highest throughput:
- Use an instance with high VRAM and local NVMe (A100 80GB preferred).
- Keep caches on local NVMe (`HF_HOME`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE`).
- Use `PROFILE=balanced` on A100/H100. Use `PROFILE=fast` on L4/T4 if memory is tight.
- `PROFILE=accurate` raises `min_up_votes` and `min_snr` and uses the slower fine-tuned MT path.
- Increase `translation.batch_size` and `translation.generation_batch_size` only after checking GPU headroom.
- Use `NUM_WORKERS` near vCPU count for CV24 audio preprocessing (set via env in the launcher script).

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
