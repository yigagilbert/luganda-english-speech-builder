# Luganda–English Paired Speech Dataset Pipeline

A production-grade, modular Python pipeline for building a structured
**Luganda–English** bilingual audio dataset from open-source Hugging Face resources.

## Target Schema

| Column | Type | Source |
|---|---|---|
| `id` | `string` | Generated (`{source}_{split}_{idx:07d}`) |
| `audio_lug` | `Audio(16kHz)` | Original Luganda audio from HF datasets |
| `text_lug` | `string` | Original Luganda transcript |
| `text_eng` | `string` | MT output — NLLB-200 or Sunbird MT |
| `audio_eng` | `Audio(16kHz)` | TTS output — Sunbird TTS / SpeechT5 |

---

## Project Structure

```
luganda_pipeline/
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
│
├── config/
│   └── config.yaml            # All pipeline knobs in one place
│
├── luganda_pipeline/          # Main package
│   ├── __init__.py
│   ├── pipeline.py            # Orchestrator — runs all stages end-to-end
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── loader.py          # Stage 1: Load & normalise HF datasets
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── audio.py           # Stage 2: Resample, VAD trim, normalise
│   │
│   ├── filtering/
│   │   ├── __init__.py
│   │   └── text.py            # Stage 3: Text clean, SNR filter, dedup
│   │
│   ├── translation/
│   │   ├── __init__.py
│   │   └── translate.py       # Stage 4: Lug→Eng via NLLB-200 / Sunbird
│   │
│   ├── tts/
│   │   ├── __init__.py
│   │   └── synthesize.py      # Stage 5: English TTS (Sunbird / SpeechT5)
│   │
│   ├── assembly/
│   │   ├── __init__.py
│   │   └── build.py           # Stage 6: Schema assembly & HF Hub push
│   │
│   ├── qa/
│   │   ├── __init__.py
│   │   └── report.py          # Stage 7: QA stats, plots, dataset card
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py         # Structured logger with Rich
│       ├── audio_utils.py     # Shared audio helpers
│       └── checkpoint.py      # Resume-from-checkpoint logic
│
├── scripts/
│   ├── run_pipeline.sh        # Full run (bash entry point)
│   └── run_stage.sh           # Run a single named stage
│
└── tests/
    ├── __init__.py
    ├── test_ingestion.py
    ├── test_preprocessing.py
    ├── test_translation.py
    └── test_assembly.py
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — add HF_TOKEN and (optionally) SUNBIRD_API_KEY
```

Edit `config/config.yaml` to adjust source datasets, filtering thresholds,
batch sizes, output paths, and Hub repo name.

### 3. Run the full pipeline

```bash
python -m luganda_pipeline.pipeline
```

Or with the shell helper:

```bash
bash scripts/run_pipeline.sh
```

### 4. Run a single stage

```bash
python -m luganda_pipeline.pipeline --stage ingestion
python -m luganda_pipeline.pipeline --stage preprocessing
python -m luganda_pipeline.pipeline --stage filtering
python -m luganda_pipeline.pipeline --stage translation
python -m luganda_pipeline.pipeline --stage tts
python -m luganda_pipeline.pipeline --stage assembly
python -m luganda_pipeline.pipeline --stage qa
```

### 5. Resume from checkpoint

The pipeline auto-saves a checkpoint after each stage under `data/checkpoints/`.
Re-running any command will automatically resume from the last completed stage.

---

## Pipeline Stages

```
[1] Ingestion     Pull & unify HF datasets → data/raw/
[2] Preprocessing Resample + VAD trim       → data/preprocessed/
[3] Filtering     SNR / text / dedup        → data/filtered/
[4] Translation   Lug→Eng NLLB-200          → data/translated/
[5] TTS           text_eng → audio_eng      → data/synthesized/
[6] Assembly      Enforce schema + push Hub → data/final/
[7] QA Report     Stats, plots, card        → data/reports/
```

---

## Source Datasets

| Dataset | HF ID | Est. Hours |
|---|---|---|
| Sunbird SALT | `Sunbird/salt` | ~120 h |
| Common Voice 17 | `mozilla-foundation/common_voice_17_0` | ~30 h |
| FLEURS | `google/fleurs` | ~12 h |
| CMU Wilderness | `cmu-wilderness/lug` | ~20 h |
| ALFFA | community upload | ~8 h |

> ⚠ Always verify licences before publishing derivative datasets.
> Common Voice: CC0 · FLEURS: CC-BY-4.0 · SALT: check Sunbird terms · CMU Wilderness: research-only.

---

## Configuration Reference (`config/config.yaml`)

| Key | Default | Description |
|---|---|---|
| `audio.sample_rate` | `16000` | Target sample rate (Hz) |
| `audio.min_duration_s` | `0.5` | Minimum clip duration |
| `audio.max_duration_s` | `30.0` | Maximum clip duration |
| `audio.min_snr_db` | `15.0` | Minimum SNR (dB) |
| `filter.max_cps` | `25.0` | Max characters per second |
| `translation.model` | `facebook/nllb-200-distilled-600M` | MT model |
| `translation.batch_size` | `32` | Translation batch size |
| `tts.model` | `microsoft/speecht5_tts` | TTS model |
| `tts.batch_size` | `16` | TTS batch size |
| `hub.repo_id` | `your-org/luganda-english-speech` | HF Hub target |

---

## Requirements

- Python ≥ 3.10
- CUDA-capable GPU recommended (A100/V100 for translation + TTS at scale)
- ~500 GB disk space for full pipeline run
- Hugging Face account with write token

---

## Estimated Compute

| Stage | A100 GPU | CPU-only |
|---|---|---|
| Ingestion | ~30 min | ~1 h |
| Preprocessing | ~1–2 h | ~6 h |
| Filtering | ~30 min | ~30 min |
| Translation | ~3–5 h | ~24 h |
| TTS synthesis | ~10–16 h | not feasible |
| Assembly + QA | ~1 h | ~1 h |

---

## Licence

This pipeline code is released under the **MIT Licence**.
Dataset output licences are determined by the upstream source datasets.
