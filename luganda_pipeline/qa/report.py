"""
Stage 7 — QA Report & Dataset Card
====================================
Generates:
  - Console stats summary (duration, split sizes, text lengths, SNR distribution)
  - Matplotlib plots: duration histogram, text-length distribution, SNR distribution
  - A spot-check CSV (N random pairs exported for human MT review)
  - A Markdown dataset card written to data/reports/dataset_card.md
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torchaudio
from datasets import Dataset

from luganda_pipeline.utils.audio_utils import get_duration_s
from luganda_pipeline.utils.logging import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Stats collection
# ─────────────────────────────────────────────────────────────────────────────

def _num_frames(audio_data: Any) -> int:
    """Estimate number of frames from 1D/2D audio containers."""
    if audio_data is None:
        return 0
    if isinstance(audio_data, torch.Tensor):
        if audio_data.ndim == 1:
            return int(audio_data.shape[0])
        if audio_data.ndim == 2:
            if audio_data.shape[0] > 8 and audio_data.shape[1] <= 8:
                return int(audio_data.shape[0])
            return int(audio_data.shape[1])
        return 0
    try:
        if hasattr(audio_data, "shape"):
            shape = tuple(audio_data.shape)
            if len(shape) == 1:
                return int(shape[0])
            if len(shape) == 2:
                if shape[0] > 8 and shape[1] <= 8:
                    return int(shape[0])
                return int(shape[1])
        return int(len(audio_data))
    except Exception:
        return 0


def _duration_from_audio_item(audio_item: Any, default_sr: int = 16_000) -> float:
    """
    Compute duration from dataset audio cells supporting dicts and AudioDecoder objects.
    """
    if audio_item is None:
        return 0.0

    if isinstance(audio_item, Mapping):
        raw_bytes = audio_item.get("bytes")
        if raw_bytes:
            try:
                return float(get_duration_s(raw_bytes))
            except Exception:
                return 0.0

        audio_array = audio_item.get("array")
        if audio_array is not None:
            sr = int(audio_item.get("sampling_rate") or default_sr)
            frames = _num_frames(audio_array)
            return (frames / sr) if sr > 0 else 0.0

        audio_path = audio_item.get("path")
        if audio_path:
            try:
                info = torchaudio.info(audio_path)
                return float(info.num_frames / info.sample_rate) if info.sample_rate else 0.0
            except Exception:
                return 0.0

        return 0.0

    get_all_samples = getattr(audio_item, "get_all_samples", None)
    if callable(get_all_samples):
        try:
            decoded = get_all_samples()
            data = getattr(decoded, "data", None)
            if data is None:
                data = getattr(decoded, "array", None)
            if data is None and isinstance(decoded, Mapping):
                data = decoded.get("data") or decoded.get("array")
            frames = _num_frames(data)
            sr = (
                getattr(decoded, "sample_rate", None)
                or getattr(audio_item, "sampling_rate", None)
                or default_sr
            )
            return (frames / int(sr)) if int(sr) > 0 else 0.0
        except Exception:
            return 0.0

    audio_array = getattr(audio_item, "array", None)
    if audio_array is not None:
        sr = int(getattr(audio_item, "sampling_rate", default_sr))
        frames = _num_frames(audio_array)
        return (frames / sr) if sr > 0 else 0.0

    audio_path = getattr(audio_item, "path", None)
    if audio_path:
        try:
            info = torchaudio.info(audio_path)
            return float(info.num_frames / info.sample_rate) if info.sample_rate else 0.0
        except Exception:
            return 0.0

    return 0.0


def collect_stats(ds: Dataset, sample_rate: int = 16_000) -> pd.DataFrame:
    """
    Compute per-record stats and return as a DataFrame.
    Columns: duration_s, text_lug_len, text_eng_len, cps_lug, cps_eng
    """
    rows = []
    for item in ds:
        dur = _duration_from_audio_item(item.get("audio_lug"), default_sr=sample_rate)

        t_lug = str(item.get("text_lug") or "")
        t_eng = str(item.get("text_eng") or "")

        rows.append({
            "duration_s":   dur,
            "text_lug_len": len(t_lug),
            "text_eng_len": len(t_eng),
            "cps_lug":      len(t_lug) / dur if dur > 0 else 0.0,
            "cps_eng":      len(t_eng) / dur if dur > 0 else 0.0,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────────────────────

def generate_plots(stats: pd.DataFrame, report_dir: Path, fmt: str = "png") -> None:
    """Save diagnostic plots to `report_dir`."""
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Duration histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(stats["duration_s"].clip(upper=30), bins=60, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("Audio Duration Distribution (audio_lug)")
    plt.tight_layout()
    fig.savefig(report_dir / f"duration_hist.{fmt}", dpi=150)
    plt.close(fig)

    # 2. Text length distribution
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, label in zip(axes, ["text_lug_len", "text_eng_len"], ["Luganda", "English"]):
        ax.hist(stats[col].clip(upper=300), bins=50, edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Characters")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} Transcript Length")
    plt.tight_layout()
    fig.savefig(report_dir / f"text_length_dist.{fmt}", dpi=150)
    plt.close(fig)

    # 3. CPS scatter
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(stats["duration_s"].clip(upper=30), stats["cps_lug"].clip(upper=30),
               alpha=0.15, s=4)
    ax.axhline(25, color="red", linestyle="--", linewidth=0.8, label="max CPS threshold (25)")
    ax.axhline(3,  color="orange", linestyle="--", linewidth=0.8, label="min CPS threshold (3)")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Chars / second")
    ax.set_title("CPS vs Duration (Luganda)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(report_dir / f"cps_scatter.{fmt}", dpi=150)
    plt.close(fig)

    log.info(f"  Plots saved to {report_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  Spot-check CSV
# ─────────────────────────────────────────────────────────────────────────────

def export_spot_check(ds: Dataset, report_dir: Path, n: int = 200) -> None:
    """Export N random text pairs for human MT evaluation."""
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    rows = [{"id": ds[i]["id"], "text_lug": ds[i]["text_lug"], "text_eng": ds[i]["text_eng"]}
            for i in indices]
    df = pd.DataFrame(rows)
    out = report_dir / "spot_check_mt.csv"
    df.to_csv(out, index=False)
    log.info(f"  Spot-check CSV ({n} pairs) → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset card
# ─────────────────────────────────────────────────────────────────────────────

def write_dataset_card(
    ds: Dataset,
    stats: pd.DataFrame,
    report_dir: Path,
    hub_repo: str,
) -> None:
    """Write a Markdown dataset card."""
    total_h = stats["duration_s"].sum() / 3600
    mean_dur = stats["duration_s"].mean()

    card = f"""---
language:
- lug
- en
license: cc-by-4.0
task_categories:
- automatic-speech-recognition
- text-to-speech
- translation
pretty_name: Luganda–English Paired Speech Dataset
size_categories:
- 100K<n<1M
---

# Luganda–English Paired Speech Dataset

A paired bilingual speech dataset for **Luganda** (`lug`) and **English** (`en`),
designed for speech translation, ASR, and TTS research.

## Dataset Summary

| Property | Value |
|---|---|
| Total examples | {len(ds):,} |
| Total audio duration (approx.) | {total_h:.1f} hours |
| Mean clip duration | {mean_dur:.2f} s |
| Audio sample rate | 16 000 Hz mono |
| Languages | Luganda (`lug`), English (`en`) |

## Schema

| Column | Type | Description |
|---|---|---|
| `id` | string | Unique record ID |
| `audio_lug` | Audio | Original Luganda speech (16 kHz) |
| `text_lug` | string | Original Luganda transcript |
| `text_eng` | string | English translation (NLLB-200) |
| `audio_eng` | Audio | Synthesised English speech (SpeechT5) |

## Source Datasets

- [Sunbird SALT](https://huggingface.co/datasets/Sunbird/salt)
- [Mozilla Common Voice 17](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)
- [Google FLEURS](https://huggingface.co/datasets/google/fleurs)

## Pipeline

1. **Ingestion** — Unified from multiple HF sources
2. **Preprocessing** — Silero VAD trim, 16 kHz resample, peak normalisation
3. **Filtering** — SNR ≥ 15 dB, CPS 3–25, deduplication
4. **Translation** — `facebook/nllb-200-distilled-600M` (lug_Latn → eng_Latn)
5. **TTS** — `microsoft/speecht5_tts` + HiFiGAN vocoder
6. **Assembly** — Schema enforcement, Hub push

## Citation

If you use this dataset, please cite the upstream sources and this pipeline repository.

## Licence

Pipeline code: MIT. Dataset contents inherit licences from upstream sources.
Please check individual source licences before redistribution.
"""

    out = report_dir / "dataset_card.md"
    out.write_text(card, encoding="utf-8")
    log.info(f"  Dataset card → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Stage entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_qa(cfg: dict) -> None:
    """
    Run Stage 7: compute QA statistics, generate plots, export spot-check,
    and write dataset card.

    Parameters
    ----------
    cfg : dict
        Parsed pipeline configuration.
    """
    from datasets import load_from_disk

    in_path    = Path(cfg["paths"]["final"])   / "final_dataset"
    report_dir = Path(cfg["paths"]["reports"])
    report_dir.mkdir(parents=True, exist_ok=True)

    log.info("[bold]Stage 7 — QA Report[/bold]")
    ds: Dataset = load_from_disk(str(in_path))
    log.info(f"  Dataset: {len(ds):,} records")

    log.info("  Computing statistics…")
    stats = collect_stats(ds, sample_rate=cfg["audio"]["sample_rate"])

    # ── Console summary ───────────────────────────────────────────────
    total_hours = stats["duration_s"].sum() / 3600
    log.info(f"\n  {'─'*40}")
    log.info(f"  Total examples:     {len(ds):>10,}")
    log.info(f"  Total audio (hrs):  {total_hours:>10.1f}")
    log.info(f"  Mean duration (s):  {stats['duration_s'].mean():>10.2f}")
    log.info(f"  Median duration(s): {stats['duration_s'].median():>10.2f}")
    log.info(f"  Mean Lug text len:  {stats['text_lug_len'].mean():>10.1f} chars")
    log.info(f"  Mean Eng text len:  {stats['text_eng_len'].mean():>10.1f} chars")
    log.info(f"  {'─'*40}\n")

    # ── Plots ─────────────────────────────────────────────────────────
    generate_plots(stats, report_dir, fmt=cfg["qa"].get("plot_format", "png"))

    # ── Spot-check export ─────────────────────────────────────────────
    export_spot_check(ds, report_dir, n=cfg["qa"].get("spot_check_n", 200))

    # ── Dataset card ──────────────────────────────────────────────────
    write_dataset_card(
        ds,
        stats,
        report_dir,
        hub_repo=cfg["hub"].get("repo_id", "your-org/luganda-english-speech"),
    )

    # ── Save stats CSV ────────────────────────────────────────────────
    stats_out = report_dir / "stats.csv"
    stats.to_csv(stats_out, index=False)
    log.info(f"  Full stats CSV → {stats_out}")
    log.info("[green]  ✓ QA stage complete[/green]")
