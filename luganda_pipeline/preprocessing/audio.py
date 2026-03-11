"""
Stage 2 — Audio Preprocessing
==============================
For every record:
  - Resample to target sample rate (16 kHz)
  - Convert to mono
  - Peak-normalise to -1 dBFS
  - Trim leading/trailing silence via Silero VAD
  - Reject clips outside [min_duration_s, max_duration_s]
  - Re-encode as 16-bit PCM WAV bytes
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from datasets import Audio, Dataset

from luganda_pipeline.utils.audio_utils import (
    load_audio_bytes,
    peak_normalise,
    tensor_to_bytes,
)
from luganda_pipeline.utils.logging import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  VAD helper (lazy-loaded so the import doesn't fail without silero)
# ─────────────────────────────────────────────────────────────────────────────

_vad_model: Any = None
_vad_utils: Any = None


def _get_vad():
    global _vad_model, _vad_utils
    if _vad_model is None:
        log.info("Loading Silero VAD model…")
        _vad_model, _vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
    return _vad_model, _vad_utils


def vad_trim(
    waveform: torch.Tensor,
    sr: int,
    threshold: float = 0.5,
    min_silence_ms: int = 200,
) -> torch.Tensor:
    """
    Trim leading and trailing silence from `waveform` using Silero VAD.

    Parameters
    ----------
    waveform : torch.Tensor, shape (1, T)
    sr : int
        Sample rate (must be 8000 or 16000 for Silero).
    threshold : float
        Speech probability threshold (0–1).
    min_silence_ms : int
        Minimum consecutive silence to trim.

    Returns
    -------
    torch.Tensor  trimmed waveform (1, T')
    """
    try:
        model, utils = _get_vad()
        get_speech_ts = utils[0]

        wav_mono = waveform.squeeze(0)
        speech_timestamps = get_speech_ts(
            wav_mono,
            model,
            threshold=threshold,
            min_silence_duration_ms=min_silence_ms,
            sampling_rate=sr,
        )
        if not speech_timestamps:
            return waveform  # All silence — return unchanged; will be filtered later

        start = speech_timestamps[0]["start"]
        end = speech_timestamps[-1]["end"]
        return waveform[:, start:end]

    except Exception as exc:
        log.debug(f"VAD trim failed ({exc}), returning original waveform")
        return waveform


# ─────────────────────────────────────────────────────────────────────────────
#  Batch processing function
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_batch(
    batch: dict,
    target_sr: int,
    min_dur: float,
    max_dur: float,
    peak_dbfs: float,
    vad_threshold: float,
    vad_silence_ms: int,
) -> dict:
    """Applied via Dataset.map — processes one batch of records."""
    new_audio: list[dict | None] = []
    keep_mask: list[bool] = []

    for audio_item in batch["audio_lug"]:
        raw_bytes: bytes = audio_item.get("bytes") or b""
        if not raw_bytes:
            keep_mask.append(False)
            new_audio.append(None)
            continue

        try:
            waveform, sr = load_audio_bytes(raw_bytes, target_sr=target_sr)
        except Exception as exc:
            log.debug(f"Audio decode failed: {exc}")
            keep_mask.append(False)
            new_audio.append(None)
            continue

        # VAD trim
        waveform = vad_trim(waveform, sr, threshold=vad_threshold, min_silence_ms=vad_silence_ms)

        # Duration gate
        duration_s = waveform.shape[1] / sr
        if not (min_dur <= duration_s <= max_dur):
            keep_mask.append(False)
            new_audio.append(None)
            continue

        # Peak normalise
        waveform = peak_normalise(waveform, target_dbfs=peak_dbfs)

        # Encode back to WAV bytes
        wav_bytes = tensor_to_bytes(waveform, sample_rate=sr)
        new_audio.append({"bytes": wav_bytes, "path": None})
        keep_mask.append(True)

    # Apply mask
    batch["audio_lug"] = [a for a, k in zip(new_audio, keep_mask) if k]
    batch["text_lug"] = [t for t, k in zip(batch["text_lug"], keep_mask) if k]

    return batch


def run_preprocessing(cfg: dict) -> Dataset:
    """
    Run Stage 2: preprocess all audio records.

    Parameters
    ----------
    cfg : dict
        Parsed pipeline configuration.

    Returns
    -------
    Dataset  saved to cfg['paths']['preprocessed']
    """
    from datasets import load_from_disk

    raw_path = Path(cfg["paths"]["raw"]) / "raw_dataset"
    out_path = Path(cfg["paths"]["preprocessed"]) / "preprocessed_dataset"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("[bold]Stage 2 — Audio Preprocessing[/bold]")
    ds: Dataset = load_from_disk(str(raw_path))
    log.info(f"  Input:  {len(ds):,} records")

    audio_cfg = cfg["audio"]

    ds = ds.map(
        _preprocess_batch,
        batched=True,
        batch_size=64,
        fn_kwargs={
            "target_sr":      audio_cfg["sample_rate"],
            "min_dur":        audio_cfg["min_duration_s"],
            "max_dur":        audio_cfg["max_duration_s"],
            "peak_dbfs":      audio_cfg["peak_norm_dbfs"],
            "vad_threshold":  audio_cfg["vad_threshold"],
            "vad_silence_ms": audio_cfg["vad_min_silence_ms"],
        },
        desc="Preprocessing audio",
        remove_columns=[],
    )

    # Re-cast audio column after byte manipulation
    ds = ds.cast_column("audio_lug", Audio(sampling_rate=audio_cfg["sample_rate"]))

    log.info(f"  Output: {len(ds):,} records (after duration filtering)")
    log.info(f"  Saving → {out_path}")
    ds.save_to_disk(str(out_path))

    return ds
