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

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torchaudio
import torchaudio.functional as F
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


def _prepare_waveform(audio_array: Any, sr: int, target_sr: int) -> tuple[torch.Tensor, int]:
    """
    Convert decoded audio arrays/tensors to mono float32 waveform at target_sr.
    """
    waveform = torch.as_tensor(audio_array, dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 2:
        # Handle either (channels, time) or (time, channels).
        if waveform.shape[0] > 8 and waveform.shape[1] <= 8:
            waveform = waveform.transpose(0, 1)
    else:
        raise ValueError(f"Unsupported audio tensor shape: {tuple(waveform.shape)}")

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        waveform = F.resample(waveform, sr, target_sr)
        sr = target_sr

    return waveform, sr


def _decode_audio_item(audio_item: Any, target_sr: int) -> tuple[torch.Tensor, int]:
    """
    Decode dataset audio cell to waveform + sr.
    Supports:
      - dict items: {"bytes"/"array"/"path", ...}
      - datasets AudioDecoder-like objects with get_all_samples()
    """
    if audio_item is None:
        raise ValueError("audio item is None")

    # Case 1: legacy dict payloads.
    if isinstance(audio_item, Mapping):
        raw_bytes = audio_item.get("bytes")
        if raw_bytes:
            return load_audio_bytes(raw_bytes, target_sr=target_sr)

        audio_array = audio_item.get("array")
        if audio_array is not None:
            sr = int(audio_item.get("sampling_rate") or target_sr)
            return _prepare_waveform(audio_array, sr=sr, target_sr=target_sr)

        audio_path = audio_item.get("path")
        if audio_path:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != target_sr:
                waveform = F.resample(waveform, sr, target_sr)
                sr = target_sr
            return waveform, sr

        raise ValueError("dict audio item missing bytes/array/path")

    # Case 2: AudioDecoder-like objects from datasets.
    get_all_samples = getattr(audio_item, "get_all_samples", None)
    if callable(get_all_samples):
        decoded = get_all_samples()

        audio_array = getattr(decoded, "data", None)
        if audio_array is None:
            audio_array = getattr(decoded, "array", None)
        if audio_array is None and isinstance(decoded, Mapping):
            audio_array = decoded.get("data") or decoded.get("array")
        if audio_array is None and isinstance(decoded, torch.Tensor):
            audio_array = decoded

        sr = (
            getattr(decoded, "sample_rate", None)
            or getattr(audio_item, "sampling_rate", None)
            or target_sr
        )
        if audio_array is None:
            raise ValueError("AudioDecoder returned samples without data/array")
        return _prepare_waveform(audio_array, sr=int(sr), target_sr=target_sr)

    # Case 3: object already exposing array/sampling_rate.
    audio_array = getattr(audio_item, "array", None)
    if audio_array is not None:
        sr = int(getattr(audio_item, "sampling_rate", target_sr))
        return _prepare_waveform(audio_array, sr=sr, target_sr=target_sr)

    # Case 4: decode() returns dict/decoder payload in some datasets versions.
    decode_fn = getattr(audio_item, "decode", None)
    if callable(decode_fn):
        decoded = decode_fn()
        if decoded is audio_item:
            raise ValueError(f"Unsupported audio item type: {type(audio_item)}")
        return _decode_audio_item(decoded, target_sr=target_sr)

    raise ValueError(f"Unsupported audio item type: {type(audio_item)}")


def _get_vad():
    global _vad_model, _vad_utils
    if _vad_model is None:
        log.info("Loading Silero VAD model…")
        _vad_model, _vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
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
        try:
            waveform, sr = _decode_audio_item(audio_item, target_sr=target_sr)
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
