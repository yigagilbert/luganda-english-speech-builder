"""
Stage 3 — Text Cleaning & Quality Filtering
============================================
Applies the following passes in order:

1. Unicode NFC normalisation + whitespace collapse
2. Minimum / maximum transcript length gate
3. SNR estimation → reject if below threshold
4. Characters-per-second ratio gate (anti-misalignment)
5. Language plausibility check (reject records with > 30% Latin digits/ASCII
   if the transcript is meant to be Luganda)
6. Exact-duplicate transcript removal (keeps first occurrence)
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torchaudio
import torchaudio.functional as F
from datasets import Dataset

from luganda_pipeline.utils.audio_utils import (
    chars_per_second,
    estimate_snr,
    load_audio_bytes,
    tensor_to_bytes,
)
from luganda_pipeline.utils.logging import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Text normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalise_text(text: str) -> str:
    """
    Apply lightweight normalisation to a Luganda transcript:
    - Unicode NFC
    - Strip leading/trailing whitespace
    - Collapse internal runs of whitespace to a single space
    - Remove zero-width / control characters
    """
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) not in {"Cc", "Cf"})
    text = " ".join(text.split())
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
#  Individual filter predicates
# ─────────────────────────────────────────────────────────────────────────────

def _passes_length(text: str, min_len: int, max_len: int) -> bool:
    n = len(text.strip())
    return min_len <= n <= max_len


def _prepare_waveform(audio_array: Any, sr: int, target_sr: int) -> tuple[torch.Tensor, int]:
    """Convert decoded audio arrays/tensors to mono float32 waveform at target_sr."""
    waveform = torch.as_tensor(audio_array, dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 2:
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


def _decode_audio_item(audio_item: Any, target_sr: int) -> tuple[torch.Tensor, int, bytes]:
    """
    Decode dataset audio cell to waveform + sr + serialized bytes.
    Supports dict audio payloads and datasets AudioDecoder-like objects.
    """
    if audio_item is None:
        raise ValueError("audio item is None")

    if isinstance(audio_item, Mapping):
        raw_bytes = audio_item.get("bytes")
        if raw_bytes:
            waveform, sr = load_audio_bytes(raw_bytes, target_sr=target_sr)
            return waveform, sr, raw_bytes

        audio_array = audio_item.get("array")
        if audio_array is not None:
            sr = int(audio_item.get("sampling_rate") or target_sr)
            waveform, sr = _prepare_waveform(audio_array, sr=sr, target_sr=target_sr)
            return waveform, sr, tensor_to_bytes(waveform, sample_rate=sr)

        audio_path = audio_item.get("path")
        if audio_path:
            waveform, sr = torchaudio.load(audio_path)
            waveform, sr = _prepare_waveform(waveform, sr=sr, target_sr=target_sr)
            return waveform, sr, tensor_to_bytes(waveform, sample_rate=sr)

        raise ValueError("dict audio item missing bytes/array/path")

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
        waveform, sr = _prepare_waveform(audio_array, sr=int(sr), target_sr=target_sr)
        return waveform, sr, tensor_to_bytes(waveform, sample_rate=sr)

    audio_array = getattr(audio_item, "array", None)
    if audio_array is not None:
        sr = int(getattr(audio_item, "sampling_rate", target_sr))
        waveform, sr = _prepare_waveform(audio_array, sr=sr, target_sr=target_sr)
        return waveform, sr, tensor_to_bytes(waveform, sample_rate=sr)

    decode_fn = getattr(audio_item, "decode", None)
    if callable(decode_fn):
        decoded = decode_fn()
        if decoded is audio_item:
            raise ValueError(f"Unsupported audio item type: {type(audio_item)}")
        return _decode_audio_item(decoded, target_sr=target_sr)

    raise ValueError(f"Unsupported audio item type: {type(audio_item)}")


def _passes_snr(waveform: torch.Tensor, sr: int, min_snr: float) -> bool:
    try:
        snr = estimate_snr(waveform, sr)
        return snr >= min_snr
    except Exception:
        return False


def _passes_cps(text: str, duration_s: float, min_cps: float, max_cps: float) -> bool:
    try:
        cps = chars_per_second(text, duration_s)
        if cps is None:
            return False
        return min_cps <= cps <= max_cps
    except Exception:
        return False


def _passes_lang_check(text: str, max_ascii_ratio: float = 0.30) -> bool:
    """
    Reject if more than `max_ascii_ratio` of characters are pure ASCII
    digits or Latin letters — a heuristic for detecting non-Luganda text.
    Luganda uses the Latin alphabet but has diacritics; very high ASCII
    digit counts suggest a wrong-language transcript.
    """
    if not text:
        return False
    ascii_count = sum(1 for ch in text if ch.isascii() and ch.isdigit())
    return (ascii_count / len(text)) <= max_ascii_ratio


# ─────────────────────────────────────────────────────────────────────────────
#  Batch map function
# ─────────────────────────────────────────────────────────────────────────────

def _filter_batch(batch: dict, filter_cfg: dict, audio_cfg: dict) -> dict:
    """Filter and normalise one batch. Returns a (possibly smaller) batch."""
    kept_audio: list = []
    kept_text: list = []

    sr = audio_cfg["sample_rate"]
    min_snr = filter_cfg["min_snr_db"]
    max_cps = filter_cfg["max_cps"]
    min_cps = filter_cfg["min_cps"]
    min_len = filter_cfg["min_transcript_len"]
    max_len = filter_cfg["max_transcript_len"]
    do_lang  = filter_cfg.get("lang_check", True)

    for audio_item, text in zip(batch["audio_lug"], batch["text_lug"]):
        # Normalise text first
        text = normalise_text(str(text or ""))

        if not _passes_length(text, min_len, max_len):
            continue

        try:
            waveform, sr_item, raw_bytes = _decode_audio_item(audio_item, target_sr=sr)
        except Exception:
            continue

        if not _passes_snr(waveform, sr_item, min_snr):
            continue

        duration_s = waveform.shape[1] / sr_item
        if not _passes_cps(text, duration_s, min_cps, max_cps):
            continue

        if do_lang and not _passes_lang_check(text):
            continue

        kept_audio.append({"bytes": raw_bytes, "path": None})
        kept_text.append(text)

    batch["audio_lug"] = kept_audio
    batch["text_lug"] = kept_text
    return batch


# ─────────────────────────────────────────────────────────────────────────────
#  Deduplication
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate(ds: Dataset) -> Dataset:
    """
    Remove records with duplicate ``text_lug`` values (keep first occurrence).
    Uses MD5 hashing for O(n) performance.
    """
    seen: set[str] = set()
    keep_indices: list[int] = []

    for i, text in enumerate(ds["text_lug"]):
        h = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()
        if h not in seen:
            seen.add(h)
            keep_indices.append(i)

    removed = len(ds) - len(keep_indices)
    if removed:
        log.info(f"  Deduplication removed {removed:,} records ({removed/len(ds)*100:.1f}%)")

    return ds.select(keep_indices)


# ─────────────────────────────────────────────────────────────────────────────
#  Stage entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_filtering(cfg: dict) -> Dataset:
    """
    Run Stage 3: text cleaning, quality filtering, and deduplication.

    Parameters
    ----------
    cfg : dict
        Parsed pipeline configuration.

    Returns
    -------
    Dataset  saved to cfg['paths']['filtered']
    """
    from datasets import load_from_disk

    in_path = Path(cfg["paths"]["preprocessed"]) / "preprocessed_dataset"
    out_path = Path(cfg["paths"]["filtered"]) / "filtered_dataset"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("[bold]Stage 3 — Text Cleaning & Quality Filtering[/bold]")
    ds: Dataset = load_from_disk(str(in_path))
    n_before = len(ds)
    log.info(f"  Input:  {n_before:,} records")

    ds = ds.map(
        _filter_batch,
        batched=True,
        batch_size=64,
        fn_kwargs={
            "filter_cfg": cfg["filter"],
            "audio_cfg":  cfg["audio"],
        },
        desc="Filtering records",
    )

    if cfg["filter"].get("deduplicate_text", True):
        ds = deduplicate(ds)

    n_after = len(ds)
    pct_kept = n_after / n_before * 100 if n_before else 0
    log.info(f"  Output: {n_after:,} records ({pct_kept:.1f}% kept)")

    log.info(f"  Saving → {out_path}")
    ds.save_to_disk(str(out_path))

    return ds
