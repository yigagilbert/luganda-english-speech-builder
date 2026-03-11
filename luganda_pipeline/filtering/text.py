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
from pathlib import Path

from datasets import Dataset

from luganda_pipeline.utils.audio_utils import (
    chars_per_second,
    estimate_snr,
    get_duration_s,
    load_audio_bytes,
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


def _passes_snr(audio_bytes: bytes, sr: int, min_snr: float) -> bool:
    try:
        waveform, _ = load_audio_bytes(audio_bytes, target_sr=sr)
        snr = estimate_snr(waveform, sr)
        return snr >= min_snr
    except Exception:
        return False


def _passes_cps(text: str, audio_bytes: bytes, min_cps: float, max_cps: float) -> bool:
    try:
        dur = get_duration_s(audio_bytes)
        cps = chars_per_second(text, dur)
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

        raw_bytes = (audio_item or {}).get("bytes") or b""
        if not raw_bytes:
            continue

        if not _passes_snr(raw_bytes, sr, min_snr):
            continue

        if not _passes_cps(text, raw_bytes, min_cps, max_cps):
            continue

        if do_lang and not _passes_lang_check(text):
            continue

        kept_audio.append(audio_item)
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
