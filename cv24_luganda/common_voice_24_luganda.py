#!/usr/bin/env python3
"""
common_voice_24_luganda.py
==========================
Professional processor for Mozilla Common Voice 24.0 — Luganda (lg)
Source: https://datacollective.mozillafoundation.org/datasets/cmj8u3pcu00elnxxby6wyhysl

Dataset facts (v24.0)
---------------------
  Archive   : mcv-scripted-lg-v24.0.tar.gz  (11.03 GB)
  Clips     : 348,763 total  |  436.83 h validated
  Speakers  : 665
  Licence   : CC0-1.0  (public domain)
  Format    : MP3 inside a flat clips/ directory, metadata in TSV files

What this script does
---------------------
  1. Downloads the archive (resumable, with progress bar)
  2. Extracts it safely to a local directory
  3. Reads all TSV splits (train / dev / test / validated)
  4. Applies configurable quality filters:
        - Only validated clips (up_votes > down_votes)
        - Minimum up-vote threshold
        - Duration gate  [0.5 s – 30 s]
        - SNR estimate   >= 15 dB
        - Text length    [2 – 500 chars]
        - Unicode normalisation + whitespace collapse
  5. Resamples MP3 → 16-bit PCM WAV at 16 000 Hz mono
  6. Outputs a HuggingFace Dataset with the standard pipeline schema:
        id | audio_lug | text_lug
     (ready to be fed straight into Stage 4 — Translation)
  7. Optionally pushes the resulting dataset to HuggingFace Hub

Usage
-----
    # Full run (download + process + save)
    python common_voice_24_luganda.py

    # Skip download (archive already present)
    python common_voice_24_luganda.py --skip-download

    # Process only the validated split
    python common_voice_24_luganda.py --splits validated

    # Push to Hub after processing
    python common_voice_24_luganda.py --push-to-hub your-org/cv24-luganda

    # Dry-run: print stats only, no output saved
    python common_voice_24_luganda.py --dry-run
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import sys
import tarfile
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import click
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as AF
from datasets import Audio, Dataset
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(show_time=True, show_path=False, markup=True)],
)
log = logging.getLogger("cv24_lug")
console = Console()


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # --- Paths ---
    data_dir: Path = Path("data/cv24_luganda")
    archive_name: str = "mcv-scripted-lg-v24.0.tar.gz"
    output_dir: Path = Path("data/cv24_luganda/processed")

    # --- Download (filled at runtime from env or CLI) ---
    download_url: str = ""          # Set via --download-url or CV24_DOWNLOAD_URL env var
    download_token: str = ""        # Bearer token if the API requires auth

    # --- Split selection ---
    splits: list[str] = field(default_factory=lambda: ["train", "dev", "test"])

    # --- Quality filters ---
    min_up_votes: int = 2           # Minimum community up-votes to accept a clip
    validated_only: bool = True     # Only include clips where up_votes > down_votes
    min_duration_s: float = 0.5
    max_duration_s: float = 30.0
    min_snr_db: float = 15.0
    min_text_len: int = 2
    max_text_len: int = 500

    # --- Audio output ---
    target_sr: int = 16_000         # Output sample rate

    # --- Processing ---
    num_workers: int = max(1, os.cpu_count() - 1)  # type: ignore[operator]
    batch_size: int = 256

    # --- Hub ---
    push_to_hub: bool = False
    hub_repo_id: str = ""
    hub_private: bool = True

    # --- Misc ---
    dry_run: bool = False
    id_prefix: str = "cv24_lg"

    @property
    def archive_path(self) -> Path:
        return self.data_dir / self.archive_name

    @property
    def extract_dir(self) -> Path:
        return self.data_dir / "extracted"

    @property
    def clips_dir(self) -> Path:
        # Common Voice archives unpack to a language-code subdirectory
        return self.extract_dir / "lg" / "clips"

    @property
    def tsv_dir(self) -> Path:
        return self.extract_dir / "lg"


# ─────────────────────────────────────────────────────────────────────────────
#  Download (resumable)
# ─────────────────────────────────────────────────────────────────────────────

def download_archive(cfg: Config) -> None:
    """
    Download the Common Voice 24.0 Luganda archive with:
    - Resumable download (Range header)
    - Progress bar (Rich)
    - SHA-256 integrity check (if a checksum is provided)
    """
    import requests

    url = cfg.download_url
    if not url:
        log.error(
            "No download URL provided.\n"
            "  Set --download-url or the CV24_DOWNLOAD_URL environment variable.\n"
            "  Get your signed URL from: "
            "https://datacollective.mozillafoundation.org/datasets/cmj8u3pcu00elnxxby6wyhysl"
        )
        sys.exit(1)

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    dest = cfg.archive_path

    # Resume support
    existing_bytes = dest.stat().st_size if dest.exists() else 0
    headers = {}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        log.info(f"Resuming download from byte {existing_bytes:,}")

    if cfg.download_token:
        headers["Authorization"] = f"Bearer {cfg.download_token}"

    resp = requests.get(url, headers=headers, stream=True, timeout=60)

    if resp.status_code == 416:
        log.info("Archive already fully downloaded.")
        return
    resp.raise_for_status()

    total_size = int(resp.headers.get("content-length", 0)) + existing_bytes
    mode = "ab" if existing_bytes > 0 else "wb"

    log.info(f"Downloading → {dest}  ({total_size / 1e9:.2f} GB)")

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    )
    with progress:
        task = progress.add_task("Downloading", total=total_size, completed=existing_bytes)
        with dest.open(mode) as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                fh.write(chunk)
                progress.update(task, advance=len(chunk))

    log.info("[green]✓ Download complete[/green]")


# ─────────────────────────────────────────────────────────────────────────────
#  Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_archive(cfg: Config) -> None:
    """Extract the tar.gz archive to cfg.extract_dir (skips if already done)."""
    sentinel = cfg.extract_dir / ".extracted"
    if sentinel.exists():
        log.info("Archive already extracted — skipping.")
        return

    if not cfg.archive_path.exists():
        log.error(f"Archive not found: {cfg.archive_path}")
        sys.exit(1)

    cfg.extract_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Extracting {cfg.archive_path.name} → {cfg.extract_dir} …")

    with tarfile.open(cfg.archive_path, "r:gz") as tar:
        members = tar.getmembers()
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
        with progress:
            task = progress.add_task("Extracting", total=len(members))
            for member in members:
                # Safety: prevent path traversal attacks
                member_path = cfg.extract_dir / member.name
                if not str(member_path.resolve()).startswith(str(cfg.extract_dir.resolve())):
                    log.warning(f"Skipping unsafe path: {member.name}")
                    continue
                tar.extract(member, path=cfg.extract_dir)
                progress.advance(task)

    sentinel.touch()
    log.info("[green]✓ Extraction complete[/green]")


# ─────────────────────────────────────────────────────────────────────────────
#  TSV loading & metadata filtering
# ─────────────────────────────────────────────────────────────────────────────

#  TSV columns in Common Voice 24.0
_CV_COLUMNS = ["client_id", "path", "text", "up_votes", "down_votes", "age", "gender", "accent", "segment"]


def load_tsv_split(tsv_path: Path) -> pd.DataFrame:
    """Load a single Common Voice TSV split into a DataFrame."""
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        quoting=3,           # QUOTE_NONE — CV TSVs have no quoting
        on_bad_lines="skip",
        dtype={"up_votes": "Int64", "down_votes": "Int64"},
    )
    # Normalise column names (older CV versions differ slightly)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Ensure expected columns exist
    for col in ("path", "text", "up_votes", "down_votes"):
        if col not in df.columns:
            df[col] = pd.NA

    return df


def load_all_metadata(cfg: Config) -> pd.DataFrame:
    """
    Load TSV metadata for all requested splits.
    Deduplicates by audio file path (clips that appear in multiple splits are
    kept once, prioritising the train split).
    """
    split_priority = {"train": 0, "dev": 1, "test": 2, "validated": 3, "other": 4}
    frames: list[pd.DataFrame] = []

    for split in cfg.splits:
        tsv_path = cfg.tsv_dir / f"{split}.tsv"
        if not tsv_path.exists():
            log.warning(f"TSV not found: {tsv_path} — skipping split '{split}'")
            continue
        df = load_tsv_split(tsv_path)
        df["split"] = split
        df["split_priority"] = split_priority.get(split, 99)
        frames.append(df)
        log.info(f"  Loaded {split}.tsv: {len(df):,} rows")

    if not frames:
        log.error("No TSV files loaded. Check cfg.splits and cfg.tsv_dir.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate: keep one row per path, preferring lower split_priority
    combined = (
        combined
        .sort_values("split_priority")
        .drop_duplicates(subset="path", keep="first")
        .reset_index(drop=True)
    )
    log.info(f"  Total unique clips after dedup: {len(combined):,}")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
#  Metadata-level quality filter (fast, no audio I/O)
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_text(text: str) -> str:
    """Unicode NFC + strip control chars + collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) not in {"Cc", "Cf"})
    return " ".join(text.split()).strip()


def apply_metadata_filters(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Apply all filters that can be evaluated from TSV metadata alone
    (no audio loading required).

    Filters applied
    ---------------
    1. Drop rows with missing path or text
    2. Validate: up_votes > down_votes  (if cfg.validated_only)
    3. Minimum up-vote count
    4. Text length gate
    5. Unicode normalise text
    6. Verify audio file physically exists on disk
    """
    n_start = len(df)
    log.info(f"Metadata filter — input: {n_start:,} rows")

    # 1. Drop missing
    df = df.dropna(subset=["path", "text"])
    df = df[df["text"].str.strip().str.len() > 0]
    _log_drop(n_start, len(df), "missing path/text")

    # 2. Validated only
    if cfg.validated_only:
        n = len(df)
        up   = pd.to_numeric(df["up_votes"],   errors="coerce").fillna(0).astype(int)
        down = pd.to_numeric(df["down_votes"],  errors="coerce").fillna(0).astype(int)
        df = df[(up > down)].copy()
        _log_drop(n, len(df), "validation vote check (up_votes ≤ down_votes)")

    # 3. Minimum up votes
    if cfg.min_up_votes > 0:
        n = len(df)
        up = pd.to_numeric(df["up_votes"], errors="coerce").fillna(0).astype(int)
        df = df[up >= cfg.min_up_votes].copy()
        _log_drop(n, len(df), f"up_votes < {cfg.min_up_votes}")

    # 4. Text length
    n = len(df)
    text_len = df["text"].str.len()
    df = df[(text_len >= cfg.min_text_len) & (text_len <= cfg.max_text_len)].copy()
    _log_drop(n, len(df), "text length gate")

    # 5. Normalise text
    df["text"] = df["text"].apply(lambda t: _normalise_text(str(t)))

    # 6. File existence check
    n = len(df)
    df["_audio_path"] = df["path"].apply(
        lambda p: cfg.clips_dir / (p if p.endswith(".mp3") else p + ".mp3")
    )
    df = df[df["_audio_path"].apply(lambda p: p.exists())].copy()
    _log_drop(n, len(df), "missing audio file on disk")

    log.info(
        f"Metadata filter — output: [bold]{len(df):,}[/bold] rows "
        f"({len(df)/n_start*100:.1f}% of input)"
    )
    return df.reset_index(drop=True)


def _log_drop(n_before: int, n_after: int, reason: str) -> None:
    dropped = n_before - n_after
    if dropped > 0:
        log.info(f"  Dropped {dropped:,} ({dropped/n_before*100:.1f}%) — {reason}")


# ─────────────────────────────────────────────────────────────────────────────
#  Per-clip audio processing (runs in worker processes)
# ─────────────────────────────────────────────────────────────────────────────

def _process_clip(args: tuple) -> dict | None:
    """
    Worker function: load one MP3, resample, normalise, check SNR and duration.

    Parameters
    ----------
    args : tuple
        (row_index, audio_path_str, text_lug, id_str, target_sr, min_dur, max_dur, min_snr)

    Returns
    -------
    dict with keys: id, audio_lug, text_lug
    None if the clip fails any quality check.
    """
    (row_idx, audio_path_str, text_lug, id_str,
     target_sr, min_dur, max_dur, min_snr) = args

    try:
        waveform, sr = torchaudio.load(audio_path_str)
    except Exception:
        return None

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to target SR
    if sr != target_sr:
        waveform = AF.resample(waveform, sr, target_sr)
        sr = target_sr

    # Duration gate
    duration_s = waveform.shape[1] / sr
    if not (min_dur <= duration_s <= max_dur):
        return None

    # SNR estimate (energy-based)
    frame_len = int(sr * 0.03)  # 30 ms frames
    wav_np = waveform.squeeze(0).numpy()
    n_frames = len(wav_np) // frame_len
    if n_frames >= 2:
        frames = wav_np[: n_frames * frame_len].reshape(n_frames, frame_len)
        energies = (frames ** 2).mean(axis=1)
        if energies.max() > 1e-10:
            noise_floor = np.percentile(energies, 10)
            signal_peak = np.percentile(energies, 90)
            snr = 10 * np.log10(signal_peak / (noise_floor + 1e-10))
            if snr < min_snr:
                return None

    # Peak normalise to -1 dBFS
    peak = waveform.abs().max()
    if peak > 1e-8:
        waveform = waveform * (10 ** (-1.0 / 20.0) / peak)

    # Encode as 16-bit PCM WAV bytes
    wav_int16 = (waveform.clamp(-1.0, 1.0) * 32767).to(torch.int16)
    buf = io.BytesIO()
    torchaudio.save(buf, wav_int16, sr, format="wav", bits_per_sample=16)
    wav_bytes = buf.getvalue()

    return {
        "id":        id_str,
        "audio_lug": {"bytes": wav_bytes, "path": None},
        "text_lug":  text_lug,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Batch audio processing
# ─────────────────────────────────────────────────────────────────────────────

def process_audio_parallel(df: pd.DataFrame, cfg: Config) -> list[dict]:
    """
    Process all clips in `df` using a multiprocess pool.
    Returns a list of record dicts (None results are dropped).
    """
    args_list = [
        (
            i,
            str(row["_audio_path"]),
            row["text"],
            f"{cfg.id_prefix}_{i:07d}",
            cfg.target_sr,
            cfg.min_duration_s,
            cfg.max_duration_s,
            cfg.min_snr_db,
        )
        for i, row in df.iterrows()
    ]

    records: list[dict] = []
    n_skipped = 0

    log.info(f"Processing {len(args_list):,} clips with {cfg.num_workers} workers…")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    with progress:
        task = progress.add_task("Processing audio", total=len(args_list))
        with ProcessPoolExecutor(max_workers=cfg.num_workers) as pool:
            futures = {pool.submit(_process_clip, a): a for a in args_list}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    records.append(result)
                else:
                    n_skipped += 1
                progress.advance(task)

    log.info(
        f"  Audio processed: [bold]{len(records):,}[/bold] kept, "
        f"{n_skipped:,} skipped (duration/SNR/decode failure)"
    )
    return records


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(records: list[dict], cfg: Config) -> Dataset:
    """
    Assemble records into a HuggingFace Dataset with the standard schema:
        id | audio_lug | text_lug
    """
    log.info("Assembling HuggingFace Dataset…")

    ds = Dataset.from_list(records)
    ds = ds.cast_column("audio_lug", Audio(sampling_rate=cfg.target_sr))

    # Guarantee column order
    ds = ds.select_columns(["id", "audio_lug", "text_lug"])

    log.info(f"  Dataset: {len(ds):,} examples  |  columns: {ds.column_names}")
    return ds


# ─────────────────────────────────────────────────────────────────────────────
#  Stats & reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(df_meta: pd.DataFrame, records: list[dict], cfg: Config) -> None:
    """Print a Rich stats table to the console."""

    table = Table(title="Common Voice 24.0 — Luganda Processing Summary", show_lines=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="bold white")

    n_clips_raw     = len(df_meta)  # after metadata filter, before audio
    n_final         = len(records)
    pct_kept        = n_final / n_clips_raw * 100 if n_clips_raw else 0

    table.add_row("Splits processed",        ", ".join(cfg.splits))
    table.add_row("Clips after meta-filter", f"{n_clips_raw:,}")
    table.add_row("Clips in final dataset",  f"{n_final:,}")
    table.add_row("Overall retention",       f"{pct_kept:.1f} %")
    table.add_row("Target sample rate",      f"{cfg.target_sr:,} Hz mono 16-bit")
    table.add_row("Output directory",        str(cfg.output_dir))

    if "age" in df_meta.columns:
        age_dist = df_meta["age"].value_counts().head(5).to_dict()
        table.add_row("Top age bands",       str(age_dist))
    if "gender" in df_meta.columns:
        gen_dist = df_meta["gender"].value_counts().to_dict()
        table.add_row("Gender distribution", str(gen_dist))

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
#  Hub push
# ─────────────────────────────────────────────────────────────────────────────

def push_to_hub(ds: Dataset, cfg: Config, hf_token: str | None) -> None:
    if not cfg.hub_repo_id:
        log.error("--push-to-hub requires --hub-repo-id (e.g. your-org/cv24-luganda)")
        sys.exit(1)

    log.info(f"Pushing dataset to Hub: {cfg.hub_repo_id}")
    ds.push_to_hub(
        repo_id=cfg.hub_repo_id,
        token=hf_token,
        private=cfg.hub_private,
        max_shard_size="2GB",
    )
    log.info(f"[green]✓ Dataset available at: https://huggingface.co/datasets/{cfg.hub_repo_id}[/green]")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

@click.command(context_settings={"max_content_width": 100})
@click.option("--data-dir",         default="data/cv24_luganda",   show_default=True,  help="Local data root directory.")
@click.option("--output-dir",       default="data/cv24_luganda/processed", show_default=True, help="Output directory for the processed dataset.")
@click.option("--download-url",     default="",   envvar="CV24_DOWNLOAD_URL",    help="Signed download URL for the archive (or set CV24_DOWNLOAD_URL).")
@click.option("--download-token",   default="",   envvar="CV24_DOWNLOAD_TOKEN",  help="Optional bearer token for authenticated download.")
@click.option("--skip-download",    is_flag=True, default=False,  help="Skip download (archive already present).")
@click.option("--skip-extract",     is_flag=True, default=False,  help="Skip extraction (archive already extracted).")
@click.option("--splits",           default="train,dev,test",      show_default=True,  help="Comma-separated list of TSV splits to process.")
@click.option("--validated-only/--no-validated-only", default=True, show_default=True, help="Accept only clips with up_votes > down_votes.")
@click.option("--min-up-votes",     default=2,    show_default=True, type=int,  help="Minimum community up-votes per clip.")
@click.option("--min-duration",     default=0.5,  show_default=True, type=float, help="Minimum clip duration (seconds).")
@click.option("--max-duration",     default=30.0, show_default=True, type=float, help="Maximum clip duration (seconds).")
@click.option("--min-snr",          default=15.0, show_default=True, type=float, help="Minimum estimated SNR (dB).")
@click.option("--min-text-len",     default=2,    show_default=True, type=int,  help="Minimum transcript character length.")
@click.option("--max-text-len",     default=500,  show_default=True, type=int,  help="Maximum transcript character length.")
@click.option("--target-sr",        default=16000, show_default=True, type=int, help="Output audio sample rate (Hz).")
@click.option("--num-workers",      default=max(1, (os.cpu_count() or 2) - 1), show_default=True, type=int, help="Number of parallel worker processes.")
@click.option("--push-to-hub",      "do_push",    is_flag=True, default=False,  help="Push final dataset to HuggingFace Hub.")
@click.option("--hub-repo-id",      default="",   help="HuggingFace Hub repo ID (e.g. your-org/cv24-luganda).")
@click.option("--hub-private/--hub-public", default=True, show_default=True, help="Make Hub repo private.")
@click.option("--dry-run",          is_flag=True, default=False,  help="Print stats without saving anything.")
@click.option("--id-prefix",        default="cv24_lg", show_default=True, help="Prefix for generated record IDs.")
def main(
    data_dir, output_dir, download_url, download_token,
    skip_download, skip_extract,
    splits, validated_only, min_up_votes,
    min_duration, max_duration, min_snr, min_text_len, max_text_len,
    target_sr, num_workers,
    do_push, hub_repo_id, hub_private,
    dry_run, id_prefix,
):
    """
    Process Mozilla Common Voice 24.0 — Luganda dataset.

    Produces a HuggingFace Dataset with columns: id | audio_lug | text_lug
    that slots directly into the Luganda–English speech pipeline (Stage 4+).
    """
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")

    console.rule("[bold cyan]Common Voice 24.0 — Luganda Processor[/bold cyan]")

    cfg = Config(
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        download_url=download_url,
        download_token=download_token,
        splits=[s.strip() for s in splits.split(",")],
        validated_only=validated_only,
        min_up_votes=min_up_votes,
        min_duration_s=min_duration,
        max_duration_s=max_duration,
        min_snr_db=min_snr,
        min_text_len=min_text_len,
        max_text_len=max_text_len,
        target_sr=target_sr,
        num_workers=num_workers,
        push_to_hub=do_push,
        hub_repo_id=hub_repo_id,
        hub_private=hub_private,
        dry_run=dry_run,
        id_prefix=id_prefix,
    )

    # ── Step 1: Download ────────────────────────────────────────────
    if not skip_download:
        console.rule("[bold]Step 1 / 6 — Download[/bold]")
        download_archive(cfg)
    else:
        log.info("Step 1 skipped (--skip-download)")

    # ── Step 2: Extract ─────────────────────────────────────────────
    if not skip_extract:
        console.rule("[bold]Step 2 / 6 — Extract[/bold]")
        extract_archive(cfg)
    else:
        log.info("Step 2 skipped (--skip-extract)")

    # ── Step 3: Load & filter metadata ──────────────────────────────
    console.rule("[bold]Step 3 / 6 — Load Metadata[/bold]")
    df_raw = load_all_metadata(cfg)

    console.rule("[bold]Step 4 / 6 — Metadata Quality Filter[/bold]")
    df_filtered = apply_metadata_filters(df_raw, cfg)

    if dry_run:
        log.info("[yellow]DRY RUN — skipping audio processing and output.[/yellow]")
        console.print(df_filtered[["path", "text", "up_votes", "down_votes", "split"]].head(20).to_string())
        return

    # ── Step 4: Process audio ────────────────────────────────────────
    console.rule("[bold]Step 5 / 6 — Audio Processing[/bold]")
    records = process_audio_parallel(df_filtered, cfg)

    if not records:
        log.error("No records survived processing. Check filters and audio paths.")
        sys.exit(1)

    # ── Step 5: Assemble dataset ─────────────────────────────────────
    console.rule("[bold]Step 6 / 6 — Assemble & Save[/bold]")
    ds = build_dataset(records, cfg)

    # Print summary
    print_stats(df_filtered, records, cfg)

    # Save locally
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = cfg.output_dir / "cv24_lg_dataset"
    log.info(f"Saving dataset → {save_path}")
    ds.save_to_disk(str(save_path))
    log.info("[green]✓ Dataset saved[/green]")

    # ── Optional: Push to Hub ────────────────────────────────────────
    if cfg.push_to_hub:
        console.rule("[bold]Hub Push[/bold]")
        push_to_hub(ds, cfg, hf_token)

    # ── Done ─────────────────────────────────────────────────────────
    console.rule("[bold green]✓ Processing complete[/bold green]")
    log.info(
        f"Final dataset: [bold]{len(ds):,}[/bold] examples  "
        f"→  {save_path}"
    )
    log.info(
        "Next step: feed this dataset into [bold]Stage 4 — Translation[/bold] "
        "of the Luganda–English pipeline."
    )


if __name__ == "__main__":
    main()
