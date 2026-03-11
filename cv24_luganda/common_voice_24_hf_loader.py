#!/usr/bin/env python3
"""
common_voice_24_hf_loader.py
============================
Alternative loader that pulls Common Voice 24.0 Luganda directly from the
HuggingFace Hub (mozilla-foundation/common_voice_17_0 or the latest version)
instead of downloading the raw tar.gz manually.

Use this when:
  - You already have HF credentials with CV access approved
  - You prefer streaming / lazy loading over a full 11 GB download

The script produces the same output schema as common_voice_24_luganda.py:
    id | audio_lug | text_lug

Usage
-----
    python common_voice_24_hf_loader.py
    python common_voice_24_hf_loader.py --streaming          # stream without full download
    python common_voice_24_hf_loader.py --splits train,dev   # specific splits
    python common_voice_24_hf_loader.py --push-to-hub your-org/cv24-luganda
"""

from __future__ import annotations

import io
import os
import unicodedata
from pathlib import Path

import click
import torch
import torchaudio
import torchaudio.functional as AF
from datasets import Audio, Dataset, IterableDataset, load_dataset
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, MofNCompleteColumn, TextColumn, TimeElapsedColumn

console = Console()

# ── HuggingFace dataset ID for Common Voice Luganda ──────────────────────────
# v17 is the latest on HF Hub as of mid-2025; bump version string as new
# releases land on  https://huggingface.co/datasets/mozilla-foundation/
HF_REPO   = "mozilla-foundation/common_voice_17_0"
HF_CONFIG = "lg"          # Luganda language config


def _normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) not in {"Cc", "Cf"})
    return " ".join(text.split()).strip()


def _to_wav_bytes(waveform: torch.Tensor, sr: int) -> bytes:
    wav_i16 = (waveform.clamp(-1.0, 1.0) * 32767).to(torch.int16)
    buf = io.BytesIO()
    torchaudio.save(buf, wav_i16, sr, format="wav", bits_per_sample=16)
    return buf.getvalue()


def process_split(
    split: str,
    target_sr: int,
    min_up_votes: int,
    max_duration_s: float,
    min_duration_s: float,
    hf_token: str | None,
    streaming: bool,
) -> list[dict]:
    """Load and process one HF split into records."""
    console.print(f"  Loading split: [bold]{split}[/bold]")

    ds = load_dataset(
        HF_REPO,
        HF_CONFIG,
        split=split,
        trust_remote_code=True,
        token=hf_token,
        streaming=streaming,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=target_sr))

    records: list[dict] = []
    skipped = 0
    idx = 0

    iterable = iter(ds)
    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  MofNCompleteColumn(), TimeElapsedColumn()) as prog:
        task = prog.add_task(f"Processing {split}", total=None)

        for item in iterable:
            idx += 1
            prog.advance(task)

            # Vote filter
            up   = item.get("up_votes") or 0
            down = item.get("down_votes") or 0
            if up < min_up_votes or up <= down:
                skipped += 1
                continue

            # Text
            text = _normalise_text(str(item.get("sentence") or ""))
            if not text or len(text) < 2:
                skipped += 1
                continue

            # Audio
            audio_arr = item.get("audio", {})
            wav = torch.tensor(audio_arr.get("array", []), dtype=torch.float32).unsqueeze(0)
            sr  = audio_arr.get("sampling_rate", target_sr)

            if sr != target_sr:
                wav = AF.resample(wav, sr, target_sr)
                sr = target_sr

            duration_s = wav.shape[1] / sr
            if not (min_duration_s <= duration_s <= max_duration_s):
                skipped += 1
                continue

            # Peak normalise
            peak = wav.abs().max()
            if peak > 1e-8:
                wav = wav * (10 ** (-1.0 / 20.0) / peak)

            wav_bytes = _to_wav_bytes(wav, sr)
            rec_id = f"cv_hf_{split}_{idx:07d}"

            records.append({
                "id":        rec_id,
                "audio_lug": {"bytes": wav_bytes, "path": None},
                "text_lug":  text,
            })

    console.print(f"    Kept: [green]{len(records):,}[/green]  Skipped: {skipped:,}")
    return records


@click.command()
@click.option("--splits",         default="train,validation",  show_default=True)
@click.option("--target-sr",      default=16000, type=int,     show_default=True)
@click.option("--min-up-votes",   default=2,     type=int,     show_default=True)
@click.option("--min-duration",   default=0.5,   type=float,   show_default=True)
@click.option("--max-duration",   default=30.0,  type=float,   show_default=True)
@click.option("--output-dir",     default="data/cv24_luganda/processed", show_default=True)
@click.option("--streaming",      is_flag=True,  default=False)
@click.option("--push-to-hub",    "do_push",     is_flag=True, default=False)
@click.option("--hub-repo-id",    default="")
def main(splits, target_sr, min_up_votes, min_duration, max_duration,
         output_dir, streaming, do_push, hub_repo_id):
    """Pull CV24 Luganda from HF Hub and convert to pipeline-ready Dataset."""
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")

    console.rule("[bold]Common Voice — HuggingFace Loader[/bold]")

    all_records: list[dict] = []
    for split in [s.strip() for s in splits.split(",")]:
        recs = process_split(
            split=split,
            target_sr=target_sr,
            min_up_votes=min_up_votes,
            max_duration_s=max_duration,
            min_duration_s=min_duration,
            hf_token=hf_token,
            streaming=streaming,
        )
        all_records.extend(recs)

    if not all_records:
        console.print("[red]No records produced. Check filters.[/red]")
        return

    ds = Dataset.from_list(all_records)
    ds = ds.cast_column("audio_lug", Audio(sampling_rate=target_sr))
    ds = ds.select_columns(["id", "audio_lug", "text_lug"])

    out = Path(output_dir) / "cv_hf_dataset"
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    console.print(f"\n[green]✓[/green] {len(ds):,} examples saved → {out}")

    if do_push and hub_repo_id:
        ds.push_to_hub(hub_repo_id, token=hf_token, private=True)
        console.print(f"[green]✓ Pushed to Hub:[/green] {hub_repo_id}")


if __name__ == "__main__":
    main()
