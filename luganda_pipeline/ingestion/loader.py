"""
Stage 1 — Data Ingestion
========================
Pulls each configured Hugging Face dataset, renames columns to the
canonical schema names (audio_lug, text_lug), casts audio to 16 kHz,
and concatenates everything into a single Arrow-backed dataset saved
under `paths.raw`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Audio, Dataset, concatenate_datasets, load_dataset
from tqdm.auto import tqdm

from luganda_pipeline.utils.logging import get_logger

log = get_logger(__name__)

# Maps each source's native column names → canonical names
_COLUMN_ALIASES: dict[str, list[str]] = {
    "text_lug": ["sentence", "transcription", "text", "transcript"],
    "audio_lug": ["audio"],
}


def _resolve_column(ds: Dataset, canonical: str) -> str | None:
    """Return the first alias that exists in `ds.column_names`, else None."""
    cols = set(ds.column_names)
    for alias in _COLUMN_ALIASES.get(canonical, [canonical]):
        if alias in cols:
            return alias
    return None


def load_source(
    repo_id: str,
    config: str | None,
    splits: list[str],
    audio_col: str,
    text_col: str,
    target_sr: int = 16_000,
    hf_token: str | None = None,
) -> Dataset:
    """
    Load one source dataset and return a unified Dataset with columns
    ``audio_lug`` and ``text_lug`` only.

    Parameters
    ----------
    repo_id : str
        HuggingFace dataset repository ID.
    config : str | None
        Dataset configuration / language code (e.g. "lg", "lug_ug").
    splits : list[str]
        List of split names to load (e.g. ["train", "validation"]).
    audio_col : str
        Name of the audio column in the upstream dataset.
    text_col : str
        Name of the transcript column in the upstream dataset.
    target_sr : int
        Target sample rate for audio column cast.
    hf_token : str | None
        HuggingFace authentication token.

    Returns
    -------
    Dataset with columns: audio_lug, text_lug
    """
    shards: list[Dataset] = []
    for split in splits:
        log.info(f"Loading {repo_id} / config={config!r} / split={split!r}")
        kwargs: dict[str, Any] = {
            "split": split,
            "trust_remote_code": True,
        }
        if hf_token:
            kwargs["token"] = hf_token

        try:
            ds: Dataset = load_dataset(
                repo_id,
                config,
                **kwargs,
            )
        except Exception as exc:
            log.warning(f"Could not load {repo_id}[{split}]: {exc}. Skipping.")
            continue

        # ── Resolve column names ──────────────────────────────────────
        resolved_audio = audio_col if audio_col in ds.column_names else _resolve_column(ds, "audio_lug")
        resolved_text = text_col if text_col in ds.column_names else _resolve_column(ds, "text_lug")

        if resolved_audio is None or resolved_text is None:
            log.warning(
                f"  Could not map audio/text columns for {repo_id}[{split}]. "
                f"Available: {ds.column_names}. Skipping."
            )
            continue

        # ── Keep only needed columns ──────────────────────────────────
        ds = ds.select_columns([resolved_audio, resolved_text])

        # ── Rename to canonical names ─────────────────────────────────
        rename_map: dict[str, str] = {}
        if resolved_audio != "audio_lug":
            rename_map[resolved_audio] = "audio_lug"
        if resolved_text != "text_lug":
            rename_map[resolved_text] = "text_lug"
        if rename_map:
            ds = ds.rename_columns(rename_map)

        # ── Cast audio to uniform sample rate ─────────────────────────
        ds = ds.cast_column("audio_lug", Audio(sampling_rate=target_sr))

        log.info(f"  → {len(ds):,} examples loaded from {repo_id}[{split}]")
        shards.append(ds)

    if not shards:
        raise RuntimeError(f"No data loaded from {repo_id}. Check config and credentials.")

    return concatenate_datasets(shards)


def run_ingestion(cfg: dict, hf_token: str | None = None) -> Dataset:
    """
    Run Stage 1: load all enabled sources and return the concatenated raw Dataset.

    Parameters
    ----------
    cfg : dict
        Parsed pipeline configuration (from config.yaml).
    hf_token : str | None
        HuggingFace token for private datasets.

    Returns
    -------
    Dataset  saved to cfg['paths']['raw']
    """
    raw_dir = Path(cfg["paths"]["raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    save_path = raw_dir / "raw_dataset"

    sources = [s for s in cfg["sources"] if s.get("enabled", True)]
    log.info(f"[bold]Stage 1 — Ingestion[/bold] ({len(sources)} sources)")

    all_shards: list[Dataset] = []
    for src in tqdm(sources, desc="Sources", unit="src"):
        ds = load_source(
            repo_id=src["repo_id"],
            config=src.get("config"),
            splits=src["splits"],
            audio_col=src.get("audio_col", "audio"),
            text_col=src.get("text_col", "text"),
            target_sr=cfg["audio"]["sample_rate"],
            hf_token=hf_token,
        )
        all_shards.append(ds)

    raw: Dataset = concatenate_datasets(all_shards)

    log.info(f"Total raw examples: {len(raw):,}")
    log.info(f"Saving raw dataset → {save_path}")
    raw.save_to_disk(str(save_path))

    return raw
