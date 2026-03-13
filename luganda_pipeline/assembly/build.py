"""
Stage 6 — Dataset Assembly & Schema Validation
===============================================
Produces the final dataset with exactly five columns:
    id | audio_eng | audio_lug | text_eng | text_lug

Steps:
  1. Assign unique IDs using source-aware prefixes
  2. Enforce exact column set and order
  3. Cast both audio columns to datasets.Audio(sampling_rate=16000)
  4. Run schema assertions (nulls, duplicates, types)
  5. Optionally push to Hugging Face Hub
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from datasets import Audio, Dataset

from luganda_pipeline.utils.logging import get_logger

log = get_logger(__name__)

FINAL_COLUMNS = ["id", "audio_eng", "audio_lug", "text_eng", "text_lug"]


# ─────────────────────────────────────────────────────────────────────────────
#  ID assignment
# ─────────────────────────────────────────────────────────────────────────────

def assign_ids(ds: Dataset, prefix: str = "lug_eng") -> Dataset:
    """
    Add a globally unique ``id`` column.
    Format: ``{prefix}_{index:07d}``  e.g.  ``lug_eng_0001234``
    """
    def _add_id(batch: dict, indices: list[int]) -> dict:
        batch["id"] = [f"{prefix}_{i:07d}" for i in indices]
        return batch

    return ds.map(_add_id, batched=True, with_indices=True, desc="Assigning IDs")


# ─────────────────────────────────────────────────────────────────────────────
#  Schema validation
# ─────────────────────────────────────────────────────────────────────────────

def _audio_item_has_payload(item: Any) -> bool:
    """
    Return True if an audio cell appears to contain data.
    Supports both dict payloads and datasets AudioDecoder-like objects.
    """
    if item is None:
        return False

    if isinstance(item, Mapping):
        if item.get("bytes"):
            return True
        if item.get("path"):
            return True
        arr = item.get("array")
        if arr is not None:
            try:
                return len(arr) > 0
            except Exception:
                return True
        return False

    get_all_samples = getattr(item, "get_all_samples", None)
    if callable(get_all_samples):
        try:
            decoded = get_all_samples()
            data = getattr(decoded, "data", None)
            if data is None:
                data = getattr(decoded, "array", None)
            if data is None and isinstance(decoded, Mapping):
                data = decoded.get("data") or decoded.get("array")
            if data is None:
                return True
            try:
                return len(data) > 0
            except Exception:
                return True
        except Exception:
            return False

    path = getattr(item, "path", None)
    if path:
        return True
    arr = getattr(item, "array", None)
    if arr is not None:
        try:
            return len(arr) > 0
        except Exception:
            return True

    return True


def validate_schema(ds: Dataset, sample_rate: int = 16_000) -> None:
    """
    Assert the final dataset satisfies all schema requirements.
    Raises AssertionError on any violation.
    """
    # 1. Column set
    assert set(ds.column_names) == set(FINAL_COLUMNS), (
        f"Column mismatch.\n  Expected: {sorted(FINAL_COLUMNS)}\n  Got:      {sorted(ds.column_names)}"
    )

    # 2. No null text values
    for col in ("text_lug", "text_eng"):
        nulls = sum(1 for v in ds[col] if not v or not str(v).strip())
        assert nulls == 0, f"Found {nulls} null/empty values in column '{col}'"

    # 3. ID uniqueness
    ids = ds["id"]
    assert len(ids) == len(set(ids)), (
        f"Duplicate IDs detected! {len(ids) - len(set(ids))} duplicates."
    )

    # 4. No null audio
    for col in ("audio_lug", "audio_eng"):
        nulls = sum(1 for item in ds[col] if not _audio_item_has_payload(item))
        assert nulls == 0, f"Found {nulls} null audio items in column '{col}'"

    log.info("[green]  ✓ Schema validation passed[/green]")


# ─────────────────────────────────────────────────────────────────────────────
#  Stage entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_assembly(cfg: dict, hf_token: str | None = None) -> Dataset:
    """
    Run Stage 6: assemble final dataset, validate schema, optionally push to Hub.

    Parameters
    ----------
    cfg : dict
        Parsed pipeline configuration.
    hf_token : str | None
        HuggingFace token for Hub push.

    Returns
    -------
    Dataset  saved to cfg['paths']['final']
    """
    from datasets import load_from_disk

    in_path  = Path(cfg["paths"]["synthesized"]) / "synthesized_dataset"
    out_path = Path(cfg["paths"]["final"])        / "final_dataset"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("[bold]Stage 6 — Dataset Assembly & Validation[/bold]")
    ds: Dataset = load_from_disk(str(in_path))
    log.info(f"  Input: {len(ds):,} records")

    # ── Assign IDs ────────────────────────────────────────────────────
    ds = assign_ids(ds, prefix="lug_eng")

    # ── Enforce column set ────────────────────────────────────────────
    existing = set(ds.column_names)
    for col in FINAL_COLUMNS:
        assert col in existing, f"Required column '{col}' is missing from dataset"

    extra_cols = existing - set(FINAL_COLUMNS)
    if extra_cols:
        ds = ds.remove_columns(list(extra_cols))

    ds = ds.select_columns(FINAL_COLUMNS)  # enforce order

    # ── Cast audio columns ────────────────────────────────────────────
    sr = cfg["audio"]["sample_rate"]
    ds = ds.cast_column("audio_lug", Audio(sampling_rate=sr))
    ds = ds.cast_column("audio_eng", Audio(sampling_rate=sr))

    # ── Validate ──────────────────────────────────────────────────────
    log.info("  Running schema validation…")
    validate_schema(ds, sample_rate=sr)

    # ── Save locally ──────────────────────────────────────────────────
    log.info(f"  Saving final dataset → {out_path}")
    ds.save_to_disk(str(out_path))

    # ── Push to Hub ───────────────────────────────────────────────────
    hub_cfg = cfg.get("hub", {})
    if hub_cfg.get("push_to_hub", False):
        repo_id = hub_cfg["repo_id"]
        log.info(f"  Pushing to Hugging Face Hub: {repo_id}")
        ds.push_to_hub(
            repo_id=repo_id,
            token=hf_token,
            private=hub_cfg.get("private", True),
            max_shard_size=hub_cfg.get("max_shard_size", "2GB"),
        )
        log.info(f"  [green]✓ Dataset pushed to: https://huggingface.co/datasets/{repo_id}[/green]")
    else:
        log.info("  Hub push disabled (set hub.push_to_hub: true in config to enable)")

    log.info(f"  Final dataset: [bold]{len(ds):,}[/bold] examples")
    return ds
