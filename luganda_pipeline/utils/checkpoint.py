"""
Lightweight checkpoint manager.

After each stage completes, the pipeline saves a marker file so that
subsequent runs can skip already-completed stages.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from luganda_pipeline.utils.logging import get_logger

log = get_logger(__name__)

STAGES = [
    "ingestion",
    "preprocessing",
    "filtering",
    "translation",
    "tts",
    "assembly",
    "qa",
]


class CheckpointManager:
    """
    Persists per-stage completion status to `{checkpoint_dir}/status.json`.

    Usage
    -----
    >>> ckpt = CheckpointManager("data/checkpoints")
    >>> if not ckpt.is_done("ingestion"):
    ...     run_ingestion()
    ...     ckpt.mark_done("ingestion")
    """

    def __init__(self, checkpoint_dir: str) -> None:
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._path = self.dir / "status.json"
        self._status: dict = self._load()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def is_done(self, stage: str) -> bool:
        """Return True if the stage has been marked complete."""
        return self._status.get(stage, {}).get("done", False)

    def mark_done(self, stage: str, meta: dict | None = None) -> None:
        """Mark a stage as complete and persist state."""
        self._status[stage] = {
            "done": True,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **(meta or {}),
        }
        self._save()
        log.info(f"[green]✓ Checkpoint saved:[/green] stage='{stage}'")

    def reset(self, stage: str | None = None) -> None:
        """Reset one stage or all stages (forces re-run)."""
        if stage:
            self._status.pop(stage, None)
            log.warning(f"Checkpoint reset for stage '{stage}'")
        else:
            self._status = {}
            log.warning("All checkpoints cleared — pipeline will run from scratch")
        self._save()

    def summary(self) -> str:
        """Return a human-readable status table."""
        lines = ["", "  Stage Checkpoints", "  " + "─" * 40]
        for s in STAGES:
            info = self._status.get(s, {})
            if info.get("done"):
                ts = info.get("timestamp", "")
                lines.append(f"  [green]✓[/green]  {s:<18} {ts}")
            else:
                lines.append(f"  [dim]○[/dim]  {s:<18} pending")
        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Private                                                             #
    # ------------------------------------------------------------------ #

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except json.JSONDecodeError:
                log.warning("Checkpoint file corrupted — starting fresh")
        return {}

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._status, indent=2))
