"""
luganda_pipeline.pipeline
=========================
Main orchestrator.  Loads config, initialises the checkpoint manager,
and runs each stage in sequence (or a single named stage via --stage).

Usage
-----
    # Full pipeline
    python -m luganda_pipeline.pipeline

    # Single stage
    python -m luganda_pipeline.pipeline --stage translation

    # Force re-run a completed stage
    python -m luganda_pipeline.pipeline --stage filtering --force

    # Reset all checkpoints and start fresh
    python -m luganda_pipeline.pipeline --reset-all
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import click
import yaml
from dotenv import load_dotenv
from rich.console import Console

from luganda_pipeline.utils.checkpoint import STAGES, CheckpointManager
from luganda_pipeline.utils.logging import get_logger, setup_logger

console = Console()
log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load and return the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        console.print(f"[red]Config file not found: {path}[/red]")
        sys.exit(1)
    with path.open() as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
#  Stage runners registry
# ─────────────────────────────────────────────────────────────────────────────

def _run_ingestion(cfg: dict, hf_token: str | None) -> None:
    from luganda_pipeline.ingestion.loader import run_ingestion
    run_ingestion(cfg, hf_token=hf_token)


def _run_preprocessing(cfg: dict, **_) -> None:
    from luganda_pipeline.preprocessing.audio import run_preprocessing
    run_preprocessing(cfg)


def _run_filtering(cfg: dict, **_) -> None:
    from luganda_pipeline.filtering.text import run_filtering
    run_filtering(cfg)


def _run_translation(cfg: dict, **_) -> None:
    from luganda_pipeline.translation.translate import run_translation
    run_translation(cfg)


def _run_tts(cfg: dict, **_) -> None:
    from luganda_pipeline.tts.synthesize import run_tts
    run_tts(cfg)


def _run_assembly(cfg: dict, hf_token: str | None) -> None:
    from luganda_pipeline.assembly.build import run_assembly
    run_assembly(cfg, hf_token=hf_token)


def _run_qa(cfg: dict, **_) -> None:
    from luganda_pipeline.qa.report import run_qa
    run_qa(cfg)


STAGE_RUNNERS = {
    "ingestion":     _run_ingestion,
    "preprocessing": _run_preprocessing,
    "filtering":     _run_filtering,
    "translation":   _run_translation,
    "tts":           _run_tts,
    "assembly":      _run_assembly,
    "qa":            _run_qa,
}


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--config", default="config/config.yaml", show_default=True,
    help="Path to pipeline config YAML.",
)
@click.option(
    "--stage", default=None, type=click.Choice(STAGES, case_sensitive=False),
    help="Run only this stage (skips checkpoint check).",
)
@click.option(
    "--force", is_flag=True, default=False,
    help="Force re-run even if the stage is already checkpointed.",
)
@click.option(
    "--reset-all", is_flag=True, default=False,
    help="Clear all checkpoints and restart the pipeline from scratch.",
)
@click.option(
    "--status", is_flag=True, default=False,
    help="Print checkpoint status and exit.",
)
def main(
    config: str,
    stage: Optional[str],
    force: bool,
    reset_all: bool,
    status: bool,
) -> None:
    """Luganda–English Paired Speech Dataset Pipeline."""
    # Load env + config
    load_dotenv()
    cfg = load_config(config)

    # Setup logging
    log_cfg = cfg.get("logging", {})
    setup_logger(
        log_dir=log_cfg.get("log_dir", "logs"),
        log_file=log_cfg.get("log_file", "pipeline.log"),
        level=log_cfg.get("level", "INFO"),
    )

    hf_token: str | None = os.environ.get("HF_TOKEN")

    # Checkpoint manager
    ckpt = CheckpointManager(cfg["paths"]["checkpoints"])

    if status:
        console.print(ckpt.summary())
        return

    if reset_all:
        ckpt.reset()
        log.info("All checkpoints cleared.")

    # ── Single stage mode ────────────────────────────────────────────
    if stage:
        if ckpt.is_done(stage) and not force:
            log.info(f"Stage '{stage}' already completed. Use --force to re-run.")
        else:
            log.info(f"Running single stage: [bold]{stage}[/bold]")
            STAGE_RUNNERS[stage](cfg=cfg, hf_token=hf_token)
            ckpt.mark_done(stage)
        return

    # ── Full pipeline mode ────────────────────────────────────────────
    console.rule("[bold]Luganda–English Speech Dataset Pipeline[/bold]")
    console.print(ckpt.summary())

    for s in STAGES:
        if ckpt.is_done(s) and not force:
            log.info(f"[dim]Skipping '{s}' (already done)[/dim]")
            continue

        console.rule(f"[bold cyan]{s.upper()}[/bold cyan]")
        try:
            STAGE_RUNNERS[s](cfg=cfg, hf_token=hf_token)
            ckpt.mark_done(s)
        except Exception as exc:
            log.error(f"[red]Stage '{s}' failed: {exc}[/red]")
            log.error("Pipeline halted. Fix the error and re-run to resume from this stage.")
            raise SystemExit(1) from exc

    console.rule("[green bold]✓ Pipeline complete[/green bold]")
    log.info("Final dataset is in: " + cfg["paths"]["final"])


if __name__ == "__main__":
    main()
