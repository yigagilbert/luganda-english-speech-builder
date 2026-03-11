"""Structured logger using loguru + Rich console output."""

import sys
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

console = Console()

_CONFIGURED = False


def setup_logger(log_dir: str = "logs", log_file: str = "pipeline.log", level: str = "INFO") -> None:
    """Configure loguru to write to both console (Rich) and a rotating log file."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Remove default loguru handler
    logger.remove()

    # Console handler — Rich formatted
    logger.add(
        RichHandler(console=console, show_time=True, show_path=False, markup=True),
        level=level,
        format="{message}",
        colorize=False,
    )

    # File handler — full detail with rotation
    logger.add(
        Path(log_dir) / log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
        rotation="100 MB",
        retention="14 days",
        compression="zip",
        enqueue=True,
    )

    _CONFIGURED = True


def get_logger(name: str):
    """Return a loguru logger bound with a module name."""
    return logger.bind(name=name)
