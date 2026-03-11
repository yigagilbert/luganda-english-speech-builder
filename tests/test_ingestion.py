"""Tests for Stage 1 — Ingestion."""

from __future__ import annotations

import pytest


def test_resolve_column_finds_alias():
    """Loader should resolve aliased column names correctly."""
    from luganda_pipeline.ingestion.loader import _resolve_column
    from unittest.mock import MagicMock

    mock_ds = MagicMock()
    mock_ds.column_names = ["sentence", "audio", "speaker_id"]

    assert _resolve_column(mock_ds, "text_lug") == "sentence"
    assert _resolve_column(mock_ds, "audio_lug") == "audio"


def test_resolve_column_returns_none_on_missing():
    from luganda_pipeline.ingestion.loader import _resolve_column
    from unittest.mock import MagicMock

    mock_ds = MagicMock()
    mock_ds.column_names = ["unrelated_col"]

    assert _resolve_column(mock_ds, "text_lug") is None
