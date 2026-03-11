"""Tests for Stage 3 — Text Cleaning & Filtering."""

from __future__ import annotations

import pytest


def test_normalise_text_strips_control_chars():
    from luganda_pipeline.filtering.text import normalise_text

    raw = "Mwesigwa\u200b okulaba\u00a0  ennyo"
    result = normalise_text(raw)
    assert "\u200b" not in result
    assert "\u00a0" not in result
    assert result == "Mwesigwa okulaba ennyo"


def test_normalise_text_nfc():
    from luganda_pipeline.filtering.text import normalise_text
    import unicodedata

    # Compose NFC — café with combining accent
    raw = "cafe\u0301"
    result = normalise_text(raw)
    assert unicodedata.is_normalized("NFC", result)


def test_passes_length_gates():
    from luganda_pipeline.filtering.text import _passes_length

    assert _passes_length("hello world", 2, 500)
    assert not _passes_length("", 2, 500)
    assert not _passes_length("x" * 600, 2, 500)


def test_passes_lang_check_rejects_numeric():
    from luganda_pipeline.filtering.text import _passes_lang_check

    # Mostly digits — should be rejected
    assert not _passes_lang_check("1234567890" * 5, max_ascii_ratio=0.30)
    # Normal Luganda text
    assert _passes_lang_check("Webale nnyo okuyamba", max_ascii_ratio=0.30)


def test_deduplicate_removes_exact_dupes():
    from datasets import Dataset
    from luganda_pipeline.filtering.text import deduplicate

    data = {
        "text_lug": ["Webale", "Webale", "Nkwagala", "Nkwagala", "Gyebale"],
        "audio_lug": [None] * 5,
    }
    ds = Dataset.from_dict(data)
    deduped = deduplicate(ds)
    assert len(deduped) == 3
    assert list(deduped["text_lug"]) == ["Webale", "Nkwagala", "Gyebale"]
