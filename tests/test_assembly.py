"""Tests for Stage 6 — Dataset Assembly."""

from __future__ import annotations

import pytest
from datasets import Dataset


FINAL_COLUMNS = ["id", "audio_eng", "audio_lug", "text_eng", "text_lug"]


def _make_mock_ds(n: int = 5) -> Dataset:
    return Dataset.from_dict({
        "audio_lug": [{"bytes": b"", "path": None}] * n,
        "audio_eng": [{"bytes": b"", "path": None}] * n,
        "text_lug":  [f"Luganda text {i}" for i in range(n)],
        "text_eng":  [f"English text {i}" for i in range(n)],
    })


def test_assign_ids_format():
    from luganda_pipeline.assembly.build import assign_ids

    ds = _make_mock_ds(3)
    ds = assign_ids(ds, prefix="test")
    assert ds["id"] == ["test_0000000", "test_0000001", "test_0000002"]


def test_assign_ids_unique():
    from luganda_pipeline.assembly.build import assign_ids

    ds = _make_mock_ds(100)
    ds = assign_ids(ds)
    assert len(set(ds["id"])) == 100


def test_validate_schema_passes_valid():
    from luganda_pipeline.assembly.build import assign_ids, validate_schema

    ds = _make_mock_ds(5)
    ds = assign_ids(ds)
    # Should not raise
    validate_schema(ds)


def test_validate_schema_fails_on_missing_column():
    from luganda_pipeline.assembly.build import assign_ids, validate_schema
    from datasets import Dataset

    ds = Dataset.from_dict({
        "id":       ["a", "b"],
        "audio_lug": [{"bytes": b"x", "path": None}] * 2,
        "text_lug": ["text"] * 2,
        "text_eng": ["text"] * 2,
        # Missing audio_eng
    })
    with pytest.raises(AssertionError, match="Column mismatch"):
        validate_schema(ds)


def test_validate_schema_catches_duplicate_ids():
    from luganda_pipeline.assembly.build import validate_schema

    ds = Dataset.from_dict({
        "id":       ["dupe", "dupe"],
        "audio_lug": [{"bytes": b"x", "path": None}] * 2,
        "audio_eng": [{"bytes": b"x", "path": None}] * 2,
        "text_lug": ["a", "b"],
        "text_eng": ["c", "d"],
    })
    with pytest.raises(AssertionError, match="Duplicate IDs"):
        validate_schema(ds)


def test_audio_item_has_payload_accepts_audio_decoder_like_items():
    from luganda_pipeline.assembly.build import _audio_item_has_payload

    class _FakeSamples:
        def __init__(self, data):
            self.data = data

    class _FakeAudioDecoder:
        def __init__(self, data):
            self._data = data

        def get_all_samples(self):
            return _FakeSamples(self._data)

    assert _audio_item_has_payload(_FakeAudioDecoder([0.1, -0.1]))
    assert not _audio_item_has_payload(None)
