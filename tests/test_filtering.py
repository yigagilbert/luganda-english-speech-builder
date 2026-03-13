"""Tests for Stage 3 — Text Cleaning & Filtering."""

from __future__ import annotations

import pytest
import torch


def _make_wav_bytes(duration_s: float = 1.0, sr: int = 16_000) -> bytes:
    from luganda_pipeline.utils.audio_utils import tensor_to_bytes

    t = torch.linspace(0, duration_s, int(sr * duration_s))
    wave = (0.5 * torch.sin(2 * 3.14159 * 220 * t)).unsqueeze(0)
    return tensor_to_bytes(wave, sample_rate=sr)


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


def test_decode_audio_item_supports_dict_bytes():
    from luganda_pipeline.filtering.text import _decode_audio_item

    wav = _make_wav_bytes(sr=22_050)
    waveform, sr, raw_bytes = _decode_audio_item({"bytes": wav, "path": None}, target_sr=16_000)
    assert sr == 16_000
    assert waveform.shape[0] == 1
    assert len(raw_bytes) > 0


def test_decode_audio_item_supports_audio_decoder_like():
    from luganda_pipeline.filtering.text import _decode_audio_item

    class _FakeSamples:
        def __init__(self, data, sample_rate):
            self.data = data
            self.sample_rate = sample_rate

    class _FakeAudioDecoder:
        def __init__(self, data, sample_rate):
            self._data = data
            self._sample_rate = sample_rate

        def get_all_samples(self):
            return _FakeSamples(self._data, self._sample_rate)

    mono = torch.randn(16_000)
    decoder = _FakeAudioDecoder(mono, 16_000)
    waveform, sr, raw_bytes = _decode_audio_item(decoder, target_sr=16_000)
    assert sr == 16_000
    assert waveform.shape == (1, 16_000)
    assert len(raw_bytes) > 0
