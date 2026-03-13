"""Tests for Stage 2 — Audio Preprocessing utilities."""

from __future__ import annotations

import pytest
import torch


def _make_wav_bytes(duration_s: float = 2.0, sr: int = 16_000) -> bytes:
    """Generate a simple sine-wave WAV as bytes for testing."""
    from luganda_pipeline.utils.audio_utils import tensor_to_bytes

    t = torch.linspace(0, duration_s, int(sr * duration_s))
    wave = (0.5 * torch.sin(2 * 3.14159 * 440 * t)).unsqueeze(0)
    return tensor_to_bytes(wave, sample_rate=sr)


def test_load_audio_bytes_resamples():
    from luganda_pipeline.utils.audio_utils import load_audio_bytes

    wav = _make_wav_bytes(sr=22_050)
    waveform, sr = load_audio_bytes(wav, target_sr=16_000)
    assert sr == 16_000
    assert waveform.shape[0] == 1  # mono


def test_peak_normalise():
    from luganda_pipeline.utils.audio_utils import peak_normalise

    wav = torch.randn(1, 16_000) * 0.1
    norm = peak_normalise(wav, target_dbfs=-1.0)
    peak_db = 20 * torch.log10(norm.abs().max())
    assert abs(peak_db.item() - (-1.0)) < 0.1


def test_get_duration_s():
    from luganda_pipeline.utils.audio_utils import get_duration_s

    wav = _make_wav_bytes(duration_s=3.5)
    dur = get_duration_s(wav)
    assert abs(dur - 3.5) < 0.05


def test_chars_per_second():
    from luganda_pipeline.utils.audio_utils import chars_per_second

    assert chars_per_second("hello", 1.0) == 5.0
    assert chars_per_second("hello", 0.0) is None


def test_estimate_snr_on_sine():
    from luganda_pipeline.utils.audio_utils import estimate_snr

    # Pure sine — SNR should be very high
    t = torch.linspace(0, 2.0, 32_000)
    wave = (0.5 * torch.sin(2 * 3.14159 * 440 * t)).unsqueeze(0)
    snr = estimate_snr(wave, sr=16_000)
    assert snr > 10.0


def test_preprocess_decode_audio_item_dict_bytes():
    from luganda_pipeline.preprocessing.audio import _decode_audio_item

    wav = _make_wav_bytes(sr=22_050)
    waveform, sr = _decode_audio_item({"bytes": wav, "path": None}, target_sr=16_000)
    assert sr == 16_000
    assert waveform.shape[0] == 1
    assert waveform.shape[1] > 0


def test_preprocess_decode_audio_item_audio_decoder_like():
    from luganda_pipeline.preprocessing.audio import _decode_audio_item

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
    waveform, sr = _decode_audio_item(decoder, target_sr=16_000)
    assert sr == 16_000
    assert waveform.shape == (1, 16_000)
