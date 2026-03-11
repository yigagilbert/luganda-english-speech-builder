"""Shared audio helper functions used across pipeline stages."""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F


def load_audio_bytes(audio_bytes: bytes, target_sr: int = 16_000) -> tuple[torch.Tensor, int]:
    """
    Load raw audio bytes into a (1, T) float32 tensor at target_sr.

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio file bytes (WAV, FLAC, MP3, etc.)
    target_sr : int
        Target sample rate in Hz.

    Returns
    -------
    waveform : torch.Tensor, shape (1, T)
    sample_rate : int
    """
    buf = io.BytesIO(audio_bytes)
    waveform, sr = torchaudio.load(buf)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        waveform = F.resample(waveform, sr, target_sr)

    return waveform, target_sr


def tensor_to_bytes(waveform: torch.Tensor, sample_rate: int = 16_000) -> bytes:
    """
    Encode a float32 waveform tensor as 16-bit PCM WAV bytes.

    Parameters
    ----------
    waveform : torch.Tensor, shape (1, T) or (T,)
    sample_rate : int

    Returns
    -------
    bytes
        Raw WAV file bytes.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Clamp and convert to int16
    waveform_int16 = (waveform.clamp(-1.0, 1.0) * 32767).to(torch.int16)

    buf = io.BytesIO()
    torchaudio.save(buf, waveform_int16, sample_rate, format="wav", bits_per_sample=16)
    return buf.getvalue()


def peak_normalise(waveform: torch.Tensor, target_dbfs: float = -1.0) -> torch.Tensor:
    """
    Peak-normalise waveform to `target_dbfs`.

    Parameters
    ----------
    waveform : torch.Tensor
    target_dbfs : float
        Target peak level in dBFS (e.g. -1.0).

    Returns
    -------
    torch.Tensor  (same shape as input)
    """
    peak = waveform.abs().max()
    if peak < 1e-8:
        return waveform  # Silence — nothing to normalise
    target_linear = 10 ** (target_dbfs / 20.0)
    return waveform * (target_linear / peak)


def estimate_snr(waveform: torch.Tensor, sr: int, frame_ms: int = 30) -> float:
    """
    Estimate wideband SNR using a simple energy-based approach
    (median frame energy as noise floor estimate).

    Parameters
    ----------
    waveform : torch.Tensor, shape (1, T)
    sr : int
    frame_ms : int  Frame length in milliseconds.

    Returns
    -------
    float  Estimated SNR in dB (returns 0.0 on silent audio).
    """
    frame_len = int(sr * frame_ms / 1000)
    wav = waveform.squeeze(0).numpy()

    # Split into non-overlapping frames
    n_frames = len(wav) // frame_len
    if n_frames < 2:
        return 0.0

    frames = wav[: n_frames * frame_len].reshape(n_frames, frame_len)
    energies = (frames ** 2).mean(axis=1)

    if energies.max() < 1e-10:
        return 0.0

    noise_floor = np.percentile(energies, 10)  # bottom 10% = noise estimate
    signal_peak = np.percentile(energies, 90)  # top 10% = speech estimate

    if noise_floor < 1e-10:
        return 60.0  # Practically noiseless

    snr_db = 10 * np.log10(signal_peak / noise_floor)
    return float(snr_db)


def get_duration_s(audio_bytes: bytes) -> float:
    """Return audio duration in seconds from raw bytes without full decode."""
    buf = io.BytesIO(audio_bytes)
    info = torchaudio.info(buf)
    return info.num_frames / info.sample_rate


def chars_per_second(text: str, duration_s: float) -> Optional[float]:
    """Compute characters-per-second ratio. Returns None if duration <= 0."""
    if duration_s <= 0:
        return None
    return len(text.strip()) / duration_s
