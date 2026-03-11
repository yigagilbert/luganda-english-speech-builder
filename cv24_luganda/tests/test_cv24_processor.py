"""
tests/test_cv24_processor.py
============================
Unit tests for common_voice_24_luganda.py
Run with:  pytest tests/test_cv24_processor.py -v
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torchaudio

# Patch sys.path so we can import the top-level script
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from common_voice_24_luganda import (
    Config,
    _normalise_text,
    _process_clip,
    apply_metadata_filters,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_mp3_bytes(duration_s: float = 2.0, sr: int = 22_050) -> bytes:
    """Create a tiny valid WAV (not MP3, but torchaudio.load handles both)."""
    t = torch.linspace(0, duration_s, int(sr * duration_s))
    wave = (0.4 * torch.sin(2 * 3.14159 * 440 * t)).unsqueeze(0)
    wave_i16 = (wave * 32767).to(torch.int16)
    buf = io.BytesIO()
    torchaudio.save(buf, wave_i16, sr, format="wav")
    return buf.getvalue()


def _write_tmp_audio(tmp_path: Path, duration_s: float = 2.0) -> Path:
    p = tmp_path / "clip_0001.wav"
    p.write_bytes(_make_mp3_bytes(duration_s))
    return p


# ─────────────────────────────────────────────────────────────────────────────
#  Text normalisation
# ─────────────────────────────────────────────────────────────────────────────

class TestNormaliseText:
    def test_strips_control_chars(self):
        raw = "Webale\u200b nnyo\u00a0 okuyamba"
        result = _normalise_text(raw)
        assert "\u200b" not in result
        assert "\u00a0" not in result

    def test_collapses_whitespace(self):
        assert _normalise_text("  Mwesigwa   okulaba  ") == "Mwesigwa okulaba"

    def test_nfc_normalisation(self):
        import unicodedata
        raw = "cafe\u0301"   # decomposed é
        result = _normalise_text(raw)
        assert unicodedata.is_normalized("NFC", result)

    def test_empty_string(self):
        assert _normalise_text("") == ""
        assert _normalise_text("   ") == ""


# ─────────────────────────────────────────────────────────────────────────────
#  Metadata filter
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyMetadataFilters:

    def _base_df(self, tmp_path: Path, n: int = 5) -> tuple[pd.DataFrame, Config]:
        """Create a minimal DataFrame and Config with real audio files."""
        paths = []
        for i in range(n):
            p = tmp_path / f"clip_{i:04d}.wav"
            p.write_bytes(_make_mp3_bytes())
            paths.append(f"clip_{i:04d}.wav")

        df = pd.DataFrame({
            "path":       paths,
            "text":       [f"Webale nnyo clip {i}" for i in range(n)],
            "up_votes":   [3, 0, 2, 1, 5],
            "down_votes": [0, 1, 0, 2, 0],
            "age":        ["twenties"] * n,
            "gender":     ["male"] * n,
        })

        cfg = Config(data_dir=tmp_path, output_dir=tmp_path / "out")
        cfg.clips_dir = tmp_path   # point clips_dir at tmp_path directly

        return df, cfg

    def test_removes_low_votes(self, tmp_path):
        df, cfg = self._base_df(tmp_path)
        cfg.min_up_votes = 2
        result = apply_metadata_filters(df, cfg)
        assert all(pd.to_numeric(result["up_votes"]) >= 2)

    def test_validated_only_removes_down_voted(self, tmp_path):
        df, cfg = self._base_df(tmp_path)
        cfg.validated_only = True
        result = apply_metadata_filters(df, cfg)
        for _, row in result.iterrows():
            assert int(row["up_votes"]) > int(row["down_votes"])

    def test_removes_missing_text(self, tmp_path):
        df, cfg = self._base_df(tmp_path)
        df.loc[0, "text"] = ""
        result = apply_metadata_filters(df, cfg)
        assert all(result["text"].str.len() > 0)

    def test_text_normalised_in_output(self, tmp_path):
        df, cfg = self._base_df(tmp_path)
        df.loc[2, "text"] = "  extra   spaces  here  "
        result = apply_metadata_filters(df, cfg)
        for t in result["text"]:
            assert "  " not in t
            assert not t.startswith(" ")


# ─────────────────────────────────────────────────────────────────────────────
#  Per-clip audio processing
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessClip:

    def test_valid_clip_returns_record(self, tmp_path):
        audio_path = _write_tmp_audio(tmp_path, duration_s=2.0)
        result = _process_clip((
            0, str(audio_path), "Webale nnyo", "cv24_lg_0000001",
            16_000, 0.5, 30.0, 0.0,   # min_snr=0 to always pass
        ))
        assert result is not None
        assert result["id"] == "cv24_lg_0000001"
        assert result["text_lug"] == "Webale nnyo"
        assert isinstance(result["audio_lug"]["bytes"], bytes)
        assert len(result["audio_lug"]["bytes"]) > 0

    def test_too_short_returns_none(self, tmp_path):
        audio_path = _write_tmp_audio(tmp_path, duration_s=0.1)
        result = _process_clip((
            0, str(audio_path), "text", "id_001",
            16_000, 0.5, 30.0, 0.0,
        ))
        assert result is None

    def test_too_long_returns_none(self, tmp_path):
        audio_path = _write_tmp_audio(tmp_path, duration_s=35.0)
        result = _process_clip((
            0, str(audio_path), "text", "id_001",
            16_000, 0.5, 30.0, 0.0,
        ))
        assert result is None

    def test_output_is_16khz(self, tmp_path):
        audio_path = _write_tmp_audio(tmp_path, duration_s=2.0)
        result = _process_clip((
            0, str(audio_path), "text", "id_001",
            16_000, 0.5, 30.0, 0.0,
        ))
        assert result is not None
        # Decode the output bytes and verify sample rate
        wav, sr = torchaudio.load(io.BytesIO(result["audio_lug"]["bytes"]))
        assert sr == 16_000
        assert wav.shape[0] == 1  # mono

    def test_bad_path_returns_none(self, tmp_path):
        result = _process_clip((
            0, str(tmp_path / "nonexistent.mp3"), "text", "id_001",
            16_000, 0.5, 30.0, 0.0,
        ))
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
#  Config defaults
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_archive_path(self):
        cfg = Config(data_dir=Path("/data"))
        assert cfg.archive_path == Path("/data/mcv-scripted-lg-v24.0.tar.gz")

    def test_clips_dir(self):
        cfg = Config(data_dir=Path("/data"))
        assert cfg.clips_dir == Path("/data/extracted/lg/clips")

    def test_tsv_dir(self):
        cfg = Config(data_dir=Path("/data"))
        assert cfg.tsv_dir == Path("/data/extracted/lg")
