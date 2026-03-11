"""
Stage 5 — English TTS (audio_eng Generation)
=============================================
Supported backends:
  - "speecht5"    : microsoft/speecht5_tts + HiFiGAN vocoder (default)
  - "spark_tts"   : Spark-TTS fine-tuned LLM + BiCodecTokenizer
  - "sunbird_api" : Sunbird AI TTS REST API
  - "coqui"       : Coqui XTTS-v2 (requires TTS package)

Produces the ``audio_eng`` column as 16-bit mono WAV bytes at 16 kHz.
"""

from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import Any

import torch
import torchaudio
import torchaudio.functional as F
from datasets import Audio, Dataset

from luganda_pipeline.utils.logging import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  SpeechT5 backend (default)
# ─────────────────────────────────────────────────────────────────────────────

class SpeechT5Synthesizer:
    """
    Synthesises English speech using microsoft/speecht5_tts + HiFiGAN vocoder.
    Speaker embeddings are loaded from CMU Arctic xvectors.
    """

    NATIVE_SR = 16_000  # SpeechT5 outputs at 16 kHz natively

    def __init__(self, cfg: dict) -> None:
        from datasets import load_dataset as _load_ds
        from transformers import (
            SpeechT5ForTextToSpeech,
            SpeechT5HifiGan,
            SpeechT5Processor,
        )

        tts_cfg = cfg["tts"]
        model_id   = tts_cfg.get("speecht5_model", "microsoft/speecht5_tts")
        vocoder_id = tts_cfg["vocoder"]
        spk_ds_id  = tts_cfg["speaker_embeddings"]
        spk_idx    = tts_cfg.get("speaker_id", 7306)

        device_str = tts_cfg.get("device", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        log.info(f"Loading SpeechT5 model: {model_id} on {self.device}")

        self.processor = SpeechT5Processor.from_pretrained(model_id)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_id).to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_id).to(self.device)

        # Load pre-computed speaker xvectors
        log.info(f"Loading speaker embeddings from {spk_ds_id}")
        embeddings_ds = _load_ds(spk_ds_id, split="validation")
        self._speaker_embedding = (
            torch.tensor(embeddings_ds[spk_idx]["xvector"])
            .unsqueeze(0)
            .to(self.device)
        )

        self._target_sr = cfg["audio"]["sample_rate"]

    def synthesize(self, text: str) -> bytes:
        """
        Synthesise a single English text string and return 16-bit WAV bytes.

        Parameters
        ----------
        text : str  English text to synthesise.

        Returns
        -------
        bytes  16-bit PCM WAV at self._target_sr.
        """
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            speech: torch.Tensor = self.model.generate_speech(
                inputs["input_ids"],
                self._speaker_embedding,
                vocoder=self.vocoder,
            )  # shape: (T,) at NATIVE_SR

        speech = speech.cpu().unsqueeze(0)  # (1, T)

        # Resample if native SR differs from target
        if self.NATIVE_SR != self._target_sr:
            speech = F.resample(speech, self.NATIVE_SR, self._target_sr)

        # Encode to int16 WAV bytes
        speech_i16 = (speech.clamp(-1.0, 1.0) * 32767).to(torch.int16)
        buf = io.BytesIO()
        torchaudio.save(buf, speech_i16, self._target_sr, format="wav", bits_per_sample=16)
        return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
#  Spark-TTS backend
# ─────────────────────────────────────────────────────────────────────────────

class SparkTTSSynthesizer:
    """
    Synthesises speech using Spark-TTS and a fine-tuned LLM checkpoint.
    This follows the inference pattern in the provided reference notebook.
    """

    _SEMANTIC_TOKEN_RE = re.compile(r"<\|bicodec_semantic_(\d+)\|>")
    _GLOBAL_TOKEN_RE = re.compile(r"<\|bicodec_global_(\d+)\|>")

    def __init__(self, cfg: dict) -> None:
        try:
            from sparktts.models.audio_tokenizer import BiCodecTokenizer
        except ImportError as exc:
            raise ImportError(
                "Spark-TTS backend requires `sparktts` to be installed and importable."
            ) from exc

        tts_cfg = cfg["tts"]
        device_str = tts_cfg.get("device", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        self._target_sr = cfg["audio"]["sample_rate"]
        self._voice_prefix = str(tts_cfg.get("spark_voice_prefix", "248")).strip()
        self._temperature = float(tts_cfg.get("spark_temperature", 0.8))
        self._top_k = int(tts_cfg.get("spark_top_k", 50))
        self._top_p = float(tts_cfg.get("spark_top_p", 1.0))
        self._max_new_audio_tokens = int(tts_cfg.get("spark_max_new_audio_tokens", 2048))
        self._max_seq_length = int(tts_cfg.get("spark_max_seq_length", 2048))

        cache_root = Path(tts_cfg.get("spark_cache_dir", "data/models/spark_tts"))
        cache_root.mkdir(parents=True, exist_ok=True)

        model_ref = tts_cfg.get("model", "jq/spark-tts-salt")
        codec_ref = tts_cfg.get("spark_codec_model", "unsloth/Spark-TTS-0.5B")

        llm_root = self._resolve_model_ref(model_ref, cache_root / "llm")
        llm_model_path = llm_root / "LLM" if (llm_root / "LLM").exists() else llm_root
        codec_path = self._resolve_model_ref(codec_ref, cache_root / "codec")

        log.info(f"Loading Spark-TTS LLM from {llm_model_path} on {self.device}")
        self.model, self.tokenizer = self._load_spark_llm(str(llm_model_path))
        self.audio_tokenizer = BiCodecTokenizer(str(codec_path), str(self.device))
        self._native_sr = int(self.audio_tokenizer.config.get("sample_rate", 24_000))

    @staticmethod
    def _resolve_model_ref(model_ref: str, local_dir: Path) -> Path:
        model_path = Path(model_ref).expanduser()
        if model_path.exists():
            return model_path

        from huggingface_hub import snapshot_download

        local_dir.parent.mkdir(parents=True, exist_ok=True)
        resolved = snapshot_download(repo_id=model_ref, local_dir=str(local_dir))
        return Path(resolved)

    def _load_spark_llm(self, model_name: str):
        try:
            from unsloth import FastModel

            model, tokenizer = FastModel.from_pretrained(
                model_name=model_name,
                max_seq_length=self._max_seq_length,
                dtype=torch.float32,
                load_in_4bit=False,
            )
            FastModel.for_inference(model)
            model = model.to(self.device)
            model.eval()
            return model, tokenizer
        except Exception as unsloth_exc:
            log.warning(
                "Unsloth-based loading failed. Falling back to transformers AutoModel. "
                f"Cause: {unsloth_exc}"
            )
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            ).to(self.device)
            model.eval()
            return model, tokenizer

    def _build_prompt(self, text: str) -> str:
        text = text.strip()
        if self._voice_prefix and not re.match(r"^\s*\d+\s*:", text):
            text = f"{self._voice_prefix}: {text}"
        return "".join(
            [
                "<|task_tts|>",
                "<|start_content|>",
                text,
                "<|end_content|>",
                "<|start_global_token|>",
            ]
        )

    def _extract_generated_token_ids(self, generated_text: str) -> tuple[torch.Tensor, torch.Tensor]:
        semantic_matches = [int(token) for token in self._SEMANTIC_TOKEN_RE.findall(generated_text)]
        if not semantic_matches:
            raise RuntimeError("Spark-TTS generation returned no semantic tokens.")

        global_matches = [int(token) for token in self._GLOBAL_TOKEN_RE.findall(generated_text)]
        if not global_matches:
            global_matches = [0]

        pred_global_ids = torch.tensor(global_matches, dtype=torch.long, device=self.device).unsqueeze(0)
        pred_semantic_ids = torch.tensor(semantic_matches, dtype=torch.long, device=self.device).unsqueeze(0)
        return pred_global_ids, pred_semantic_ids

    def synthesize(self, text: str) -> bytes:
        prompt = self._build_prompt(text)
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self._max_new_audio_tokens,
                do_sample=True,
                temperature=self._temperature,
                top_k=self._top_k,
                top_p=self._top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1] :]
        generated_text = self.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False
        )[0]

        pred_global_ids, pred_semantic_ids = self._extract_generated_token_ids(generated_text)
        wav_np = self.audio_tokenizer.detokenize(pred_global_ids, pred_semantic_ids)
        if wav_np is None or len(wav_np) == 0:
            raise RuntimeError("Spark-TTS produced empty waveform output.")

        speech = torch.as_tensor(wav_np, dtype=torch.float32).flatten().unsqueeze(0).cpu()
        if self._native_sr != self._target_sr:
            speech = F.resample(speech, self._native_sr, self._target_sr)

        speech_i16 = (speech.clamp(-1.0, 1.0) * 32767).to(torch.int16)
        buf = io.BytesIO()
        torchaudio.save(buf, speech_i16, self._target_sr, format="wav", bits_per_sample=16)
        return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
#  Sunbird TTS API backend
# ─────────────────────────────────────────────────────────────────────────────

class SunbirdTTSSynthesizer:
    """
    Calls the Sunbird AI TTS REST API.
    Requires environment variables: SUNBIRD_API_KEY, SUNBIRD_API_URL.
    """

    def __init__(self, target_sr: int = 16_000) -> None:
        self._api_key = os.environ.get("SUNBIRD_API_KEY", "")
        self._api_url = os.environ.get("SUNBIRD_API_URL", "https://api.sunbird.ai/v1")
        self._target_sr = target_sr

        if not self._api_key:
            raise EnvironmentError("SUNBIRD_API_KEY not set. Cannot use sunbird_api TTS backend.")

    def synthesize(self, text: str) -> bytes:
        import requests

        resp = requests.post(
            f"{self._api_url}/tts",
            json={"text": text, "language": "eng", "format": "wav"},
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.content  # Raw WAV bytes from API


# ─────────────────────────────────────────────────────────────────────────────
#  Batch processing
# ─────────────────────────────────────────────────────────────────────────────

def _make_synth_fn(synthesizer: Any):
    """Return a Dataset.map-compatible batch function using `synthesizer`."""

    def _synth_batch(batch: dict) -> dict:
        audio_eng_items: list[dict | None] = []
        keep_mask: list[bool] = []

        for text in batch["text_eng"]:
            if not text.strip():
                audio_eng_items.append(None)
                keep_mask.append(False)
                continue
            try:
                wav_bytes = synthesizer.synthesize(text)
                audio_eng_items.append({"bytes": wav_bytes, "path": None})
                keep_mask.append(True)
            except Exception as exc:
                log.debug(f"TTS failed for text snippet: {exc}")
                audio_eng_items.append(None)
                keep_mask.append(False)

        # Apply mask
        for key in batch:
            batch[key] = [v for v, k in zip(batch[key], keep_mask) if k]
        batch["audio_eng"] = [a for a, k in zip(audio_eng_items, keep_mask) if k]
        return batch

    return _synth_batch


# ─────────────────────────────────────────────────────────────────────────────
#  Stage entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_tts(cfg: dict) -> Dataset:
    """
    Run Stage 5: generate ``audio_eng`` from ``text_eng``.

    Parameters
    ----------
    cfg : dict
        Parsed pipeline configuration.

    Returns
    -------
    Dataset  saved to cfg['paths']['synthesized']
    """
    from datasets import load_from_disk

    in_path  = Path(cfg["paths"]["translated"])  / "translated_dataset"
    out_path = Path(cfg["paths"]["synthesized"]) / "synthesized_dataset"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("[bold]Stage 5 — English TTS Synthesis[/bold]")
    ds: Dataset = load_from_disk(str(in_path))
    log.info(f"  Input: {len(ds):,} records")

    backend = cfg["tts"].get("backend", "speecht5")
    log.info(f"  Backend: {backend}")

    if backend == "sunbird_api":
        synthesizer = SunbirdTTSSynthesizer(target_sr=cfg["audio"]["sample_rate"])
    elif backend in {"spark_tts", "spark"}:
        synthesizer = SparkTTSSynthesizer(cfg)
    elif backend == "coqui":
        from luganda_pipeline.tts._coqui import CoquiSynthesizer  # optional heavy dep
        synthesizer = CoquiSynthesizer(cfg)
    else:
        synthesizer = SpeechT5Synthesizer(cfg)

    synth_fn = _make_synth_fn(synthesizer)

    ds = ds.map(
        synth_fn,
        batched=True,
        batch_size=cfg["tts"]["batch_size"],
        desc="Synthesising English audio",
    )

    # Cast new audio column
    ds = ds.cast_column("audio_eng", Audio(sampling_rate=cfg["audio"]["sample_rate"]))

    log.info(f"  Output: {len(ds):,} records")
    log.info(f"  Saving → {out_path}")
    ds.save_to_disk(str(out_path))

    return ds
