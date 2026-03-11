"""
Stage 4 — Machine Translation (Luganda → English)
==================================================
Supported backends:
  - "nllb"        : Hugging Face translation pipeline (NLLB-style models)
  - "nllb_custom" : Custom NLLB tokenizer + M2M generation (for jq fine-tune)
  - "sunbird_api" : Sunbird AI REST API (requires SUNBIRD_API_KEY)

Produces the ``text_eng`` column by translating each ``text_lug`` record.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset

from luganda_pipeline.utils.logging import get_logger

log = get_logger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"([.?!])")
_HAS_DELIMITER_RE = re.compile(r"[.?!]")
_DEFAULT_LANGUAGE_TOKEN_IDS = {
    "eng": 256047,
    "ach": 256111,
    "lgg": 256008,
    "lug": 256110,
    "nyn": 256002,
    "teo": 256006,
}


def _normalize_lang_code(lang: str) -> str:
    """Normalize language tags like lug_Latn -> lug."""
    return lang.split("_", 1)[0].strip().lower()


def _chunk_text(text: str, chunk_size: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]


def _split_text_with_delimiters(text: str) -> list[str]:
    split_result = _SENTENCE_SPLIT_RE.split(text)
    combined = [
        split_result[i] + split_result[i + 1]
        for i in range(0, len(split_result) - 1, 2)
    ]
    if len(split_result) % 2 != 0 and split_result[-1].strip():
        combined.append(split_result[-1])
    return [part.strip() for part in combined if part.strip()]


def _segment_text(text: str, chunk_size: int) -> list[str]:
    if _HAS_DELIMITER_RE.search(text):
        return _split_text_with_delimiters(text)
    return _chunk_text(text, chunk_size=chunk_size)


# ─────────────────────────────────────────────────────────────────────────────
#  NLLB backend
# ─────────────────────────────────────────────────────────────────────────────

class NLLBTranslator:
    """
    Wrapper around facebook/nllb-200-distilled-600M for batched Lug→Eng translation.
    """

    def __init__(self, cfg: dict) -> None:
        from transformers import pipeline as hf_pipeline

        model_id = cfg["translation"]["model"]
        dtype_str = cfg["translation"].get("dtype", "bfloat16")
        device = cfg["translation"].get("device", "auto")
        num_beams = cfg["translation"].get("num_beams", 4)
        src_lang = cfg["translation"].get("src_lang", "lug_Latn")
        tgt_lang = cfg["translation"].get("tgt_lang", "eng_Latn")
        if "_" not in src_lang:
            src_lang = {
                "lug": "lug_Latn",
                "eng": "eng_Latn",
            }.get(src_lang.lower(), src_lang)
        if "_" not in tgt_lang:
            tgt_lang = {
                "lug": "lug_Latn",
                "eng": "eng_Latn",
            }.get(tgt_lang.lower(), tgt_lang)

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
        }.get(dtype_str, torch.bfloat16)

        if device == "auto":
            device_id = 0 if torch.cuda.is_available() else -1
        elif device == "cpu":
            device_id = -1
        else:
            device_id = int(device)

        log.info(f"Loading translation model: {model_id} (device={device_id}, dtype={dtype_str})")

        self._pipe = hf_pipeline(
            "translation",
            model=model_id,
            tokenizer=model_id,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            device=device_id,
            torch_dtype=torch_dtype,
            max_length=cfg["translation"]["max_length"],
            num_beams=num_beams,
        )
        self._batch_size = cfg["translation"]["batch_size"]

    def translate_batch(self, texts: list[str]) -> list[str]:
        """Translate a list of Luganda strings to English."""
        results = self._pipe(texts, batch_size=self._batch_size)
        return [r["translation_text"].strip() for r in results]


# ─────────────────────────────────────────────────────────────────────────────
#  Custom NLLB + M2M backend (reference-model compatible)
# ─────────────────────────────────────────────────────────────────────────────

class CustomNLLBTranslator:
    """
    Custom translation backend aligned with the reference script:
    - NllbTokenizer (can be a base tokenizer checkpoint)
    - M2M100ForConditionalGeneration (fine-tuned checkpoint)
    - Manual language token control via first token + forced BOS.
    """

    def __init__(self, cfg: dict) -> None:
        from transformers import M2M100ForConditionalGeneration, NllbTokenizer

        t_cfg = cfg["translation"]
        model_id = t_cfg["model"]
        tokenizer_id = t_cfg.get("tokenizer_model", model_id)
        self._chunk_size = int(t_cfg.get("chunk_size", 20))
        self._max_length = int(t_cfg.get("max_length", 100))
        self._max_input_length = int(t_cfg.get("max_input_length", 256))
        self._num_beams = int(t_cfg.get("num_beams", 5))
        self._generation_batch_size = int(
            t_cfg.get("generation_batch_size", t_cfg.get("batch_size", 16))
        )
        dtype_str = t_cfg.get("dtype", "float32")

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype_str, torch.float32)

        device_cfg = t_cfg.get("device", "auto")
        if device_cfg == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_cfg)

        language_token_ids_cfg = t_cfg.get("language_token_ids", _DEFAULT_LANGUAGE_TOKEN_IDS)
        self._language_token_ids = {str(k): int(v) for k, v in language_token_ids_cfg.items()}

        self._src_lang = _normalize_lang_code(t_cfg.get("src_lang", "lug"))
        self._tgt_lang = _normalize_lang_code(t_cfg.get("tgt_lang", "eng"))
        if self._src_lang not in self._language_token_ids:
            raise ValueError(f"Missing source language token id for '{self._src_lang}'")
        if self._tgt_lang not in self._language_token_ids:
            raise ValueError(f"Missing target language token id for '{self._tgt_lang}'")

        self._src_lang_token_id = self._language_token_ids[self._src_lang]
        self._tgt_lang_token_id = self._language_token_ids[self._tgt_lang]

        log.info(
            "Loading custom MT model: "
            f"{model_id} with tokenizer={tokenizer_id} on {self.device} (dtype={dtype_str})"
        )

        self.tokenizer = NllbTokenizer.from_pretrained(tokenizer_id)
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        )
        self.model.to(self.device)
        self.model.eval()

        # GPU throughput knobs (safe defaults on Ampere/Hopper+).
        if self.device.type == "cuda" and t_cfg.get("allow_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True

        if self.device.type == "cuda" and t_cfg.get("compile", False):
            try:
                self.model = torch.compile(self.model)
            except Exception as exc:
                log.warning(f"torch.compile disabled due to runtime error: {exc}")

    def _translate_segments_batched(self, segments: list[str]) -> list[str]:
        if not segments:
            return []

        translations: list[str] = []
        for start in range(0, len(segments), self._generation_batch_size):
            seg_batch = [s.strip() for s in segments[start : start + self._generation_batch_size]]
            inputs = self.tokenizer(
                seg_batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self._max_input_length,
            ).to(self.device)

            # Match reference behavior: explicit source language token in first position.
            inputs["input_ids"][:, 0] = self._src_lang_token_id
            with torch.inference_mode():
                translated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self._tgt_lang_token_id,
                    max_length=self._max_length,
                    num_beams=self._num_beams,
                )

            translations.extend(
                self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            )
        return [t.strip() for t in translations]

    def translate_batch(self, texts: list[str]) -> list[str]:
        text_segments: list[list[str]] = [
            _segment_text(text, chunk_size=self._chunk_size) for text in texts
        ]
        flat_segments = [segment for segments in text_segments for segment in segments]
        flat_translations = self._translate_segments_batched(flat_segments)

        results: list[str] = []
        cursor = 0
        for segments in text_segments:
            n = len(segments)
            if n == 0:
                results.append("")
                continue
            chunk = flat_translations[cursor : cursor + n]
            cursor += n
            results.append(" ".join(part for part in chunk if part).strip())
        return results


# ─────────────────────────────────────────────────────────────────────────────
#  Sunbird API backend
# ─────────────────────────────────────────────────────────────────────────────

class SunbirdAPITranslator:
    """
    Calls the Sunbird AI Translation REST API.
    Requires environment variables: SUNBIRD_API_KEY, SUNBIRD_API_URL.
    """

    def __init__(self) -> None:
        import requests  # noqa: F401 – just check it is available

        self._api_key = os.environ.get("SUNBIRD_API_KEY", "")
        self._api_url = os.environ.get("SUNBIRD_API_URL", "https://api.sunbird.ai/v1")

        if not self._api_key:
            raise EnvironmentError("SUNBIRD_API_KEY is not set. Cannot use sunbird_api backend.")

    def translate_batch(self, texts: list[str]) -> list[str]:
        """Translate a list of Luganda strings to English via Sunbird API."""
        import requests

        url = f"{self._api_url}/translate"
        translations: list[str] = []

        for text in texts:
            resp = requests.post(
                url,
                json={"text": text, "source_language": "lug", "target_language": "eng"},
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=30,
            )
            resp.raise_for_status()
            translations.append(resp.json().get("translation", "").strip())
            time.sleep(0.05)  # polite rate limiting

        return translations


# ─────────────────────────────────────────────────────────────────────────────
#  Batch map helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_translate_fn(translator: Any):
    """Return a Dataset.map-compatible function using `translator`."""
    def _translate_batch(batch: dict) -> dict:
        try:
            batch["text_eng"] = translator.translate_batch(batch["text_lug"])
        except Exception as exc:
            log.warning(f"Translation batch failed: {exc}. Filling with empty strings.")
            batch["text_eng"] = [""] * len(batch["text_lug"])
        return batch
    return _translate_batch


# ─────────────────────────────────────────────────────────────────────────────
#  Stage entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_translation(cfg: dict) -> Dataset:
    """
    Run Stage 4: translate ``text_lug`` → ``text_eng``.

    Parameters
    ----------
    cfg : dict
        Parsed pipeline configuration.

    Returns
    -------
    Dataset  saved to cfg['paths']['translated']
    """
    from datasets import load_from_disk

    in_path  = Path(cfg["paths"]["filtered"])    / "filtered_dataset"
    out_path = Path(cfg["paths"]["translated"]) / "translated_dataset"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("[bold]Stage 4 — Machine Translation (Lug → Eng)[/bold]")
    ds: Dataset = load_from_disk(str(in_path))
    log.info(f"  Input: {len(ds):,} records")

    backend = cfg["translation"].get("backend", "nllb")
    log.info(f"  Backend: {backend}")

    if backend == "sunbird_api":
        translator = SunbirdAPITranslator()
    elif backend in {"nllb_custom", "custom_nllb"}:
        translator = CustomNLLBTranslator(cfg)
    else:
        translator = NLLBTranslator(cfg)

    translate_fn = _make_translate_fn(translator)

    ds = ds.map(
        translate_fn,
        batched=True,
        batch_size=cfg["translation"]["batch_size"],
        desc="Translating Lug→Eng",
    )

    # Drop any records where translation failed (empty text_eng)
    n_before = len(ds)
    ds = ds.filter(lambda x: bool(x["text_eng"].strip()), desc="Removing empty translations")
    removed = n_before - len(ds)
    if removed:
        log.warning(f"  Removed {removed:,} records with empty translations")

    log.info(f"  Output: {len(ds):,} records")
    log.info(f"  Saving → {out_path}")
    ds.save_to_disk(str(out_path))

    return ds
