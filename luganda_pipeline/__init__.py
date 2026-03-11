"""
luganda_pipeline
================
End-to-end pipeline for building a Luganda–English paired speech dataset.

Stages
------
1. Ingestion     – pull & unify HuggingFace datasets
2. Preprocessing – resample, VAD-trim, normalise audio
3. Filtering     – SNR, CPS, dedup, text quality gates
4. Translation   – Lug → Eng via NLLB-200 / Sunbird MT
5. TTS           – text_eng → audio_eng via Sunbird / SpeechT5
6. Assembly      – enforce schema, assign IDs, push to Hub
7. QA            – stats report, plots, dataset card
"""

__version__ = "1.0.0"
__author__ = "Your Name"
