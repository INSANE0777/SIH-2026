"""ASR layer — live transcript source for the intent engine (Detector E).

faster-whisper (CTranslate2) with Whisper small on CUDA when available, tiny on
CPU otherwise. Language auto-detect handles Hindi/Devanagari + English + code-
switching within a single call.

Two modes:
- transcribe(wav):        one-shot (REST /v1/analyze when transcript not given)
- StreamingTranscriber:   rolling transcription for the WS call path; feeds
  per-utterance text to the intent rules as the call progresses.

ponytail: whisper 'medium' is the floor for usable Hindi (small mangles
TTS Hindi beyond repair: 'तुरंत' -> 'तुर्ण्ट'); CUDA fp16 keeps RTF < 0.3.
CPU-only environments fall back to tiny (English acceptable, Hindi degraded).
Swap via SWK_ASR_MODEL env.
"""
import os
import time

import numpy as np

from . import config

_state = {"model": None, "failed": False}


class StreamingTranscriber:
    """Rolling window ASR: feed PCM chunks, get utterance texts as they close."""

    def __init__(self, language: str | None = None):
        self.language = language
        self.buf = np.zeros(0, dtype=np.float32)
        self.text_parts: list[str] = []
        self.last_emit_len = 0

    def feed(self, wav: np.ndarray) -> str | None:
        """Add audio; returns a new utterance transcript when enough has
        accumulated (>= 4 s since last emit), else None."""
        self.buf = np.concatenate([self.buf, np.asarray(wav, dtype=np.float32)])
        hop = 4 * config.SAMPLE_RATE
        if len(self.buf) - self.last_emit_len < hop:
            return None
        text = transcribe(self.buf)
        self.buf = np.zeros(0, dtype=np.float32)
        self.last_emit_len = 0
        if text and text.strip():
            self.text_parts.append(text.strip())
            return text.strip()
        return None


def _load():
    if _state["failed"]:
        return False
    if _state["model"] is not None:
        return True
    try:
        from faster_whisper import WhisperModel

        model_id = os.environ.get("SWK_ASR_MODEL", "medium")
        try:
            _state["model"] = WhisperModel(model_id, device="cuda", compute_type="float16")
        except Exception:
            # no CUDA / no model cache: tiny int8 CPU keeps the pipeline alive
            _state["model"] = WhisperModel("tiny", device="cpu", compute_type="int8")
        return True
    except Exception:
        _state["failed"] = True
        return False


def available() -> bool:
    return _load()


def transcribe(wav: np.ndarray) -> str:
    """One-shot transcription (16 kHz float32 mono). Returns '' on failure."""
    if not _load():
        return ""
    try:
        segs, _ = _state["model"].transcribe(
            np.asarray(wav, dtype=np.float32), language=None,
            beam_size=1, vad_filter=True)
        return " ".join(s.text for s in segs).strip()
    except Exception:
        return ""


def latency_probe(wav: np.ndarray) -> dict:
    """RTF + wall-clock latency of one transcription (used by the latency eval)."""
    t0 = time.perf_counter()
    text = transcribe(wav)
    dt = time.perf_counter() - t0
    return {"text": text[:80], "latency_s": round(dt, 3),
            "rtf": round(dt / max(len(wav) / config.SAMPLE_RATE, 1e-9), 3)}
