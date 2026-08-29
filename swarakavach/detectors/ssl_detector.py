"""Layer B — SSL detector: wav2vec2 fine-tuned on ASVspoof5.

Loads a community-published ASVspoof5-finetuned wav2vec2 classifier at runtime
(cached by HF hub). Prosody-robust, language-agnostic representation — the
generalization workhorse of the ensemble.
"""
import numpy as np

from .. import config

_state = {"model": None, "fx": None, "spoof_idx": None, "failed": False}


def _load():
    if _state["failed"]:
        return False
    if _state["model"] is not None:
        return True
    if not config.SSL_ENABLED:
        return False
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        mid = config.SSL_MODEL_ID
        _state["model"] = AutoModelForAudioClassification.from_pretrained(mid).eval()
        _state["fx"] = AutoFeatureExtractor.from_pretrained(mid)
        if config.SSL_SPOOF_IDX >= 0:
            _state["spoof_idx"] = config.SSL_SPOOF_IDX
        else:
            id2label = _state["model"].config.id2label
            spoof = [i for i, l in id2label.items()
                     if any(k in str(l).lower() for k in ("spoof", "fake", "aivoice", "synthetic"))]
            _state["spoof_idx"] = spoof[0] if spoof else 1
        return True
    except Exception:
        _state["failed"] = True
        return False


def available() -> bool:
    return _load()


def score(wav: np.ndarray, sr: int = config.SAMPLE_RATE):
    """Spoof probability from the SSL classifier; None if unavailable."""
    if not _load():
        return None
    import torch

    fx = _state["fx"]
    inputs = fx(np.asarray(wav, dtype=np.float32), sampling_rate=sr,
                return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = _state["model"](**inputs).logits
        p = torch.softmax(logits, dim=-1)[0]
    return float(p[_state["spoof_idx"]])
