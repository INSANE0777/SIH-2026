"""Fusion + risk engine: late fusion → calibration → EMA trajectory → policy.

The PS demands a *dynamic* impersonation risk score while the conversation is
ongoing. RiskEngine consumes per-chunk layer probabilities, produces a smoothed
trajectory, maps to risk levels and the response action set.
"""
import math

import numpy as np

from . import config


def fuse(layer_probs: dict, weights: dict | None = None, temperature: float = None) -> float:
    """Weighted late fusion of available layer probabilities.

    Missing layers (None) are dropped and weights renormalized. The weighted
    mean is treated as a logit-of-evidence and passed through a temperature-
    softened sigmoid for calibration.
    """
    weights = weights or config.LAYER_WEIGHTS
    temperature = temperature or config.TEMPERATURE
    num = den = 0.0
    for k, p in layer_probs.items():
        if p is None:
            continue
        w = float(weights.get(k, 0.0))
        num += w * float(np.clip(p, 0.0, 1.0))
        den += w
    if den == 0.0:
        return 0.0
    p = num / den
    # temperature-softened sigmoid: pushes extreme fusion outputs toward 0.5
    logit = math.log((p + 1e-6) / (1.0 - p + 1e-6))
    return float(1.0 / (1.0 + math.exp(-logit / temperature)))


def level_for(risk: float) -> str:
    for thr, name in config.RISK_LEVELS:
        if risk >= thr:
            return name
    return "LOW"


def actions_for(level: str) -> list:
    return list(config.ACTIONS.get(level, []))


class RiskEngine:
    """Per-call session engine: chunk updates → smoothed trajectory → verdict."""

    def __init__(self, context: dict | None = None, ema_alpha: float = None):
        self.context = context or {}
        self.ema_alpha = ema_alpha or config.EMA_ALPHA
        self.ema = None
        self.trajectory: list[float] = []
        self.layer_history: list[dict] = []
        self.chunk_count = 0

    def update(self, layer_probs: dict) -> dict:
        """Feed one chunk's layer probabilities; returns the risk event."""
        self.chunk_count += 1
        fused = fuse(layer_probs)
        self.ema = fused if self.ema is None else (
            self.ema_alpha * fused + (1 - self.ema_alpha) * self.ema)
        self.trajectory.append(self.ema)
        self.layer_history.append(
            {k: (None if v is None else round(float(v), 4)) for k, v in layer_probs.items()})
        lvl = level_for(self.ema)
        return self._event(lvl)

    def _event(self, lvl: str) -> dict:
        return {
            "chunk": self.chunk_count,
            "risk": round(self.ema, 4),
            "level": lvl,
            "actions": actions_for(lvl),
            "layers": self.layer_history[-1],
            "trajectory": [round(x, 4) for x in self.trajectory],
        }

    def verdict(self) -> dict:
        """Final call verdict. Level comes from the MEDIAN of the smoothed
        trajectory — robust to single-chunk detector wobble on genuine calls,
        while sustained attack patterns still drive the level up. Peak is kept
        as an informational field (and for alert thresholds)."""
        peak = max(self.trajectory) if self.trajectory else 0.0
        final = self.trajectory[-1] if self.trajectory else 0.0
        med = float(np.median(self.trajectory)) if self.trajectory else 0.0
        lvl = level_for(max(final, med))
        actions = sorted({a for l in set(map(level_for, [med, final])) for a in actions_for(l)})
        means = _layer_means(self.layer_history)
        return {
            "chunks": self.chunk_count,
            "risk_final": round(final, 4),
            "risk_peak": round(peak, 4),
            "level": lvl,
            "actions": actions,
            "layer_means": means,
            "breakdown": _breakdown(means),
        }


def _breakdown(means: dict) -> dict:
    """Explainable risk decomposition — why the system acted.

    Maps raw layers onto judge-facing risk dimensions (each 0-100, higher =
    riskier): voice authenticity (A+B), speaker mismatch (C), call context (D),
    social engineering (E)."""
    def pct(v):
        return None if v is None else int(round(v * 100))

    synth = [means[k] for k in ("artifact", "ssl") if means.get(k) is not None]
    return {
        "voice_authenticity_risk": pct(float(np.mean(synth))) if synth else None,
        "speaker_mismatch_risk": pct(means.get("speaker")),
        "call_context_risk": pct(means.get("context")),
        "social_engineering_risk": pct(means.get("intent")),
    }


def _layer_means(history: list[dict]) -> dict:
    out = {}
    for k in history[0] if history else []:
        vals = [h[k] for h in history if h.get(k) is not None]
        out[k] = round(float(np.mean(vals)), 4) if vals else None
    return out
