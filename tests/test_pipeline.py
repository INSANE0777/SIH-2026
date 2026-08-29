"""Runnable self-checks: the smallest tests that fail if core logic breaks.

Run:  python -m pytest tests/ -v   (or python -m tests.test_pipeline)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarakavach import config                       # noqa: E402
from swarakavach.audio import float_to_pcm16, iter_chunks, pcm16_to_float  # noqa: E402
from swarakavach.detectors import context as ctx     # noqa: E402
from swarakavach.fusion import RiskEngine, fuse      # noqa: E402
from swarakavach.synth import synth_phrase, vocode   # noqa: E402


def test_pcm16_roundtrip():
    x = np.random.default_rng(0).uniform(-0.9, 0.9, 5000).astype(np.float32)
    y = pcm16_to_float(float_to_pcm16(x))
    assert np.max(np.abs(x - y)) < 2e-5  # <= 0.5 LSB + clip


def test_chunks_cover_signal():
    x = np.zeros(40000, dtype=np.float32)
    chunks = list(iter_chunks(x, chunk=16000, overlap=8000))
    assert all(len(c) <= 16000 for c in chunks)
    assert sum(len(c) for c in chunks) >= 40000


def test_fusion_renormalizes_missing_layers():
    p_full = fuse({"artifact": 0.9, "ssl": 0.9, "speaker": None, "context": None})
    assert 0.5 < p_full <= 1.0
    p_none = fuse({"artifact": None, "ssl": None})
    assert p_none == 0.0


def test_ema_rises_on_transition_to_attack():
    eng = RiskEngine()
    for _ in range(3):                     # benign opening
        eng.update({"artifact": 0.05, "ssl": 0.05})
    for _ in range(5):                     # attack mid-call
        eng.update({"artifact": 0.9, "ssl": 0.9})
    assert eng.trajectory[-1] > eng.trajectory[0] + 0.3
    assert eng.verdict()["level"] in ("HIGH", "CRITICAL")


def test_high_risk_triggers_policy_actions():
    eng = RiskEngine({"caller_id_match": "false", "tx_amount": 5_000_000})
    ev = eng.update({"artifact": 0.9, "ssl": 0.95})
    assert "HOLD_TRANSACTION" in ev["actions"]
    assert "LIVENESS_CHALLENGE" in ev["actions"]


def test_context_rules():
    benign = ctx.score({"caller_id_match": "true", "enrolled_contact": "true"})
    hostile = ctx.score({"caller_id_match": "false", "known_fraud_flag": "true",
                         "tx_amount": 5_000_000, "intl_origin": "true"})
    assert ctx.score(None) is None
    assert benign <= 0.15 < hostile >= 0.75


def test_vocode_adds_detectable_artifacts():
    """The vocoder proxy must be separable from its bonafide source."""
    from swarakavach.features import lfcc_mean
    rng = np.random.default_rng(7)
    gaps = []
    for i in range(4):
        real = synth_phrase(seed=50 + i, dur=2.5)
        fake = vocode(real)
        a, b = lfcc_mean(real), lfcc_mean(fake)
        gaps.append(float(np.linalg.norm(a - b)))
    assert np.mean(gaps) > 0.05, f"vocoder proxy not separable: {gaps}"


def test_speaker_layer_requires_enrollment():
    from swarakavach.detectors import speaker
    assert speaker.score("nobody_enrolled", np.zeros(16000, dtype=np.float32)) is None


def test_end_to_end_ordering():
    """Full pipeline: cloned call must score higher risk than genuine call."""
    from swarakavach.detectors import artifact
    if not artifact.available():
        print("SKIP (no bootstrap checkpoint yet — run scripts/train_artifact.py)")
        return
    real = synth_phrase(seed=424242, dur=3.0)
    clone = vocode(real)
    eng_real, eng_fake = RiskEngine(), RiskEngine()
    for c in iter_chunks(real):
        eng_real.update({"artifact": artifact.score(c)})
    for c in iter_chunks(clone):
        eng_fake.update({"artifact": artifact.score(c)})
    assert max(eng_fake.trajectory) > max(eng_real.trajectory) + 0.1, (
        f"clone {max(eng_fake.trajectory):.3f} vs real {max(eng_real.trajectory):.3f}")


if __name__ == "__main__":
    for name, fn in sorted({k: v for k, v in globals().items()
                            if k.startswith("test_") and callable(v)}.items()):
        fn()
        print(f"PASS {name}")
    print("all checks passed")
