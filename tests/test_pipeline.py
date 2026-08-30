"""Runnable self-checks: the smallest tests that fail if core logic breaks.

Run:  python -m pytest tests/ -v   (or python -m tests.test_pipeline)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarakavach import config                       # noqa: E402
from swarakavach.audio import (bandlimit_8k, degrade, float_to_pcm16,  # noqa: E402
                               iter_chunks, mulaw_roundtrip, pcm16_to_float)
from swarakavach.detectors import context as ctx     # noqa: E402
from swarakavach.detectors import intent             # noqa: E402
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


def test_intent_rules_detect_social_engineering():
    scam = intent.score("This is the regional manager, approve the fifty lakh rupees "
                        "transfer immediately, do not discuss this with anyone.")
    hindi = intent.score("पचास लाख रुपये का ट्रांसफर तुरंत स्वीकृत करें। यह बात किसी से न कहें।")
    benign = intent.score("Team, please review the monthly compliance report.")
    assert intent.score(None) is None
    assert benign <= 0.2 < scam >= 0.7
    assert hindi >= 0.6
    ix = intent.explain("share the OTP and PIN immediately")
    assert "credentials" in ix["matched"] and "urgency" in ix["matched"]


def test_audit_chain_detects_tampering(tmp_path=None):
    import tempfile
    from pathlib import Path as P
    from swarakavach import audit
    with tempfile.TemporaryDirectory() as td:
        led = P(td) / "ledger.jsonl"
        r1 = audit.append({"type": "incident", "risk": 0.9}, path=led)
        r2 = audit.append({"type": "decision", "decision": "BLOCKED"}, path=led)
        ok, _ = audit.verify_chain(path=led)
        assert ok and r2["prev_hash"] == r1["hash"] and r1["index"] == 0
        # tamper with the first record
        recs = audit.read(led)
        recs[0]["risk"] = 0.1
        led.write_text("\n".join(__import__("json").dumps(r, ensure_ascii=False)
                                 for r in recs), encoding="utf-8")
        ok, broken = audit.verify_chain(path=led)
        assert not ok and broken == 0


def test_mulaw_and_bandlimit_are_sane():
    t = np.arange(16000, dtype=np.float32) / 16000.0
    sine = (0.9 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)  # band-limited
    y = mulaw_roundtrip(sine)
    assert len(y) == len(sine) and np.max(np.abs(sine - y)) < 0.05
    z = bandlimit_8k(sine)   # 1 kHz sine survives the 4 kHz band limit
    assert len(z) == len(sine) and np.max(np.abs(z)) > 0.8 * np.max(np.abs(sine))
    noise = np.random.default_rng(3).uniform(-0.9, 0.9, 16000).astype(np.float32)
    for lvl in ("clean", "narrowband", "mulaw", "noisy", "pl"):
        d = degrade(noise, lvl)
        assert len(d) == len(noise)


def test_verdict_has_explainable_breakdown():
    eng = RiskEngine({"caller_id_match": "false", "tx_amount": 5_000_000})
    for _ in range(3):
        eng.update({"artifact": 0.9, "ssl": 0.9, "intent": 0.8, "context": 0.55})
    v = eng.verdict()
    bd = v["breakdown"]
    assert bd["voice_authenticity_risk"] == 90
    assert bd["social_engineering_risk"] == 80
    assert bd["speaker_mismatch_risk"] is None


def test_end_to_end_ordering():
    """Ordering property of the artifact layer: synthetic stays flagged and
    vocoding never lowers the score. When the demo corpus is present, also
    verify the real headline: real human speech scores far below neural TTS."""
    from swarakavach.detectors import artifact
    if not artifact.available():
        print("SKIP (no bootstrap checkpoint yet — run scripts/train_artifact.py)")
        return
    base = synth_phrase(seed=424242, dur=3.0)
    fake = vocode(base)
    assert artifact.score(fake) >= artifact.score(base) - 0.05  # vocode never lowers

    human = Path("data/human/jfk.wav")
    tts = Path("data/demo/scenario/manager_clone.wav")
    if not (human.exists() and tts.exists()):
        print("SKIP (no corpus — run scripts/make_demo_data.py for the full check)")
        return
    from swarakavach.audio import load_wav
    s_real = [artifact.score(c) for c in iter_chunks(load_wav(human))]
    s_tts = [artifact.score(c) for c in iter_chunks(load_wav(tts))]
    assert np.mean(s_tts) > np.mean(s_real) + 0.2, (
        f"neural TTS {np.mean(s_tts):.3f} vs human {np.mean(s_real):.3f}")


if __name__ == "__main__":
    for name, fn in sorted({k: v for k, v in globals().items()
                            if k.startswith("test_") and callable(v)}.items()):
        fn()
        print(f"PASS {name}")
    print("all checks passed")
