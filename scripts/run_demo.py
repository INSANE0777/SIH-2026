"""CLI finale-demo rehearsal: enroll the 'regional manager', then run four calls.

Call 1 — genuine manager    → LOW      → transaction proceeds
Call 2 — cloned manager     → HIGH     → HOLD + liveness challenge + supervisor
Call 3 — TTS attacker (EN)  → CRITICAL → all actions (SSL layer misses it; layer A catches)
Call 4 — TTS attacker (HI)  → CRITICAL → multilingual attack; intent rules fire in Hindi

Usage: python scripts/run_demo.py [--scenario-dir data/demo/scenario]
"""
import argparse
from pathlib import Path

from swarakavach import config
from swarakavach.audio import iter_chunks, load_wav
from swarakavach.detectors import artifact, context as ctx_engine, intent, speaker, ssl_detector
from swarakavach.fusion import RiskEngine

CTX = {"caller_id_match": "false", "enrolled_contact": "true", "tx_amount": 5_000_000}

TRANSCRIPTS = {
    # genuine call: routine, no social-engineering markers
    "genuine": "Team, please review the monthly compliance report when you get a chance.",
    # cloned manager: urgent transfer + secrecy + authority
    "clone": ("Hello, this is the regional manager. I need you to approve the fifty lakh "
              "rupees transfer immediately. It is extremely urgent. Do not discuss this "
              "with anyone else."),
    # English TTS attacker: OTP + transfer + urgency + authority
    "attacker": ("This is the bank manager calling. Please share the OTP and your PIN "
                 "right now so we can complete the urgent rupees transfer."),
    # Hindi TTS attacker: transfer + urgency + secrecy + authority (Devanagari rules)
    "hindi": ("नमस्ते, मैं क्षेत्रीय प्रबंधक बोल रहा हूँ। पचास लाख रुपये का ट्रांसफर "
              "तुरंत स्वीकृत करें। यह बात किसी से न कहें।"),
}

BREAKDOWN_LABELS = [
    ("voice_authenticity_risk", "voice authenticity"),
    ("speaker_mismatch_risk", "speaker mismatch  "),
    ("call_context_risk", "call context      "),
    ("social_engineering_risk", "social engineering"),
]


def call(name: str, wav_path: Path, speaker_id: str, transcript: str) -> RiskEngine:
    engine = RiskEngine(dict(CTX))
    print(f"\n-- incoming call: {name} ({wav_path.name}) --")
    if transcript:
        ix = intent.explain(transcript)
        print(f"  intent rules: {ix['matched'] or 'none'}")
    for i, chunk in enumerate(iter_chunks(load_wav(wav_path)), 1):
        ev = engine.update({
            "artifact": artifact.score(chunk),
            "ssl": ssl_detector.score(chunk),
            "speaker": speaker.score(speaker_id, chunk),
            "context": ctx_engine.score(engine.context),
            "intent": intent.score(transcript),
        })
        bar = "#" * int(ev["risk"] * 40)
        print(f"  chunk {i:02d}  risk={ev['risk']:.3f} [{bar:<40}] {ev['level']}")
    v = engine.verdict()
    print(f"  VERDICT: level={v['level']} risk_final={v['risk_final']} peak={v['risk_peak']}")
    print(f"  actions: {', '.join(v['actions']) or 'none -- transaction proceeds'}")
    print("  why (explainable breakdown, 0-100 risk):")
    for key, label in BREAKDOWN_LABELS:
        val = v["breakdown"].get(key)
        if val is not None:
            print(f"    {label:<20} {val:3d}  {'#' * (val // 5)}")
    print(f"    {'FINAL RISK':<20} {int(round(v['risk_peak'] * 100)):3d}  "
          f"{'#' * int(v['risk_peak'] * 20)}  [{v['level']}]")
    return engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario-dir", default=str(config.DATA_DIR / "demo" / "scenario"))
    ap.add_argument("--speaker-id", default="regional_manager")
    args = ap.parse_args()
    sd = Path(args.scenario_dir)
    real, clone = sd / "manager_real.wav", sd / "manager_clone.wav"
    if not (real.exists() and clone.exists()):
        raise SystemExit("scenario files missing -- run scripts/make_demo_data.py first")

    print("enrolling regional manager voiceprint ...")
    speaker.enroll(args.speaker_id, load_wav(real))
    print("layers:", f"artifact={artifact.available()}", f"ssl={ssl_detector.available()}")

    calls = [("genuine", "genuine manager", real, TRANSCRIPTS["genuine"]),
             ("cloned", "CLONED manager", clone, TRANSCRIPTS["clone"])]
    attacker, hindi = sd / "attacker_tts.wav", sd / "attacker_hindi.wav"
    if attacker.exists():
        calls.append(("attacker_en", "TTS ATTACKER (English)", attacker, TRANSCRIPTS["attacker"]))
    if hindi.exists():
        calls.append(("attacker_hi", "TTS ATTACKER (Hindi)", hindi, TRANSCRIPTS["hindi"]))

    engines = {}
    for key, name, path, tr in calls:
        engines[key] = call(name, path, args.speaker_id, tr)

    peaks = {k: max(e.trajectory) for k, e in engines.items()}
    print(f"\npeaks: { {k: round(v, 3) for k, v in peaks.items()} }")
    ok = peaks["cloned"] > peaks["genuine"] + 0.1
    print(f"{'PASS' if ok else 'FAIL'}: clone peak {peaks['cloned']:.3f} "
          f"vs genuine peak {peaks['genuine']:.3f}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
