"""CLI finale-demo rehearsal: enroll the 'regional manager', then run two calls.

Call 1 — genuine manager   → expect LOW risk, transaction proceeds.
Call 2 — cloned manager    → expect HIGH/CRITICAL risk, transaction HOLD,
                             MFA challenge, supervisor alert, liveness challenge.

Usage: python scripts/run_demo.py [--scenario-dir data/demo/scenario]
"""
import argparse
from pathlib import Path

from swarakavach import config
from swarakavach.audio import iter_chunks, load_wav
from swarakavach.detectors import artifact, context as ctx_engine, speaker, ssl_detector
from swarakavach.fusion import RiskEngine

CTX = {"caller_id_match": "false", "enrolled_contact": "true", "tx_amount": 5_000_000}


def call(name: str, wav_path: Path, speaker_id: str) -> RiskEngine:
    engine = RiskEngine(dict(CTX))
    print(f"\n-- incoming call: {name} ({wav_path.name}) --")
    for i, chunk in enumerate(iter_chunks(load_wav(wav_path)), 1):
        ev = engine.update({
            "artifact": artifact.score(chunk),
            "ssl": ssl_detector.score(chunk),
            "speaker": speaker.score(speaker_id, chunk),
            "context": ctx_engine.score(engine.context),
        })
        bar = "#" * int(ev["risk"] * 40)
        print(f"  chunk {i:02d}  risk={ev['risk']:.3f} [{bar:<40}] {ev['level']}")
    v = engine.verdict()
    print(f"  VERDICT: level={v['level']} risk_final={v['risk_final']} peak={v['risk_peak']}")
    print(f"  actions: {', '.join(v['actions']) or 'none -- transaction proceeds'}")
    print(f"  layers : {v['layer_means']}")
    return engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario-dir", default=str(config.DATA_DIR / "demo" / "scenario"))
    ap.add_argument("--speaker-id", default="regional_manager")
    args = ap.parse_args()
    sd = Path(args.scenario_dir)
    real, clone = sd / "manager_real.wav", sd / "manager_clone.wav"
    if not (real.exists() and clone.exists()):
        raise SystemExit("scenario files missing — run scripts/make_demo_data.py first")

    print("enrolling regional manager voiceprint ...")
    speaker.enroll(args.speaker_id, load_wav(real))
    print("layers:", f"artifact={artifact.available()}", f"ssl={ssl_detector.available()}")

    e_real = call("genuine manager", real, args.speaker_id)
    e_clone = call("CLONED manager", clone, args.speaker_id)
    attacker = sd / "attacker_tts.wav"
    e_att = None
    if attacker.exists():
        e_att = call("TTS ATTACKER (different voice)", attacker, args.speaker_id)

    peaks = {"genuine": max(e_real.trajectory), "clone": max(e_clone.trajectory)}
    if e_att is not None:
        peaks["attacker"] = max(e_att.trajectory)
    print(f"\npeaks: { {k: round(v, 3) for k, v in peaks.items()} }")
    ok = peaks["clone"] > peaks["genuine"] + 0.1
    print(f"{'PASS' if ok else 'FAIL'}: clone peak {peaks['clone']:.3f} "
          f"vs genuine peak {peaks['genuine']:.3f}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
