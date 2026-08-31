"""Full-system ablation — makes the 'five independent questions' claim measured.

Two complementary views:

1. SCENARIO CALLS (call-level verdicts): for each layer subset
   A -> A+B -> A+B+C -> A+B+C+D -> A+B+C+D+E, run the four finale calls
   and print the verdict. Shows WHAT each layer adds that the others can't:
   A/B answer 'synthetic?', C 'this person?', D 'risky call?', E 'dangerous ask?'

2. CORPUS EER (audio layers): A alone, B alone, A+B — the honest audio-only
   ablation on the demo corpus (C/D/E are call-level layers; they don't
   produce per-audio-chunk scores on a static corpus).

Usage: python scripts/eval_ablation.py
"""
import csv

import numpy as np

from swarakavach import config
from swarakavach.asr import transcribe
from swarakavach.audio import iter_chunks, load_wav
from swarakavach.detectors import artifact, context as ctx_engine, intent, speaker, ssl_detector
from swarakavach.fusion import fuse, level_for
from swarakavach.train_utils import eer
from swarakavach.audio import degrade

CTX = {"caller_id_match": "false", "enrolled_contact": "true", "tx_amount": 5_000_000}

SUBSETS = [
    ("A", ["artifact"]),
    ("A+B", ["artifact", "ssl"]),
    ("A+B+C", ["artifact", "ssl", "speaker"]),
    ("A+B+C+D", ["artifact", "ssl", "speaker", "context"]),
    ("A+B+C+D+E", ["artifact", "ssl", "speaker", "context", "intent"]),
]


def run_call(wav, transcript, subset, speaker_id="regional_manager"):
    """Fused risk events for one call using only the layers in `subset`."""
    traj = []
    for chunk in iter_chunks(wav, config.CHUNK_SAMPLES, config.CHUNK_OVERLAP_SAMPLES):
        lp = {}
        if "artifact" in subset:
            lp["artifact"] = artifact.score(chunk)
        if "ssl" in subset:
            lp["ssl"] = ssl_detector.score(chunk)
        if "speaker" in subset:
            lp["speaker"] = speaker.score(speaker_id, chunk)
        if "context" in subset:
            lp["context"] = ctx_engine.score(CTX)
        if "intent" in subset:
            lp["intent"] = intent.score(transcript)
        traj.append(fuse(lp))
    med = float(np.median(traj)) if traj else 0.0
    return level_for(med), round(med, 3)


def scenario_ablation():
    from pathlib import Path

    sd = config.DATA_DIR / "demo" / "scenario"
    calls = [
        ("genuine", sd / "manager_real.wav"),
        ("clone(XTTS)", sd / "manager_clone.wav"),
        ("attack_EN", sd / "attacker_tts.wav"),
        ("attack_HI", sd / "attacker_hindi.wav"),
        ("social_eng", sd / "social_engineer_real.wav"),  # optional real-human scam
    ]
    calls = [(n, p) for n, p in calls if p.exists()]

    print("== SCENARIO CALL ABLATION (verdict per call, median decision rule) ==")
    hdr = f"{'layers':<12}" + "".join(f"{n:>14}" for n, _ in calls)
    print(hdr)

    # enroll the manager for layer C in all subsets that include it
    real = sd / "manager_real.wav"
    if real.exists():
        speaker.enroll("regional_manager", load_wav(real))

    transcripts = {}
    for n, p in calls:
        transcripts[n] = transcribe(load_wav(p)) or None  # live ASR, honest input

    for name, subset in SUBSETS:
        row = f"{name:<12}"
        for call_name, path in calls:
            lvl, med = run_call(load_wav(path), transcripts[call_name], subset)
            row += f"{lvl[:8]+'/'+str(med):>14}"
        print(row)


def corpus_ablation():
    with open(config.DATA_DIR / "demo" / "labels.csv") as f:
        rows = list(csv.DictReader(f))
    a_scores, b_scores, ab_scores, ys = [], [], [], []
    for r in rows:
        wav = load_wav(r["path"])
        y = int(r["label"])
        a = artifact.score(wav)
        b = ssl_detector.score(wav)
        a_scores.append(a if a is not None else 0.5)
        b_scores.append(b if b is not None else 0.5)
        ab_scores.append(fuse({"artifact": a, "ssl": b}))
        ys.append(y)
    ys = np.array(ys)
    print("\n== CORPUS EER ABLATION (demo corpus, audio layers) ==")
    print(f"{'layers':<12}{'EER':>8}")
    print(f"{'A':<12}{eer(np.array(a_scores), ys):>8.4f}")
    print(f"{'B':<12}{eer(np.array(b_scores), ys):>8.4f}")
    print(f"{'A+B':<12}{eer(np.array(ab_scores), ys):>8.4f}")
    print("(C/D/E are call-level layers — see the scenario table above)")


def hindi_call_level_fa():
    """Call-level false-alarm rate on REAL Hindi human speech (the number that
    matters for 'blocked legitimate transactions')."""
    import pathlib

    hd = config.DATA_DIR / "hindi-human" / "chunks"
    labels = hd / "labels.csv"
    if not labels.exists():
        return
    with open(labels) as f:
        rows = list(csv.DictReader(f))
    # group chunks back into source clips by audio_id
    groups = {}
    for r in rows:
        stem = pathlib.Path(r["path"]).stem
        aid = "_".join(stem.split("_")[:-1])  # strip chunk idx suffix
        groups.setdefault(aid, []).append(r["path"])
    flagged = 0
    n = 0
    for aid, paths in sorted(groups.items()):
        scores = [artifact.score(load_wav(p)) for p in paths]
        med = float(np.median(scores))
        n += 1
        if med >= 0.25:  # MID+ = caution actions; >=0.45 = HOLD territory
            flagged += 1
    print("\n== HINDI FALSE-ALARM (real human speech) ==")
    print(f"chunk-level >0.5 rate: measured previously 12.9%")
    print(f"CALL-level (median rule) MID+ verdicts: {flagged}/{n} calls "
          f"= {flagged/max(n,1)*100:.1f}%")
    print("A transaction is held only when the CALL median is high — chunk spikes decay.")


if __name__ == "__main__":
    scenario_ablation()
    corpus_ablation()
    hindi_call_level_fa()
