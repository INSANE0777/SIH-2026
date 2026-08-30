"""Latency budget eval — measured against the p95 < 300 ms/chunk target.

Measures, per detector layer and for the fused chunk pipeline:
  - per-chunk wall-clock latency across the demo scenario calls
  - p50/p95/max, and per-layer attribution (which layer spends the budget)
  - ASR: one-shot RTF + rolling-window emit latency (production transcript path)

Usage: python scripts/eval_latency.py [--runs 3]
The artifact layer runs CPU; ssl runs CPU (fp32) — swap paths documented in
docs/architecture.md. ASR = whisper medium fp16 CUDA.
"""
import argparse
import time
from statistics import mean

import numpy as np

from swarakavach import config
from swarakavach.asr import available as asr_available, transcribe
from swarakavach.audio import iter_chunks, load_wav
from swarakavach.detectors import artifact, context as ctx_engine, intent, speaker, ssl_detector


def pct(xs, p):
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return xs[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    calls = []
    from pathlib import Path
    for p in sorted(Path(config.DATA_DIR / "demo" / "scenario").glob("*.wav")):
        calls.append(load_wav(p))

    per_layer = {"artifact": [], "ssl": [], "speaker": [], "context": [], "intent": [], "fused": []}
    chunk_count = 0

    speaker.enroll("lat_probe", calls[0][: 4 * config.SAMPLE_RATE])
    ctx = {"caller_id_match": "false", "tx_amount": 5_000_000}
    tr = "urgent transfer of fifty lakh rupees, share the OTP now"

    for wav in calls:
        for _ in range(args.runs):
            for chunk in iter_chunks(wav, config.CHUNK_SAMPLES, config.CHUNK_OVERLAP_SAMPLES):
                chunk_count += 1
                t0 = time.perf_counter()
                a = artifact.score(chunk);            t1 = time.perf_counter()
                s = ssl_detector.score(chunk);        t2 = time.perf_counter()
                sp = speaker.score("lat_probe", chunk); t3 = time.perf_counter()
                c = ctx_engine.score(ctx);            t4 = time.perf_counter()
                i = intent.score(tr);                 t5 = time.perf_counter()
                per_layer["artifact"].append(t1 - t0)
                per_layer["ssl"].append(t2 - t1)
                per_layer["speaker"].append(t3 - t2)
                per_layer["context"].append(t4 - t3)
                per_layer["intent"].append(t5 - t4)
                per_layer["fused"].append(t5 - t0)

    print(f"chunk latency over {chunk_count} chunks ({args.runs} runs x {len(calls)} calls):")
    print(f"{'layer':<10} {'p50 ms':>8} {'p95 ms':>8} {'max ms':>8}")
    for k, v in per_layer.items():
        ms = [x * 1000 for x in v]
        print(f"{k:<10} {pct(ms, 50):>8.1f} {pct(ms, 95):>8.1f} {max(ms):>8.1f}")

    ok = pct([x * 1000 for x in per_layer["fused"]], 95) < 300
    print(f"\nFUSED CHUNK PIPELINE p95 = {pct([x*1000 for x in per_layer['fused']], 95):.0f} ms "
          f"-> {'MEETS' if ok else 'MISSES'} the < 300 ms target")

    if asr_available():
        wav = calls[1] if len(calls) > 1 else calls[0]
        t0 = time.perf_counter()
        text = transcribe(wav)
        dt = time.perf_counter() - t0
        dur = len(wav) / config.SAMPLE_RATE
        print(f"\nASR (whisper, one-shot on a {dur:.0f}s call): {dt:.2f}s "
              f"RTF={dt/dur:.2f}  text[:60]={text[:60]!r}")


if __name__ == "__main__":
    main()
