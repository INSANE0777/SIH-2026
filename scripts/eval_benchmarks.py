"""Benchmark: per-layer and fused EER on a labeled corpus.

Usage:
    python scripts/eval_benchmarks.py --labels data/demo/labels.csv
    python scripts/eval_benchmarks.py --asvspoof-dir <root>   # dev protocol not needed; labels.csv format
EER = point where false-accept and false-reject rates cross (lower = better).
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from swarakavach import config
from swarakavach.audio import load_wav
from swarakavach.detectors import artifact, ssl_detector
from swarakavach.fusion import fuse


def eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """EER where higher score = more spoofy. Lower is better."""
    order = np.argsort(scores)
    s, y = scores[order], labels[order]
    P, N = float(y.sum()), float((1 - y).sum())
    # threshold after index i: predict spoof for scores > s[i]
    fn = np.cumsum(y)          # spoofs at/below threshold → missed
    tn = np.cumsum(1 - y)      # bonafide at/below → correctly cleared
    tp = P - fn
    fp = N - tn
    fpr = fp / max(N, 1.0)
    fnr = fn / max(P, 1.0)
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(config.DATA_DIR / "demo" / "labels.csv"))
    ap.add_argument("--speaker-id", default=None, help="optional enrolled speaker for layer C")
    args = ap.parse_args()

    with open(args.labels) as f:
        rows = list(csv.DictReader(f))
    # per-class split: neural TTS attacks vs vocoder-proxy attacks — the proxy
    # class is corpus training variety, not a claimed capability; report both.
    def cls(r):
        return "voc_proxy" if "voc_" in Path(r["path"]).name else "main"

    art, ssl, fused, ys = [], [], [], []
    art_m, ssl_m, fused_m, ys_m = [], [], [], []   # main classes only
    for r in rows:
        wav = load_wav(r["path"])
        y = int(r["label"])
        a = artifact.score(wav)
        s = ssl_detector.score(wav)
        row = (a, s, fuse({"artifact": a, "ssl": s}), y)
        art.append(row[0] if row[0] is not None else 0.5)
        ssl.append(row[1] if row[1] is not None else 0.5)
        fused.append(row[2])
        ys.append(y)
        if cls(r) == "main":
            art_m.append(row[0] if row[0] is not None else 0.5)
            ssl_m.append(row[1] if row[1] is not None else 0.5)
            fused_m.append(row[2])
            ys_m.append(y)

    ys = np.array(ys)
    ys_m = np.array(ys_m)
    print(f"n={len(ys)}  (spoof={ys.sum()}, bonafide={len(ys)-ys.sum()})")
    print(f"EER artifact : {eer(np.array(art), ys):.4f}   [layer available: {artifact.available()}]")
    print(f"EER ssl      : {eer(np.array(ssl), ys):.4f}   [layer available: {ssl_detector.available()}]")
    print(f"EER fused    : {eer(np.array(fused), ys):.4f}")
    print(f"-- excluding the vocoder-proxy class (real speech vs neural TTS only): --")
    print(f"EER artifact : {eer(np.array(art_m), ys_m):.4f}")
    print(f"EER ssl      : {eer(np.array(ssl_m), ys_m):.4f}")
    print(f"EER fused    : {eer(np.array(fused_m), ys_m):.4f}")


if __name__ == "__main__":
    main()
