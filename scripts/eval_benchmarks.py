"""Benchmark: per-layer and fused EER on a labeled corpus.

Usage:
    python scripts/eval_benchmarks.py --labels data/demo/labels.csv
    python scripts/eval_benchmarks.py --asvspoof-dir <root>   # dev protocol not needed; labels.csv format
EER = point where false-accept and false-reject rates cross (lower = better).
"""
import argparse
import csv

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

    art, ssl, fused, ys = [], [], [], []
    for r in rows:
        wav = load_wav(r["path"])
        y = int(r["label"])
        a = artifact.score(wav)
        s = ssl_detector.score(wav)
        lp = {"artifact": a, "ssl": s}
        art.append(a if a is not None else 0.5)
        ssl.append(s if s is not None else 0.5)
        fused.append(fuse(lp))
        ys.append(y)

    ys = np.array(ys)
    print(f"n={len(ys)}  (spoof={ys.sum()}, bonafide={len(ys)-ys.sum()})")
    print(f"EER artifact : {eer(np.array(art), ys):.4f}   [layer available: {artifact.available()}]")
    print(f"EER ssl      : {eer(np.array(ssl), ys):.4f}   [layer available: {ssl_detector.available()}]")
    print(f"EER fused    : {eer(np.array(fused), ys):.4f}")


if __name__ == "__main__":
    main()
