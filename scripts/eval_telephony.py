"""Telephony robustness eval: EER of each layer under channel degradation.

Telephone audio is not 16 kHz clean WAV. This script pushes the corpus through
increasingly hostile channel simulation and reports how each layer degrades:
  clean       16 kHz full band
  narrowband  8 kHz bandlimit (PSTN channel)
  mulaw       narrowband + G.711 mu-law 8-bit companding
  noisy       mulaw + 20 dB SNR white noise
  pl          noisy + 2% random 20 ms frame drops (packet loss)

Usage: python scripts/eval_telephony.py [--labels data/demo/labels.csv]
Note: layer A was trained on clean audio only — the degradation deltas are the
honest 'why we need codec-aware training' numbers, not a flaw to hide.
"""
import argparse
import csv

import numpy as np

from swarakavach import config
from swarakavach.audio import degrade, load_wav
from swarakavach.detectors import artifact, ssl_detector
from swarakavach.fusion import fuse

LEVELS = ["clean", "narrowband", "mulaw", "noisy", "pl"]


def eer(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    y = labels[order]
    P, N = float(y.sum()), float((1 - y).sum())
    fn = np.cumsum(y)
    tn = np.cumsum(1 - y)
    fpr = (N - tn) / max(N, 1.0)
    fnr = fn / max(P, 1.0)
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(config.DATA_DIR / "demo" / "labels.csv"))
    args = ap.parse_args()

    with open(args.labels) as f:
        rows = list(csv.DictReader(f))
    print(f"n={len(rows)}  (levels: {' > '.join(LEVELS)})")
    print(f"{'level':<12} {'EER art':>8} {'EER ssl':>8} {'EER fused':>10}")

    for level in LEVELS:
        art, ssl, fused, ys = [], [], [], []
        for r in rows:
            wav = degrade(load_wav(r["path"]), level)
            y = int(r["label"])
            a = artifact.score(wav)
            s = ssl_detector.score(wav)
            art.append(a if a is not None else 0.5)
            ssl.append(s if s is not None else 0.5)
            fused.append(fuse({"artifact": a, "ssl": s}))
            ys.append(y)
        ys = np.array(ys)
        print(f"{level:<12} {eer(np.array(art), ys):>8.4f} "
              f"{eer(np.array(ssl), ys):>8.4f} {eer(np.array(fused), ys):>10.4f}")


if __name__ == "__main__":
    main()
