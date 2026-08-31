"""Experiment: fusion weight search on held-out data.

Layer B (Hemgg wav2vec2) measures 0.51 EER (chance) on our corpus — does its
0.28 fusion weight help or hurt? Grid-search artifact/ssl weights (and
temperature) on the SPEAKER-DISJOINT held-out val set of the demo corpus,
reporting fused EER for each candidate.

Usage: python scripts/experiments/fusion_grid.py
"""
import csv
import itertools

import numpy as np

from swarakavach import config
from swarakavach.audio import load_wav
from swarakavach.detectors import artifact, ssl_detector
from swarakavach.fusion import fuse
from swarakavach.train_utils import eer, speaker_key

import random


def heldout_rows(frac=0.3, seed=1):
    """Speaker-disjoint held-out subset of the demo corpus."""
    with open(config.DATA_DIR / "demo" / "labels.csv") as f:
        rows = [(r["path"], int(r["label"])) for r in csv.DictReader(f)]
    groups = {}
    for p, l in rows:
        groups.setdefault(speaker_key(p), []).append((p, l))
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    hold = set(keys[: max(2, int(frac * len(keys)))])
    return [it for k in hold for it in groups[k]]


def main():
    rows = heldout_rows()
    print(f"held-out rows: {len(rows)} (speaker-disjoint, AMR-degraded = phone operating point)")

    from swarakavach.codecs import degrade_codec

    data = []
    for p, y in rows:
        wav = degrade_codec(load_wav(p), "amr")
        a = artifact.score(wav)
        b = ssl_detector.score(wav)
        data.append((a if a is not None else 0.5, b if b is not None else 0.5, y))

    A = np.array([d[0] for d in data])
    B = np.array([d[1] for d in data])
    ys = np.array([d[2] for d in data])
    print(f"layer A EER alone: {eer(A, ys):.4f} | layer B EER alone: {eer(B, ys):.4f}")

    print(f"{'w_artifact':>10} {'w_ssl':>6} {'T':>5} {'fused EER':>10}")
    best = (None, 1.0)
    for wa, wb in [(0.50, 0.28), (0.60, 0.20), (0.70, 0.15), (0.80, 0.10),
                   (0.85, 0.08), (0.90, 0.06), (1.0, 0.0), (0.55, 0.25),
                   (0.65, 0.15), (0.75, 0.12)]:
        scores = [fuse({"artifact": a, "ssl": b},
                       weights={"artifact": wa, "ssl": wb}) for a, b in zip(A, B)]
        for T in (1.3, 1.0):
            # fuse already applies config temperature; emulate T by re-fusing
            # with a monkey-free trick: fuse() accepts temperature kwarg
            scores = [fuse({"artifact": a, "ssl": b}, weights={"artifact": wa, "ssl": wb},
                           temperature=T) for a, b in zip(A, B)]
            e = eer(np.array(scores), ys)
            tag = " <- current" if (wa, wb, T) == (0.50, 0.28, 1.3) else ""
            print(f"{wa:>10.2f} {wb:>6.2f} {T:>5.1f} {e:>10.4f}{tag}")
            if e < best[1]:
                best = ((wa, wb, T), e)
    print(f"\nBEST: weights=({best[0][0]}, {best[0][1]}), T={best[0][2]} -> EER {best[1]:.4f}")


if __name__ == "__main__":
    main()
