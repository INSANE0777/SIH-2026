"""Telephony robustness eval: EER of each layer under REAL channel degradation.

Telephone audio is not 16 kHz clean WAV. This script pushes the corpus through
REAL codec pipelines (ffmpeg AMR-NB 4.75k cellular, Opus 8k/16k VoIP, packet
loss, EVS-style narrowband+noise) and reports how each layer degrades:

  clean     16 kHz full band
  amr       AMR-NB 4.75 kbps roundtrip (worst cellular rate)
  opus      Opus 8 kbps roundtrip (low-rate VoIP)
  opus_wb   Opus 16 kbps roundtrip (normal VoIP)
  pl2       opus + 2% random 20 ms packet loss
  evs_like  AMR-NB + 30 dB noise (EVS-style profile, documented approximation)

Usage: python scripts/eval_telephony.py [--labels data/demo/labels.csv]
The codec-trained model's deltas vs the clean-trained one are the week-3 result.
"""
import argparse
import csv

import numpy as np

from swarakavach import config
from swarakavach.audio import load_wav
from swarakavach.codecs import CODEC_LEVELS, degrade_codec
from swarakavach.detectors import artifact, ssl_detector
from swarakavach.fusion import fuse
from swarakavach.train_utils import eer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(config.DATA_DIR / "demo" / "labels.csv"))
    args = ap.parse_args()

    with open(args.labels) as f:
        rows = list(csv.DictReader(f))
    print(f"n={len(rows)}  levels: {' > '.join(CODEC_LEVELS)}")
    print(f"{'level':<10} {'EER art':>8} {'EER ssl':>8} {'EER fused':>10}")

    for level in CODEC_LEVELS:
        art, ssl, fused, ys = [], [], [], []
        for r in rows:
            wav = degrade_codec(load_wav(r["path"]), level)
            y = int(r["label"])
            a = artifact.score(wav)
            s = ssl_detector.score(wav)
            art.append(a if a is not None else 0.5)
            ssl.append(s if s is not None else 0.5)
            fused.append(fuse({"artifact": a, "ssl": s}))
            ys.append(y)
        ys = np.array(ys)
        print(f"{level:<10} {eer(np.array(art), ys):>8.4f} "
              f"{eer(np.array(ssl), ys):>8.4f} {eer(np.array(fused), ys):>10.4f}")


if __name__ == "__main__":
    main()
