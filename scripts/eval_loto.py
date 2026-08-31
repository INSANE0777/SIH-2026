"""Leave-one-generator-out (LOTO) evaluation — the generalization claim.

The question: does the detector catch a generator it has NEVER seen in
training? For each attack family F: train on ALL other corpora (demo + clone
families != F + LibriSpeech + Hindi humans) — the full diversity stack — and
test EER on [held-out family F + humans].

NOTE (experiment log EXP-2): fold training must use the FULL corpus. An early
version trained folds on the clone-corpus only (385 clips) and reported xtts
0.21 — that measured data scarcity, not generalization. With full-corpus
folds the same model scores ~0.001 on the xtts fold.

Usage: python scripts/eval_loto.py [--epochs 12]
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch

from swarakavach import config
from swarakavach.audio import load_wav
from swarakavach.features import lfcc
from swarakavach.train_utils import eer, per_speaker_balance, speaker_key, train_cnn

# Fixed human holdout (deterministic, all folds): unseen-in-training bonafide
# speakers for the false-alarm side of each fold. EER needs BOTH classes in the
# test set — an all-attack test set makes EER degenerate (that was the EXP-2
# grid artifact: '0.0008' was an undefined EER, not a result).
HUMAN_HOLDOUT_PAT = re.compile(r"li(1089|1188|121|1221)_|ih_hi_m_agri_")


def fold_items(hold: str):
    """Full-diversity fold: train on everything EXCEPT family `hold` and the
    fixed human-holdout speakers; test = held-out attacks + those humans.
    Attacks unseen (generalization) AND humans unseen (false alarms) — the
    honest two-sided fold."""
    train, test = [], []
    csvs = [
        config.DATA_DIR / "demo" / "labels.csv",
        config.DATA_DIR / "clone-corpus" / "chunks" / "labels.csv",
        config.DATA_DIR / "libri-human" / "chunks" / "labels.csv",
        config.DATA_DIR / "hindi-human" / "chunks" / "labels.csv",
    ]
    for csv_path in csvs:
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                path, label = r["path"], int(r["label"])
                if label == 0 and HUMAN_HOLDOUT_PAT.search(path):
                    test.append((path, label))  # held-out human, all folds
                elif ((hold in path) or (r.get("family") == hold)) and label == 1:
                    test.append((path, label))  # held-out attack family
                else:
                    train.append((path, label))
    return per_speaker_balance(train), test


def score_fold(model, test):
    model.eval()
    scores, ys = [], []
    with torch.no_grad():
        for path, label in test:
            feats = lfcc(load_wav(path))
            feats = ((feats - feats.mean(axis=1, keepdims=True))
                     / (feats.std(axis=1, keepdims=True) + 1e-6))
            x = torch.from_numpy(feats).float().unsqueeze(0)
            scores.append(float(torch.softmax(model(x), -1)[0][1]))
            ys.append(label)
    return eer(np.array(scores), np.array(ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--codec-augment", action="store_true",
                    help="train folds with real codec augmentation (needs ffmpeg)")
    args = ap.parse_args()

    with open(config.DATA_DIR / "clone-corpus" / "chunks" / "labels.csv") as f:
        families = sorted({r["family"] for r in csv.DictReader(f)} - {"human"})
    print(f"families: {families} (full-diversity folds, codec_augment={args.codec_augment})")
    print(f"{'held-out family':<18} {'train n':>8} {'test n':>8} {'EER':>8}")

    results = {}
    for hold in families:
        train, test = fold_items(hold)
        model = train_cnn(train, epochs=args.epochs, silent=True,
                          codec_augment=args.codec_augment)
        e = score_fold(model, test)
        results[hold] = e
        print(f"{hold:<18} {len(train):>8} {len(test):>8} {e:>8.4f}")

    mean_eer = float(np.mean(list(results.values())))
    print(f"\nmean held-out EER (full-diversity folds): {mean_eer:.4f}")
    print("hold-out = the detector never trained on this generator family;")
    print("this is the honest cross-generator generalization number.")


if __name__ == "__main__":
    main()
