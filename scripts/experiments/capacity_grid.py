"""Experiment: layer-A capacity/features grid, measured on the HARDEST fold.

The xtts LOTO fold (train without xtts, test on xtts clones) is the one place
layer A errs (held-out EER 0.21). Grid:
  - LFCC dims: 20 (current) vs 40
  - filterbanks: 60 (current) vs 128
  - CNN width: base (~60k params) vs wide (~180k) — still edge-sized
One winner, one number: xtts-fold EER vs the 0.212 baseline.

Usage: python scripts/experiments/capacity_grid.py [--epochs 12]
"""
import argparse
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))          # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))     # repo root

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from swarakavach import config
from swarakavach.audio import load_wav
from swarakavach.train_utils import LFCCDataset, collate, eer
from eval_loto import fold_items   # scripts/ on sys.path

# --- LFCC feature dim/bank override (module-global swap) ---------------
import swarakavach.features as F
import swarakavach.detectors.artifact as art_mod


def make_lfcc(n_lfcc, n_filter):
    import torchaudio
    return torchaudio.transforms.LFCC(
        sample_rate=16000, n_lfcc=n_lfcc, n_filter=n_filter,
        speckwargs={"n_fft": 512, "win_length": 400, "hop_length": 160, "center": True})


class WideCNN(nn.Module):
    """Same shape as ArtifactCNN but ~3x channels (still ~180k params)."""

    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 96, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(96),
            nn.Conv1d(96, 192, 5, padding=2, stride=2), nn.ReLU(), nn.BatchNorm1d(192),
            nn.Conv1d(192, 192, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(192, 96), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(96, 2),
        )

    def forward(self, x):
        return self.net(x)


def xtts_fold_items():
    """The FIXED two-sided xtts fold from eval_loto (EXP-3): held-out xtts
    attacks + never-seen human holdout speakers in the test set."""
    return fold_items("xtts")


def train_eval(train, test, n_lfcc, n_filter, wide, epochs, device):
    torch.manual_seed(0)
    F._lfcc_tf = make_lfcc(n_lfcc, n_filter)  # swap the feature extractor

    model = (WideCNN(n_lfcc) if wide else art_mod.ArtifactCNN(n_lfcc)).to(device)
    dl = DataLoader(LFCCDataset(train, augment=True), batch_size=16, shuffle=True,
                    collate_fn=collate)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n_pos = sum(1 for _, l in train if l == 1)
    n_neg = len(train) - n_pos
    w = torch.tensor([len(train) / (2 * max(n_neg, 1)), len(train) / (2 * max(n_pos, 1))]).to(device)
    lossf = nn.CrossEntropyLoss(weight=w)
    for _ in range(epochs):
        model.train()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            lossf(model(x), y).backward()
            opt.step()
    model.eval()
    scores, ys = [], []
    with torch.no_grad():
        for p, y in test:
            wav = load_wav(p)
            f_ = F.lfcc(wav)
            f_ = ((f_ - f_.mean(axis=1, keepdims=True)) / (f_.std(axis=1, keepdims=True) + 1e-6))
            x = torch.from_numpy(f_).float().unsqueeze(0).to(device)
            scores.append(float(torch.softmax(model(x), -1)[0][1]))
            ys.append(y)
    F._lfcc_tf = None  # restore
    return eer(np.array(scores), np.array(ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, test = xtts_fold_items()
    print(f"xtts fold: train={len(train)} test={len(test)} on {device}")
    print(f"{'n_lfcc':>7} {'n_filter':>9} {'model':>6} {'params':>8} {'xtts EER':>10}")

    for n_lfcc, n_filter, wide in [(20, 60, False), (20, 60, True),
                                  (40, 60, False), (40, 128, False),
                                  (40, 128, True), (60, 128, True)]:
        e = train_eval(train, test, n_lfcc, n_filter, wide, args.epochs, device)
        m = WideCNN(n_lfcc) if wide else art_mod.ArtifactCNN(n_lfcc)
        n_par = sum(p.numel() for p in m.parameters())
        tag = " <- baseline" if (n_lfcc, n_filter, wide) == (20, 60, False) else ""
        print(f"{n_lfcc:>7} {n_filter:>9} {'wide' if wide else 'base':>6} {n_par:>8} {e:>10.4f}{tag}")


if __name__ == "__main__":
    main()
