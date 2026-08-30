"""Shared training utilities for the artifact CNN (used by train_artifact.py,
eval_loto.py, and future real-data scripts)."""
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .audio import load_wav
from .detectors.artifact import ArtifactCNN
from .features import lfcc


class LFCCDataset(Dataset):
    """Train-time augmentation: gain, pitch shift (resample trick), light noise —
    forces speaker-invariant anti-spoof features on a tiny multi-speaker corpus."""

    def __init__(self, items, augment: bool = False, seed: int = 0):
        self.items = items
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def _aug(self, wav: np.ndarray) -> np.ndarray:
        x = wav * (0.7 + 0.6 * self.rng.random())               # gain 0.7-1.3
        if self.rng.random() < 0.5:                             # pitch +-1.5 st
            r = 2 ** (self.rng.uniform(-1.5, 1.5) / 12)
            x = np.interp(np.arange(0, len(x), r), np.arange(len(x)), x).astype(np.float32)
        if self.rng.random() < 0.5:                             # 30 dB noise
            sig = float(np.mean(x ** 2)) + 1e-12
            n = self.rng.standard_normal(len(x)).astype(np.float32)
            x = x + n * np.sqrt(sig / 10 ** (30 / 10.0))
        return np.clip(x, -1.0, 1.0)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        wav = load_wav(path)
        if self.augment:
            wav = self._aug(wav)
        f = lfcc(wav)
        f = (f - f.mean(axis=1, keepdims=True)) / (f.std(axis=1, keepdims=True) + 1e-6)
        return torch.from_numpy(f).float(), torch.tensor(label)


def collate(batch):
    # right-pad LFCC time dimension to batch max
    max_t = max(x.shape[1] for x, _ in batch)
    xs, ys = [], []
    for x, y in batch:
        pad = torch.zeros(x.shape[0], max_t)
        pad[:, : x.shape[1]] = x
        xs.append(pad)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


def speaker_key(path: str) -> str:
    """Group key for speaker-disjoint splits: strip family prefixes and chunk
    index(es), so a speaker and their clones share one group.
    human/jfk_03.wav -> jfk · xtts_jfk_03.wav -> jfk · edge_neerja_02 -> neerja
    """
    stem = re.sub(r"^(voc_|edge_|xtts_|human_)", "", Path(path).stem)
    return re.sub(r"(?:_\d+)+$", "", stem)


def eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """EER where higher score = more spoofy. Lower is better."""
    order = np.argsort(scores)
    y = labels[order]
    P, N = float(y.sum()), float((1 - y).sum())
    fn = np.cumsum(y)          # spoofs at/below threshold -> missed
    tn = np.cumsum(1 - y)      # bonafide at/below -> correctly cleared
    fpr = (N - tn) / max(N, 1.0)
    fnr = fn / max(P, 1.0)
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def per_speaker_balance(items):
    """Within each speaker group, oversample the minority class to the majority —
    a speaker's real clips must not be outnumbered 10:1 by their clones.
    Callers pass [(path, label)] tuples."""
    groups = {}
    for it in items:
        groups.setdefault(speaker_key(it[0]), []).append(it)
    balanced = []
    for grp in groups.values():
        pos = [it for it in grp if it[1] == 1]
        neg = [it for it in grp if it[1] == 0]
        if pos and neg:
            m = max(len(pos), len(neg))
            balanced += pos * (m // len(pos)) + neg * (m // len(neg))
        else:
            balanced += grp
    return balanced


def train_cnn(train_items, epochs: int = 30, silent: bool = False):
    """Train a fresh ArtifactCNN on [(path, label)] items with augmentation,
    per-SPEAKER class balancing (a speaker's real clips must not be outnumbered
    10:1 by their clones, or the CNN learns 'this voice = spoof'), and global
    class weighting. Returns the trained model (eval mode)."""
    random.seed(0)
    items = per_speaker_balance(list(train_items))
    random.shuffle(items)
    n_pos = sum(1 for _, l in items if l == 1)
    n_neg = len(items) - n_pos
    if n_neg and n_pos / max(n_neg, 1) > 1.5:
        mult = n_pos // n_neg
        items = items + [it for it in items if it[1] == 0] * mult
        random.shuffle(items)

    dl = DataLoader(LFCCDataset(items, augment=True), batch_size=16,
                    shuffle=True, collate_fn=collate)
    model = ArtifactCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    w = torch.tensor([len(items) / (2 * max(n_neg, 1)), len(items) / (2 * max(n_pos, 1))])
    lossf = nn.CrossEntropyLoss(weight=w)

    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x, y in dl:
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        if not silent:
            print(f"  epoch {ep+1:02d}  train_loss={tot/len(items):.4f}")
    model.eval()
    return model
