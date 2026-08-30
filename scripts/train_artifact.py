"""Train Detector A (artifact CNN) on a labeled corpus.

Default: synthetic bootstrap corpus (data/demo/labels.csv).
Real:    ASVspoof2019 LA — pass --asvspoof-dir <root> containing
         flac/ (audio) and LA/ASVspoof2019_LA_cm_protocols files.

The validation split is SPEAKER-DISJOINT: whole speaker groups (human
speakers and TTS voices) are held out, so val accuracy measures
generalization to unseen speakers, not memorization of seen ones.
Saves:   models/artifact_bootstrap.pt  {state_dict, n_lfcc, val_acc}
"""
import argparse
import csv
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from swarakavach import config
from swarakavach.audio import load_wav
from swarakavach.detectors.artifact import ArtifactCNN
from swarakavach.features import lfcc


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


def speaker_key(path: str) -> str:
    """Group key for speaker-disjoint splits: strip prefixes and chunk index(es).
    real/jfk_03.wav -> jfk · attack/voc_jfk_03.wav -> jfk · edge_neerja_02 -> neerja
    voc_sb_spk1_snt1.wav and sb_spk1_snt1_00.wav -> sb_spk1_snt (same speaker)
    """
    stem = re.sub(r"^(voc_|edge_)", "", Path(path).stem)
    return re.sub(r"(?:_\d+)+$", "", stem)


def asvspoof_items(root: Path):
    """ASVspoof2019 LA: bonafide/spoof labels from CM protocol."""
    prot = next(root.glob("**/ASVspoof2019.LA.cm.train.trn.txt"), None)
    if prot is None:
        raise SystemExit(f"no LA cm train protocol under {root}")
    flac_root = next(root.glob("**/flac"), None) or root
    items = []
    for line in prot.read_text().splitlines():
        parts = line.split()
        utt, label = parts[1], 1 if parts[-1] == "spoof" else 0
        f = flac_root / f"{utt}.flac"
        if f.exists():
            items.append((str(f), label))
    return items


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(config.DATA_DIR / "demo" / "labels.csv"))
    ap.add_argument("--asvspoof-dir", default=None, help="ASVspoof2019 LA root (overrides --labels)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default=str(config.ARTIFACT_CKPT))
    args = ap.parse_args()

    if args.asvspoof_dir:
        items = asvspoof_items(Path(args.asvspoof_dir))
    else:
        with open(args.labels) as f:
            items = [(r["path"], int(r["label"])) for r in csv.DictReader(f)]
    random.seed(0)
    # SPEAKER-DISJOINT split: hold out entire speaker/voice groups until val
    # holds >= 20% of rows — the honest generalization number on this corpus.
    groups = {}
    for it in items:
        groups.setdefault(speaker_key(it[0]), []).append(it)
    val, train = [], list(items)
    target = 0.20 * len(items)
    for key in sorted(groups):
        if len(val) >= target:
            break
        val.extend(groups[key])
        train = [it for it in train if speaker_key(it[0]) != key]
    random.shuffle(train)
    print(f"train={len(train)} val={len(val)} "
          f"(held-out speakers: {sorted({speaker_key(it[0]) for it in val})})")

    # oversample the minority class (real) to match attacks — the corpus is
    # ~1:3 real:attack and an unbalanced CNN gets trigger-happy on genuine calls
    n_pos = sum(1 for _, l in train if l == 1)
    n_neg = len(train) - n_pos
    if n_neg and n_pos / n_neg > 1.5:
        mult = n_pos // n_neg
        train = train + [it for it in train if it[1] == 0] * mult
        random.shuffle(train)
        print(f"oversampled real x{mult + 1} -> train={len(train)}")

    dl = DataLoader(LFCCDataset(train, augment=True), batch_size=16, shuffle=True, collate_fn=collate)
    vdl = DataLoader(LFCCDataset(val), batch_size=16, collate_fn=collate)

    model = ArtifactCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    w = torch.tensor([len(train) / (2 * max(n_neg, 1)), len(train) / (2 * max(n_pos, 1))])
    lossf = nn.CrossEntropyLoss(weight=w)

    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in dl:
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        model.eval()
        correct = n = 0
        cls = {0: [0, 0], 1: [0, 0]}  # label -> [correct, total]
        with torch.no_grad():
            for x, y in vdl:
                pred = model(x).argmax(-1)
                correct += int((pred == y).sum())
                n += len(y)
                for p, t in zip(pred.tolist(), y.tolist()):
                    cls[t][0] += int(p == t)
                    cls[t][1] += 1
        bon = cls[0][0] / max(cls[0][1], 1)
        spf = cls[1][0] / max(cls[1][1], 1)
        print(f"epoch {ep+1:02d}  train_loss={tot/len(train):.4f}  val_acc={correct/n:.4f} "
              f"(bonafide={bon:.2f} spoof={spf:.2f})")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_lfcc": config.LFCC_COEFFS,
                "val_acc": correct / n}, args.out)
    print(f"saved {args.out} (val_acc={correct/n:.4f})")


if __name__ == "__main__":
    main()
