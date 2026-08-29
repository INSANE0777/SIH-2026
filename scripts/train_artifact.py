"""Train Detector A (artifact CNN) on a labeled corpus.

Default: synthetic bootstrap corpus (data/demo/labels.csv).
Real:    ASVspoof2019 LA — pass --asvspoof-dir <root> containing
         flac/ (audio) and LA/ASVspoof2019_LA_cm_protocols files.
Saves:   models/artifact_bootstrap.pt  {state_dict, n_lfcc, val_acc}
"""
import argparse
import csv
import random
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
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        wav = load_wav(path)
        f = lfcc(wav)
        f = (f - f.mean(axis=1, keepdims=True)) / (f.std(axis=1, keepdims=True) + 1e-6)
        return torch.from_numpy(f).float(), torch.tensor(label)


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
    random.shuffle(items)
    # stratified split (tiny bootstrap corpora need both classes in val)
    pos = [it for it in items if it[1] == 1]
    neg = [it for it in items if it[1] == 0]
    n_pos_val = max(1, int(0.2 * len(pos)))
    n_neg_val = max(1, int(0.2 * len(neg)))
    val = pos[:n_pos_val] + neg[:n_neg_val]
    train = pos[n_pos_val:] + neg[n_neg_val:]
    random.shuffle(train)
    print(f"train={len(train)} val={len(val)}")

    dl = DataLoader(LFCCDataset(train), batch_size=16, shuffle=True, collate_fn=collate)
    vdl = DataLoader(LFCCDataset(val), batch_size=16, collate_fn=collate)

    model = ArtifactCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_pos = sum(1 for _, l in train if l == 1)
    n_neg = len(train) - n_pos
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
        with torch.no_grad():
            for x, y in vdl:
                pred = model(x).argmax(-1)
                correct += int((pred == y).sum())
                n += len(y)
        print(f"epoch {ep+1:02d}  train_loss={tot/len(train):.4f}  val_acc={correct/n:.4f}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_lfcc": config.LFCC_COEFFS,
                "val_acc": correct / n}, args.out)
    print(f"saved {args.out} (val_acc={correct/n:.4f})")


if __name__ == "__main__":
    main()
