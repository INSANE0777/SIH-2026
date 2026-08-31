"""Experiment: fine-tune wav2vec2 (layer B) on OUR clone corpus.

Replaces the borrowed Hemgg checkpoint (0.51 EER — chance-level on modern
neural TTS) with our own: frozen wav2vec2-base encoder + trained classification
head, fine-tuned on the 4-family corpus (edge/piper/sapi/xtts + real humans
incl. Hindi). Validated SPEAKER-DISJOINT (humans AND attack voices held out).

Usage: python scripts/experiments/finetune_ssl.py [--epochs 3]
Output: models/ssl_finetuned/ (HF-format save) + val metrics
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from swarakavach import config
from swarakavach.audio import load_wav
from swarakavach.train_utils import speaker_key

MAX_SEC = 4.0  # wav2vec2 input cap per sample (memory)


class ChunkDS(Dataset):
    """(wav, label) dataset; caches decoded audio in RAM (corpus fits)."""

    def __init__(self, items, limit_sec=MAX_SEC):
        self.items = items
        self.n = int(limit_sec * config.SAMPLE_RATE)
        self.cache = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if i not in self.cache:
            p, _ = self.items[i]
            w = load_wav(p)
            if len(w) < self.n:
                w = np.pad(w, (0, self.n - len(w)))
            else:
                w = w[: self.n]
            self.cache[i] = w.astype(np.float32)
        return self.cache[i], torch.tensor(self.items[i][1])


def collate_wav(batch):
    xs = torch.stack([torch.from_numpy(x) for x, _ in batch])
    ys = torch.stack([y for _, y in batch])
    return xs, ys


def load_all_corpora():
    items = []
    for csv_path in [
        config.DATA_DIR / "demo" / "labels.csv",
        config.DATA_DIR / "clone-corpus" / "chunks" / "labels.csv",
        config.DATA_DIR / "libri-human" / "chunks" / "labels.csv",
        config.DATA_DIR / "hindi-human" / "chunks" / "labels.csv",
    ]:
        if csv_path.exists():
            with open(csv_path) as f:
                for r in csv.DictReader(f):
                    items.append((r["path"], int(r["label"])))
    return items


def speaker_disjoint_split(items, hold_frac=0.2):
    groups = {}
    for it in items:
        groups.setdefault(speaker_key(it[0]), []).append(it)
    keys = sorted(groups)
    val, train = [], list(items)
    target = hold_frac * len(items)
    for k in keys:
        if len(val) >= target:
            break
        val.extend(groups[k])
        train = [it for it in train if speaker_key(it[0]) != k]
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="models/ssl_finetuned")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    base = "facebook/wav2vec2-base"
    fx = AutoFeatureExtractor.from_pretrained(base)

    items = load_all_corpora()
    # balance: attacks ~2600, humans ~5700 -> cap humans to attack count * 1.2
    att = [it for it in items if it[1] == 1]
    hum = [it for it in items if it[1] == 0]
    rng = np.random.default_rng(0)
    hum_idx = rng.permutation(len(hum))[: int(1.2 * len(att))]
    items = att + [hum[i] for i in hum_idx]
    train, val = speaker_disjoint_split(items)
    print(f"train={len(train)} val={len(val)} (speaker-disjoint, balanced)")

    # build model: wav2vec2 encoder + head; UNFREEZE top 2 transformer layers
    # (the ASVspoof recipe: frozen lower layers, tuned top — head-only plateaus
    # at 0.55 val with bonafide 0.35, same failure mode as the borrowed ckpt)
    model = AutoModelForAudioClassification.from_pretrained(base, num_labels=2,
                                                             ignore_mismatched_sizes=True)
    enc = model.wav2vec2
    n_layers = len(enc.encoder.layers)
    for i, layer in enumerate(enc.encoder.layers):
        for p in layer.parameters():
            p.requires_grad = i >= n_layers - 2
    for p in enc.encoder.layers[n_layers - 2:].parameters():
        p.requires_grad = True
    model.to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train/1e6:.1f}M (top-2 layers + head)")

    # head config: mean-pool + linear (the HF default head) trains fine
    tds, vds = ChunkDS(train), ChunkDS(val)
    dl = DataLoader(tds, batch_size=args.batch, shuffle=True, collate_fn=collate_wav)
    vdl = DataLoader(vds, batch_size=args.batch, collate_fn=collate_wav)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5,
                            weight_decay=0.01)
    lossf = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        model.train()
        tot = n = 0
        for x, y in dl:
            inp = fx(x.numpy(), sampling_rate=config.SAMPLE_RATE, return_tensors="pt",
                     padding=True)
            inp = {k: v.to(device) for k, v in inp.items()}
            y = y.to(device)
            opt.zero_grad()
            loss = lossf(model(**inp).logits, y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        # val
        model.eval()
        correct = nv = 0
        cls = {0: [0, 0], 1: [0, 0]}
        with torch.no_grad():
            for x, y in vdl:
                inp = fx(x.numpy(), sampling_rate=config.SAMPLE_RATE, return_tensors="pt",
                         padding=True)
                inp = {k: v.to(device) for k, v in inp.items()}
                pred = model(**inp).logits.argmax(-1).cpu()
                for p_, t_ in zip(pred.tolist(), y.tolist()):
                    cls[t_][0] += int(p_ == t_)
                    cls[t_][1] += 1
                correct += int((pred == y).sum())
                nv += len(y)
        bon = cls[0][0] / max(cls[0][1], 1)
        spf = cls[1][0] / max(cls[1][1], 1)
        print(f"epoch {ep+1}: train_loss={tot/max(n,1):.4f} val_acc={correct/max(nv,1):.4f} "
              f"(bonafide={bon:.2f} spoof={spf:.2f})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    fx.save_pretrained(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
