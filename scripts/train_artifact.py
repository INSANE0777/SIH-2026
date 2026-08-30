"""Train Detector A (artifact CNN) on a labeled corpus.

Default: synthetic bootstrap corpus (data/demo/labels.csv).
Real:    ASVspoof2019 LA — pass --asvspoof-dir <root> containing
         flac/ (audio) and LA/ASVspoof2019_LA_cm_protocols files.
Clone    corpus: pass --labels data/clone-corpus/manifest-derived csv
         (see scripts/eval_loto.py chunking).

The validation split is SPEAKER-DISJOINT: whole speaker groups (human
speakers and TTS voices) are held out, so val accuracy measures
generalization to unseen speakers, not memorization of seen ones.
Saves:   models/artifact_bootstrap.pt  {state_dict, n_lfcc, val_acc}
"""
import argparse
import csv
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from swarakavach import config
from swarakavach.audio import load_wav
from swarakavach.detectors.artifact import ArtifactCNN
from swarakavach.train_utils import LFCCDataset, collate, speaker_key, per_speaker_balance


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(config.DATA_DIR / "demo" / "labels.csv"))
    ap.add_argument("--extra-labels", default=None,
                    help="extra labeled csv(s), comma-separated — e.g. the clone "
                         "corpus manifest (path,family,label) for family diversity")
    ap.add_argument("--asvspoof-dir", default=None, help="ASVspoof2019 LA root (overrides --labels)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default=str(config.ARTIFACT_CKPT))
    args = ap.parse_args()

    if args.asvspoof_dir:
        items = asvspoof_items(Path(args.asvspoof_dir))
    else:
        with open(args.labels) as f:
            items = [(r["path"], int(r["label"])) for r in csv.DictReader(f)]
        if args.extra_labels:
            for csv_path in args.extra_labels.split(","):
                with open(csv_path.strip()) as f:
                    for r in csv.DictReader(f):
                        items.append((r["path"], int(r["label"])))
        # dedupe paths (clone corpus human rows overlap demo corpus reals)
        seen = set()
        dedup = []
        for p, l in items:
            if p not in seen:
                seen.add(p)
                dedup.append((p, l))
        items = dedup
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

    # per-speaker class balancing (see train_utils.per_speaker_balance) +
    # global class weighting — unbalanced clone corpora make the CNN
    # trigger-happy on the genuine speaker's own voice
    train = per_speaker_balance(train)
    random.shuffle(train)
    n_pos = sum(1 for _, l in train if l == 1)
    n_neg = len(train) - n_pos

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
