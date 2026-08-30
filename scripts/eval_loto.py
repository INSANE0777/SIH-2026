"""Leave-one-generator-out (LOTO) evaluation — the generalization claim.

The question ASVspoof-style benchmarks never answer: does the detector catch a
generator it has NEVER seen in training?

For each attack family F in {edge, piper, sapi, xtts}:
  train layer A on [human bonafide + all attack families except F]
  evaluate  EER on [human bonafide + F only]

The per-family EER table is the honest cross-generator generalization number.
In-distribution rows (train on all families incl. F) are reported alongside
for contrast — the gap between in-distribution and held-out IS the finding.

Usage: python scripts/eval_loto.py [--manifest data/clone-corpus/manifest.csv]
                                   [--epochs 30]
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from swarakavach import config
from swarakavach.audio import iter_chunks, load_wav, save_wav
from swarakavach.features import lfcc
from swarakavach.train_utils import train_cnn, eer


def chunked(rows, cache: Path):
    """Chunk every wav to 2 s windows (1 s hop); returns [(path, label, family)]."""
    items = []
    for path, family, label in rows:
        wav = load_wav(path)
        base = cache / f"{family}_{Path(path).stem}"
        for i, ch in enumerate(iter_chunks(wav, chunk=config.TRAIN_CHUNK_SAMPLES,
                                           overlap=config.TRAIN_OVERLAP_SAMPLES)):
            if len(ch) < int(0.75 * config.SAMPLE_RATE):
                continue
            p = base.parent / f"{base.name}_{i:02d}.wav"
            if not p.exists():
                save_wav(p, ch, config.SAMPLE_RATE)
            items.append((str(p), label, family))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(config.DATA_DIR / "clone-corpus" / "manifest.csv"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--codec-augment", action="store_true",
                    help="train folds with real codec augmentation (needs ffmpeg)")
    ap.add_argument("--cache", default=str(config.DATA_DIR / "clone-corpus" / "chunks"))
    args = ap.parse_args()

    with open(args.manifest) as f:
        rows = [(r["path"], r["family"], int(r["label"])) for r in csv.DictReader(f)]

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    items = chunked(rows, cache)

    families = sorted({fam for _, _, fam in items if fam != "human"})
    human = [it for it in items if it[2] == "human"]
    print(f"items: {len(items)}  human={len(human)}  families={families}")
    print(f"{'held-out family':<18} {'train n':>8} {'test n':>8} {'EER':>8}")

    results = {}
    for hold in families:
        train = [it[:2] for it in items if it[2] != hold and it[2] != "human"] + [it[:2] for it in human]
        test = [it[:2] for it in items if it[2] == hold] + [it[:2] for it in human]
        model = train_cnn(train, epochs=args.epochs, silent=True,
                          codec_augment=args.codec_augment)
        scores, ys = [], []
        model.eval()
        with torch.no_grad():
            for path, label in test:
                wav = load_wav(path)
                feats = lfcc(wav)
                feats = ((feats - feats.mean(axis=1, keepdims=True))
                         / (feats.std(axis=1, keepdims=True) + 1e-6))
                x = torch.from_numpy(feats).float().unsqueeze(0)
                scores.append(float(torch.softmax(model(x), -1)[0][1]))
                ys.append(label)
        e = eer(np.array(scores), np.array(ys))
        results[hold] = e
        print(f"{hold:<18} {len(train):>8} {len(test):>8} {e:>8.4f}")

    mean_eer = float(np.mean(list(results.values())))
    print(f"\nmean held-out EER: {mean_eer:.4f}")
    print("in-distribution (train on all, test on same) is upper-bound flattery;")
    print("the held-out column above is the honest cross-generator number.")


if __name__ == "__main__":
    main()
