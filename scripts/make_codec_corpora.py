"""Materialize codec-augmented copies of all training corpora (week-3).

Pre-encoding through real ffmpeg AMR-NB/Opus pipelines ONCE, then training
from files, is 100x cheaper than per-epoch subprocess roundtrips and gives
deterministic corpora. Codec copies share speaker groups (speaker_key strips
codec prefixes) so speaker-disjoint splits stay leak-free.

Usage: python scripts/make_codec_corpora.py
Output: data/<corpus>/codec/<level>_<name>.wav + labels.csv per corpus
"""
import csv
from collections import Counter
from pathlib import Path

from swarakavach import config
from swarakavach.audio import load_wav, save_wav
from swarakavach.codecs import degrade_codec

CORPORA = {
    "demo": config.DATA_DIR / "demo" / "labels.csv",
    "clone": config.DATA_DIR / "clone-corpus" / "chunks" / "labels.csv",
    "libri": config.DATA_DIR / "libri-human" / "chunks" / "labels.csv",
}
LEVELS = ("amr", "opus")  # the two real codecs; numpy noise/PL stay in-dataloader


def main():
    for name, labels in CORPORA.items():
        if not labels.exists():
            print(f"skip {name}: {labels} missing")
            continue
        out = labels.parent / "codec"
        out.mkdir(exist_ok=True)
        rows = []
        with open(labels) as f:
            src = list(csv.DictReader(f))
        for r in src:
            wav = load_wav(r["path"])
            for lvl in LEVELS:
                p = out / f"{lvl}_{Path(r['path']).stem}.wav"
                if not p.exists():
                    save_wav(p, degrade_codec(wav, lvl), 16000)
                rows.append([str(p), r.get("family", ""), int(r["label"])])
        with open(out / "labels.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "family", "label"])
            w.writerows(rows)
        print(f"{name}: {len(rows)} codec chunks ({len(src)} src x {len(LEVELS)}) -> {out}")


if __name__ == "__main__":
    main()
