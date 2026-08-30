"""Prepare LibriSpeech test-clean (40 speakers) as bonafide training data.

Downloads (once) and chunks to 2 s / 1 s-hop windows, named li<speaker>_NN.wav
so train_utils.speaker_key groups by speaker for speaker-disjoint validation.

Real-speaker diversity is the cure for 'unseen human = spoof' false positives:
6 hand-curated speakers cannot teach what natural human speech is; 40 can.

Usage: python scripts/prepare_libri.py [--src C:/SIH/LibriSpeech/test-clean]
Output: data/libri-human/chunks/*.wav + labels.csv
License: LibriSpeech is CC BY 4.0 (public-domain audiobook readings).
"""
import argparse
import csv
import tarfile
import urllib.request
from pathlib import Path

from swarakavach import config
from swarakavach.audio import iter_chunks, load_wav, save_wav

URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="C:/SIH/LibriSpeech/test-clean",
                    help="extracted test-clean dir (downloaded automatically if missing)")
    ap.add_argument("--per-speaker", type=int, default=4,
                    help="utterances per speaker (80 utts -> ~400 chunks)")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"downloading {URL} (~347 MB) ...")
        tgz = src.parent / "libri-test-clean.tar.gz"
        urllib.request.urlretrieve(URL, tgz)
        print("extracting ...")
        with tarfile.open(tgz) as t:
            t.extractall(src.parent.parent)
        tgz.unlink()

    out = config.DATA_DIR / "libri-human" / "chunks"
    out.mkdir(parents=True, exist_ok=True)

    n_clips, n_utts = 0, 0
    rows = []
    for spk_dir in sorted(src.iterdir()):
        if not spk_dir.is_dir():
            continue
        flacs = sorted(src.glob(f"{spk_dir.name}/*/*.flac"))[: args.per_speaker]
        for f in flacs:
            wav = load_wav(f)
            for i, ch in enumerate(iter_chunks(wav, chunk=config.TRAIN_CHUNK_SAMPLES,
                                               overlap=config.TRAIN_OVERLAP_SAMPLES)):
                if len(ch) < int(0.75 * config.SAMPLE_RATE):
                    continue
                p = out / f"li{spk_dir.name}_{i:02d}.wav"
                if not p.exists():
                    save_wav(p, ch, config.SAMPLE_RATE)
                rows.append([str(p), "human", 0])
                n_clips += 1
            n_utts += 1

    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "family", "label"])
        w.writerows(rows)
    print(f"{n_utts} utterances -> {n_clips} chunks written to {out}")


if __name__ == "__main__":
    main()
