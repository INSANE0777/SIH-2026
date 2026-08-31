"""Prepare real Hindi human-speech bonafide data (week-5 tail).

The corpus had Hindi *attacks* (edge-tts hi-IN, piper hi-IN) but ZERO real
Hindi human speech — a judge speaking Hindi live could false-alarm the
detector. This closes that hole.

Source: damerajee/Hindi-audio-speech (HF, open) — multi-speaker multi-domain
real Hindi recordings with audio_id/gender metadata (audio_ids like
hi_m_agri_00025 = language _ gender _ domain _ id).

Downloads N clips via the datasets-server assets API, chunks to 1 s windows
(0.5 s hop, matching training), writes data/hindi-human/chunks + labels.csv
with speaker groups (hi<gender><domain>) for speaker-disjoint splits.

Usage: python scripts/prepare_hindi.py [--clips 40]
"""
import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from swarakavach import config
from swarakavach.audio import iter_chunks, load_wav, save_wav

API = "https://datasets-server.huggingface.co"
DS = "damerajee/Hindi-audio-speech"
UA = {"User-Agent": "Mozilla/5.0"}


def rows_page(offset: int, n: int = 30, retries: int = 3):
    u = (f"{API}/rows?dataset={urllib.parse.quote(DS)}&config=default"
         f"&split=train&offset={offset}&length={n}")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(u, headers=UA), timeout=90) as r:
                return json.load(r)["rows"]
        except Exception:
            if attempt == retries - 1:
                return []
            time.sleep(2)
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=40)
    ap.add_argument("--max-speaker-clips", type=int, default=3,
                    help="diversity guard: clips per audio_id prefix")
    args = ap.parse_args()

    out = config.DATA_DIR / "hindi-human" / "chunks"
    out.mkdir(parents=True, exist_ok=True)

    picked, per_speaker = [], {}
    offset = 0
    while len(picked) < args.clips:
        rows = rows_page(offset)
        if not rows:
            break
        for r in rows:
            row = r["row"]
            aud = row.get("audio")
            if not aud:
                continue
            src = aud[0]["src"] if isinstance(aud, list) else aud["src"]
            aid = row.get("audio_id", f"row{offset}")
            # speaker group: hi_m_agri_00025 -> hi_m_agri (gender+domain bucket
            # is a coarse speaker proxy; the id tail separates recordings)
            grp = "_".join(aid.split("_")[:-1])
            if per_speaker.get(grp, 0) >= args.max_speaker_clips:
                continue
            picked.append((aid, grp, src))
            per_speaker[grp] = per_speaker.get(grp, 0) + 1
            if len(picked) >= args.clips:
                break
        offset += 30

    print(f"picked {len(picked)} clips from {len(per_speaker)} speaker-ish groups")
    n_chunks = 0
    for aid, grp, src in picked:
        raw = out / f"raw_{aid}.wav"
        if not raw.exists():
            try:
                urllib.request.urlretrieve(src, str(raw.with_suffix(".bin")))
                # datasets-server serves raw audio bytes; ffmpeg converts any container
                import subprocess
                subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                                "-i", str(raw.with_suffix(".bin")), "-ar", "16000",
                                "-ac", "1", str(raw)], check=True)
                raw.with_suffix(".bin").unlink(missing_ok=True)
            except Exception as e:
                print(f"  {aid}: fetch failed {str(e)[:60]}")
                continue
        wav = load_wav(raw)
        for i, ch in enumerate(iter_chunks(wav, chunk=config.TRAIN_CHUNK_SAMPLES,
                                           overlap=config.TRAIN_OVERLAP_SAMPLES)):
            if len(ch) < int(0.75 * config.SAMPLE_RATE):
                continue
            p = out / f"ih_{grp}_{i:02d}_{aid}.wav"
            if not p.exists():
                save_wav(p, ch, config.SAMPLE_RATE)
            n_chunks += 1
        raw.unlink(missing_ok=True)

    # labels.csv from the written chunk files (idempotent on re-run)
    rows = []
    for p in sorted(out.glob("ih_*.wav")):
        rows.append([str(p), "human", 0])
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "family", "label"])
        w.writerows(rows)
    print(f"wrote {len(rows)} Hindi bonafide chunks ({len(per_speaker)} speaker groups) "
          f"-> {out}")


if __name__ == "__main__":
    main()
