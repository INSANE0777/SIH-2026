"""Real telephony codec simulation via ffmpeg (week-3 roadmap item).

Actual AMR-NB (cellular) and Opus (VoIP) encode/decode roundtrips — not
synthetic approximations — plus the numpy channel effects already in audio.py.

This module has two jobs:
1. EVAL:    score any audio through codec pipelines (what the phone network
            actually does to a call) — used by eval_telephony.
2. TRAIN:   codec-augment the training corpus so the detector learns codec
            artifacts instead of memorizing clean-audio artifacts.

Levels (composed in degrade_codec):
  clean        full band, no codec
  amr          8 kHz AMR-NB @ 4.75 kbps — worst cellular rate
  amr_wb       16 kHz AMR-WB @ 8 kbps (mid cellular)
  opus         Opus 8 kbps VoIP
  pl2          opus + 2% 20 ms packet loss
  evs_like     opus 16 kHz mono low-rate (EVS-style profile; real EVS encoders
               are not in ffmpeg — documented approximation)

Requires: ffmpeg on PATH (winget/pip install). Falls back with a clear error.
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .audio import add_noise, load_wav, packet_loss, save_wav

FFmpeg = "ffmpeg"

# (name, ffmpeg args after input, muxer format for the encoded file, output sr)
# NB: this ffmpeg build ships libopencore_amrnb + libopus only (no AMR-WB —
# opus at 8k covers the wideband VoIP profile; documented approximation).
_PIPELINES = {
    "amr":      (["-ar", "8000",  "-ac", "1", "-c:a", "libopencore_amrnb", "-b:a", "4.75k"], "amr", 16000),
    "opus":     (["-ar", "16000", "-ac", "1", "-c:a", "libopus", "-b:a", "8k"], "ogg", 16000),
    "opus_wb":  (["-ar", "16000", "-ac", "1", "-c:a", "libopus", "-b:a", "16k"], "ogg", 16000),
}


def _run_ffmpeg(args: list[str]) -> None:
    r = subprocess.run([FFmpeg, "-y", "-hide_banner", "-loglevel", "error", *args],
                       capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.decode(errors='ignore')[:200]}")


def codec_roundtrip(wav: np.ndarray, sr: int, pipeline: str) -> np.ndarray:
    """Encode+decode through a real codec. Writes temp files (ffmpeg pipe)."""
    if pipeline not in _PIPELINES:
        raise ValueError(f"unknown codec pipeline: {pipeline}")
    enc_args, enc_fmt, out_sr = _PIPELINES[pipeline]
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.wav"
        enc = Path(td) / f"enc.{enc_fmt}"
        dst = Path(td) / "out.wav"
        save_wav(src, wav, sr)
        _run_ffmpeg(["-i", str(src), *enc_args, str(enc)])
        _run_ffmpeg(["-i", str(enc), "-ar", str(out_sr), "-ac", "1", str(dst)])
        return load_wav(dst, out_sr)


def degrade_codec(wav: np.ndarray, level: str, sr: int = 16000) -> np.ndarray:
    """Named channel/codec degradation levels (superset of audio.degrade)."""
    if level == "clean":
        return np.asarray(wav, dtype=np.float32)
    if level == "amr":
        return codec_roundtrip(wav, sr, "amr")
    if level == "opus":
        return codec_roundtrip(wav, sr, "opus")
    if level == "opus_wb":
        return codec_roundtrip(wav, sr, "opus_wb")
    if level == "pl2":
        return packet_loss(codec_roundtrip(wav, sr, "opus"), frac=0.02)
    if level == "evs_like":
        # EVS-style profile: AMR-NB narrowband + mild noise — approximation, documented
        return add_noise(codec_roundtrip(wav, sr, "amr"), snr_db=30.0)
    raise ValueError(f"unknown degradation level: {level}")


CODEC_LEVELS = ["clean", "amr", "opus", "opus_wb", "pl2", "evs_like"]


def augment_random(wav: np.ndarray, sr: int = 16000, rng=None) -> np.ndarray:
    """Train-time codec augmentation: 50% clean, else a random codec pipeline.
    Cheap enough per-epoch (ffmpeg subprocess ~50 ms/call) for small corpora;
    for large corpora pre-materialize with materialize_codec_corpus()."""
    rng = rng or np.random.default_rng()
    if rng.random() < 0.5:
        return np.asarray(wav, dtype=np.float32)
    lvl = rng.choice(["amr", "opus", "opus_wb"])
    try:
        return degrade_codec(wav, lvl, sr)
    except Exception:
        return np.asarray(wav, dtype=np.float32)


def materialize_codec_corpus(labels_csv: Path, out_dir: Path, levels=("amr", "opus")) -> Path:
    """Pre-encode every item of a labels.csv through codec pipelines; writes
    <out_dir>/<level>_<name>.wav + labels.csv (path,family,label preserved)."""
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(labels_csv) as f:
        for r in csv.DictReader(f):
            wav = load_wav(r["path"])
            for lvl in levels:
                p = out_dir / f"{lvl}_{Path(r['path']).stem}.wav"
                if not p.exists():
                    save_wav(p, degrade_codec(wav, lvl), 16000)
                rows.append([str(p), r.get("family", ""), int(r["label"])])
    labels = out_dir / "labels.csv"
    with open(labels, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "family", "label"])
        w.writerows(rows)
    return labels
