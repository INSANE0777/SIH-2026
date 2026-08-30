"""Layer C — speaker consistency vs enrolled voiceprint.

Enrollment stores a compact voiceprint (prosody + timbre stats). At call time
the live chunk's stats are compared by cosine similarity; large deviation from
the enrolled caller raises the impersonation contribution. Without enrollment
the layer abstains (None) — fusion renormalizes.
"""
import numpy as np

from .. import config
from ..features import speaker_stats

ENROLL_DIR = config.DATA_DIR / "enrollments"


def _path(speaker_id: str):
    safe = "".join(c for c in speaker_id if c.isalnum() or c in "-_") or "default"
    return ENROLL_DIR / f"{safe}.npz"


def enroll(speaker_id: str, wav: np.ndarray, sr: int = config.SAMPLE_RATE) -> dict:
    ENROLL_DIR.mkdir(parents=True, exist_ok=True)
    v = speaker_stats(wav)
    np.savez(_path(speaker_id), stats=v, sr=sr)
    return {"speaker_id": speaker_id, "dim": len(v)}


def enrolled(speaker_id: str) -> bool:
    return _path(speaker_id).exists()


def score(speaker_id: str, wav: np.ndarray, sr: int = config.SAMPLE_RATE):
    p = _path(speaker_id)
    if not p.exists():
        return None
    from ..audio import frame_rms
    # near-silent chunks (pauses, hold music gaps) carry no speaker evidence —
    # scoring them would read every pause as full impersonation mismatch
    rms = frame_rms(np.asarray(wav, dtype=np.float32), sr)
    if float(np.median(rms)) < 10 ** (-40 / 20):  # quieter than -40 dBFS
        return None
    ref = np.load(p)["stats"]
    cur = speaker_stats(wav)
    cos = float(np.dot(ref, cur) / ((np.linalg.norm(ref) * np.linalg.norm(cur)) + 1e-9))
    # ponytail: linear cosine→risk mapping; recalibrate SPEAKER_* on real data
    risk = np.clip((config.SPEAKER_COS_REF - cos) / config.SPEAKER_COS_SPAN, 0.0, 1.0)
    return float(risk)
