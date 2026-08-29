"""Feature extraction: LFCC spectrograms (Detector A input) and prosody statistics.

Prosody features are the hand-crafted complement to learned features — neural TTS
is characteristically *too regular*: low jitter, over-smoothed energy, few natural
pauses. These six statistics capture that signature.
"""
import numpy as np
import torch
import torchaudio

from . import config

_lfcc_tf = None


def lfcc(wav: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """(n_lfcc, T) Linear-Frequency Cepstral Coefficients matrix."""
    global _lfcc_tf
    if _lfcc_tf is None:
        _lfcc_tf = torchaudio.transforms.LFCC(
            sample_rate=sr,
            n_filter=60,
            n_lfcc=config.LFCC_COEFFS,
            speckwargs={"n_fft": 512, "win_length": 400, "hop_length": 160, "center": True},
        )
    y = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        return _lfcc_tf(y).squeeze(0).numpy()


def lfcc_mean(wav: np.ndarray) -> np.ndarray:
    """Utterance-level LFCC mean — compact timbre vector for speaker stats."""
    return lfcc(wav).mean(axis=1)


def f0_track(x: np.ndarray, sr: int, fmin: float = 70.0, fmax: float = 400.0,
             frame_s: float = 0.04, hop_s: float = 0.01):
    """Autocorrelation F0 tracker. Returns (f0_hz array, voiced mask)."""
    frame, hop = int(sr * frame_s), int(sr * hop_s)
    n = max(1, 1 + (len(x) - frame) // hop)
    win = np.hanning(frame)
    lag_min, lag_max = int(sr / fmax), int(sr / fmin)
    f0s, voiced = np.zeros(n), np.zeros(n, dtype=bool)
    for i in range(n):
        seg = x[i * hop : i * hop + frame] * win
        rms = float(np.sqrt(np.mean(seg**2)))
        if rms < 1e-4:
            continue
        spec = np.fft.rfft(seg, 2 * frame)
        ac = np.fft.irfft(np.abs(spec) ** 2)[: lag_max + 1]
        ac /= ac[0] + 1e-12
        seg_lag = ac[lag_min:lag_max]
        k = int(np.argmax(seg_lag)) + lag_min
        conf = ac[k]
        if conf > 0.35:
            f0s[i], voiced[i] = sr / k, True
    return f0s, voiced


def prosody_vector(wav: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Six prosody statistics: [f0_mean, f0_std, jitter, pause_ratio, energy_dyn, zcr]."""
    from .audio import frame_rms

    x = np.asarray(wav, dtype=np.float32)
    f0s, voiced = f0_track(x, sr)
    v = f0s[voiced]
    f0_mean = float(v.mean()) if len(v) else 0.0
    f0_std = float(v.std()) if len(v) else 0.0
    if len(v) > 2:
        periods = sr / v
        jitter = float(np.mean(np.abs(np.diff(periods))) / (periods.mean() + 1e-9))
    else:
        jitter = 0.0
    rms = frame_rms(x, sr)
    pause_ratio = float(np.mean(rms < 10 ** (-45 / 20)))
    nz = rms[rms > 1e-5]
    energy_dyn = float(np.percentile(nz, 90) / (np.percentile(nz, 10) + 1e-9)) if len(nz) else 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0)) if len(x) > 1 else 0.0
    return np.array([f0_mean, f0_std, jitter, pause_ratio, energy_dyn, zcr], dtype=np.float32)


PROSODY_SCALE = np.array([150.0, 30.0, 0.05, 0.3, 8.0, 0.6], dtype=np.float32)


def speaker_stats(wav: np.ndarray) -> np.ndarray:
    """Normalized voiceprint stats: prosody (6) + timbre LFCC-mean (20)."""
    return np.concatenate([prosody_vector(wav) / PROSODY_SCALE, lfcc_mean(wav)])
