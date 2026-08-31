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


def _lfcc_numpy(wav: np.ndarray, sr: int) -> np.ndarray:
    """Pure-numpy LFCC (torch-free edge path): STFT -> power -> linear filter
    bank -> log -> DCT-II. Matches torchaudio.transforms.LFCC(speckwargs
    n_fft=512, win=400, hop=160, center=True) within float tolerance.
    Filter bank: triangular filters on linear frequency scale (n_filter=60).
    """
    from .features_np import _triangular_linear_fb, _dct2

    x = np.asarray(wav, dtype=np.float64)
    n_fft, win, hop = 512, 400, 160
    pad_l, pad_r = n_fft // 2, n_fft // 2
    xp = np.concatenate([np.zeros(pad_l - (win // 2)), x[: win // 2] if len(x) else x,
                         x, np.zeros(pad_r)])
    # frame the padded signal at hop with win-length windows (zero-padded to n_fft)
    n_frames = 1 + max(0, (len(xp) - n_fft)) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    win_f = np.zeros(n_fft)
    win_f[: win // 2] = np.hanning(win + 2)[1 : win // 2 + 1]
    win_f[-(win // 2):] = np.hanning(win + 2)[win // 2 + 1 : -1]
    frames = xp[idx] * win_f[None, :]
    spec = np.abs(np.fft.rfft(frames, n_fft, axis=1)) ** 2          # (T, 257)
    fb = _triangular_linear_fb(n_filters=60, n_fft=n_fft, sr=sr)  # (60, 257)
    feats = np.log(np.clip(spec @ fb.T, a_min=1e-10, a_max=None) + 1e-10)
    lfcc = _dct2(feats, n_out=config.LFCC_COEFFS)                 # (T, 20)
    return lfcc.T.astype(np.float32)                              # (20, T)


def lfcc(wav: np.ndarray, sr: int = config.SAMPLE_RATE, backend: str = "auto") -> np.ndarray:
    """(n_lfcc, T) LFCC matrix. backend: torch | numpy | auto (torch if
    available, else the numpy edge implementation)."""
    if backend != "numpy":
        global _lfcc_tf
        try:
            import torch  # noqa: F401
        except ImportError:
            backend = "numpy"
        else:
            if _lfcc_tf is None:
                _lfcc_tf = torchaudio.transforms.LFCC(
                    sample_rate=sr,
                    n_filter=60,
                    n_lfcc=config.LFCC_COEFFS,
                    speckwargs={"n_fft": 512, "win_length": 400, "hop_length": 160,
                                "center": True},
                )
            y = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
            with torch.no_grad():
                return _lfcc_tf(y).squeeze(0).numpy()
    return _lfcc_numpy(wav, sr)


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
