"""Synthetic bootstrap corpus generator.

Honest scope: this is *bootstrap* data for an immediately-runnable pipeline, NOT
a substitute for real training data. Two generators:

- ``synth_phrase``: source-filter formant speech with natural micro-prosody
  (jitter, shimmer, drift F0, natural pauses) — proxy for *bonafide* speech.
- ``vocode``: mel-spectrogram smoothing + Griffin-Lim resynthesis — the same
  phase-reconstruction / over-smoothing artifact family that neural vocoders
  (the core of voice cloning) leave in audio — proxy for *spoofed* speech.

For real training use ASVspoof5 (Zenodo 14498691) via ``scripts/train_artifact.py
--asvspoof-dir`` and the 2026 clone corpus (F5-TTS, XTTS-v2, RVC, ...) per
docs/architecture.md.
"""
import numpy as np
import torch
import torchaudio
from scipy.signal import lfilter

SR = 16000

# F1, F2, F3, F3-bandwidth for common vowels
VOWELS = {
    "a": (730, 1090, 2440), "i": (270, 2290, 3010), "u": (300, 870, 2240),
    "e": (530, 1840, 2480), "o": (570, 840, 2410),
}


def _resonator(f: float, bw: float, sr: int):
    """Two-pole resonator H(z) = (1-r^2) / (1 - 2 r cos(w) z^-1 + r^2 z^-2)."""
    r = float(np.exp(-np.pi * bw / sr))
    w = 2 * np.pi * f / sr
    return [1.0 - r * r], [1.0, -2.0 * r * np.cos(w), r * r]


def _pulse_train(f0_contour, sr: int, jitter: float = 0.010, shimmer: float = 0.03,
                 rng: np.random.Generator = None):
    """Glottal pulse train with cycle-to-cycle jitter and shimmer."""
    n = len(f0_contour)
    out = np.zeros(n)
    t, amp = 0.0, 1.0
    i = 0
    while t < n:
        out[int(t)] = amp
        period = sr / max(f0_contour[int(min(t, n - 1))], 50.0)
        period *= 1.0 + rng.normal(0, jitter)
        amp = 1.0 + rng.normal(0, shimmer)
        t += max(period, 20.0)
        i += 1
    return out


def synth_phrase(seed: int, dur: float = 3.0, sr: int = SR) -> np.ndarray:
    """Formant-synthesized utterance with natural micro-prosody."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    t = np.arange(n) / sr

    # syllable plan: vowel seq + per-syllable durations, phrase pauses
    syls, pos = [], 0.0
    while pos < dur - 0.25:
        d = rng.uniform(0.12, 0.25)
        gap = rng.uniform(0.02, 0.06)
        if rng.random() < 0.25:            # occasional natural pause
            gap += rng.uniform(0.15, 0.40)
        syls.append((rng.choice(list(VOWELS)), pos, pos + d))
        pos += d + gap
    if not syls:
        syls = [("a", 0.1, 0.4)]

    # drifting F0 contour with slow vibrato-ish drift
    base = rng.uniform(105, 170)
    f0 = base * (1 + 0.08 * np.sin(2 * np.pi * rng.uniform(0.2, 0.9) * t)
                 + 0.02 * rng.standard_normal(n).cumsum() / np.sqrt(n))
    src = _pulse_train(f0, sr, rng=rng)

    # formant filtering per syllable + breath + envelope
    out = np.zeros(n)
    env = np.zeros(n)
    for vow, s, e in syls:
        i0, i1 = int(s * sr), min(int(e * sr), n)
        if i1 <= i0:
            continue
        seg = src[i0:i1]
        sig = np.zeros_like(seg)
        for k, f in enumerate(VOWELS[vow]):
            b, a = _resonator(f + rng.normal(0, 25), rng.uniform(60, 110), sr)
            sig += lfilter(b, a, seg) / (k + 1)
        m = i1 - i0
        att, dec = max(1, int(0.015 * sr)), max(1, int(0.04 * sr))
        env_seg = np.ones(m)
        env_seg[:att] = np.linspace(0, 1, att)
        env_seg[-dec:] *= np.linspace(1, 0, dec)
        out[i0:i1] = sig * env_seg
        env[i0:i1] = env_seg
    out *= 0.8 / (np.abs(out).max() + 1e-9)
    breath = rng.normal(0, 0.003, n) * env
    return (out + breath).astype(np.float32)


_mel_tf = _gl_tf = None


def _voc_tfs(sr: int):
    global _mel_tf, _gl_tf
    if _mel_tf is None:
        _mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=512, win_length=400, hop_length=160,
            n_mels=80, f_min=50, f_max=7600, power=1.0, center=True)
        _gl_tf = torchaudio.transforms.GriffinLim(
            n_fft=512, win_length=400, hop_length=160, power=1.0, n_iter=64)
    return _mel_tf, _gl_tf


def vocode(wav: np.ndarray, sr: int = SR, smooth: int = 5) -> np.ndarray:
    """Mel-smooth + Griffin-Lim resynthesis — vocoder-artifact proxy for clones."""
    mel_tf, gl_tf = _voc_tfs(sr)
    x = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        m = mel_tf(x)                                        # (1, 80, T)
        m = torch.nn.functional.avg_pool1d(m, smooth, 1, smooth // 2)
        m = torch.pow(m.clamp_min(1e-5), 0.75)               # dynamic-range squash
        m = m / (m.max() + 1e-9)
        # stable inverse mel: pseudo-inverse of the filterbank (InverseMelScale's
        # lstsq is rank-deficient for this filterbank)
        fb = getattr(mel_tf, "fb", None)
        if fb is None:  # older/newer torchaudio nest it under mel_scale
            fb = mel_tf.mel_scale.fb
        if fb.shape[0] != 80:  # some versions store (n_stft, n_mels)
            fb = fb.transpose(0, 1)
        pinv = torch.linalg.pinv(fb)                        # (257, 80)
        spec = pinv @ m.squeeze(0)                           # (257, T)
        y = gl_tf(spec.unsqueeze(0))
    y = y / (y.abs().max() + 1e-9)
    return y.squeeze(0).numpy().astype(np.float32) * 0.8
