"""Torch-free numerics for the edge path — EXACT torchaudio LFCC parity.

torchaudio LFCC = Spectrogram(power=2.0, n_fft=512, win=400, hop=160, center)
-> matmul(filter_mat) -> AmplitudeToDB('power', top_db=80) -> matmul(dct_mat).

The deterministic matrices (filterbank 257x60, DCT 60x20) are exported once
from torchaudio into models/lfcc_consts.npz; the STFT + dB conversion are
reimplemented in numpy with bit-matching conventions. Parity is asserted in
scripts/export_onnx.py and tests.
"""
import numpy as np

_CONSTS = None


def _consts():
    global _CONSTS
    if _CONSTS is None:
        from pathlib import Path

        from . import config

        p = config.MODELS_DIR / "lfcc_consts.npz"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing — run scripts/export_onnx.py once (it dumps the "
                "torchaudio filterbank/DCT matrices for the torch-free path)")
        z = np.load(p)
        _CONSTS = {k: z[k] for k in z.files}
    return _CONSTS


def lfcc_numpy(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """(n_lfcc=20, T) — exact-parity torch-free LFCC."""
    c = _consts()
    assert sr == int(c["sr"][0]), "edge constants are for 16 kHz audio"
    n_fft, win, hop = int(c["n_fft"][0]), int(c["win"][0]), int(c["hop"][0])

    # --- STFT with center=True: REFLECT-pad n_fft//2 both sides (torch
    # default), window CENTERED in the n_fft frame, hop strides over the PADDED
    # signal (verified empirically: maxdiff vs torchaudio ~4e-4 float noise)
    x = np.asarray(wav, dtype=np.float64).ravel()
    reflect = n_fft // 2
    if len(x) > reflect + 1:
        xp = np.concatenate([x[reflect:0:-1], x, x[-2 : -2 - reflect : -1]])
    else:  # very short input: zero-pad (edge case)
        xp = np.concatenate([np.zeros(reflect), x, np.zeros(reflect)])
    if len(xp) < n_fft:
        xp = np.concatenate([xp, np.zeros(n_fft - len(xp))])

    w = np.hanning(win + 1)[:win]          # periodic hanning, length win=400
    w_pad = np.zeros(n_fft)
    w_pad[n_fft // 2 - win // 2: n_fft // 2 - win // 2 + win] = w
    n_frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = xp[idx] * w_pad[None, :]
    spec = np.abs(np.fft.rfft(frames, n_fft, axis=1)) ** 2.0        # power=2

    # --- filterbank energies
    fb = spec @ c["filter_mat"]                                     # (T, 60)

    # --- AmplitudeToDB('power', top_db=80): 10*log10, clamped to the GLOBAL
    # max across the whole spectrogram (torchaudio clamps to x_db.max() - top_db,
    # NOT per frame — per-frame clamping was the parity bug)
    db = 10.0 * np.log10(np.clip(fb, a_min=1e-10, a_max=None))
    top = float(c["top_db"][0])
    db = np.maximum(db, db.max() - top)

    # --- DCT-II ortho via exported matrix: (T, 60) @ (60, 20)
    lfcc = db @ c["dct_mat"]
    return lfcc.T.astype(np.float32)                                # (20, T)
