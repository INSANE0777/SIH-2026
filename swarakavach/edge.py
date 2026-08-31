"""Edge runtime — torch-free Detector A for on-device deployment.

Bundle (models/): artifact_edge.onnx (416 KB) + lfcc_consts.npz (132 KB)
= 0.55 MB total, vs the < 10 MB target. No torch/torchaudio import anywhere:
numpy STFT/filterbank (exact torchaudio parity, see features_np.py) +
onnxruntime. The wav2vec2/speaker/ASR layers stay server-side by design —
edge deployments run layer A on the device and defer the rest.

Usage:
    from swarakavach.edge import EdgeArtifact
    edge = EdgeArtifact()                      # loads onnx + consts
    p_spoof = edge.score(pcm_float32_16k)     # 0..1, ~1-2 ms/chunk on CPU
"""
from pathlib import Path

import numpy as np

from . import config
from .features_np import lfcc_numpy


class EdgeArtifact:
    def __init__(self, onnx_path: str | None = None):
        import onnxruntime as ort

        p = Path(onnx_path or config.MODELS_DIR / "artifact_edge.onnx")
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing — run scripts/export_onnx.py to export the edge model")
        self.sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])

    def score(self, wav: np.ndarray, sr: int = 16000) -> float:
        """Spoof probability for one audio chunk (float mono, 16 kHz)."""
        feats = lfcc_numpy(wav, sr)
        feats = ((feats - feats.mean(axis=1, keepdims=True))
                 / (feats.std(axis=1, keepdims=True) + 1e-6))
        lg = self.sess.run(None, {"lfcc": feats[None].astype(np.float32)})[0][0]
        e = np.exp(lg - lg.max())
        return float(e[1] / e.sum())


def bundle_size_mb() -> float:
    total = sum((config.MODELS_DIR / f).stat().st_size
                for f in ("artifact_edge.onnx", "lfcc_consts.npz")
                if (config.MODELS_DIR / f).exists())
    return total / 1e6
