"""Layer A — spectral artifact detector: LFCC + 1D CNN.

The CNN reads the full LFCC spectrogram of a chunk and outputs a spoof
probability. Small enough for edge deployment (~60k params, ~240 KB fp32).
"""
import numpy as np
import torch
import torch.nn as nn

from .. import config
from ..features import lfcc


class ArtifactCNN(nn.Module):
    def __init__(self, in_ch: int = config.LFCC_COEFFS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 5, padding=2, stride=2), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):  # x: (B, n_lfcc, T)
        return self.net(x)


_model, _loaded = None, False


def _load():
    global _model, _loaded
    if _loaded:
        return _model
    _loaded = True
    if not config.ARTIFACT_CKPT.exists():
        return None
    ck = torch.load(config.ARTIFACT_CKPT, map_location="cpu", weights_only=True)
    m = ArtifactCNN()
    m.load_state_dict(ck["state_dict"])
    m.eval()
    _model = m
    return _model


def available() -> bool:
    return _load() is not None


def score(wav: np.ndarray, sr: int = config.SAMPLE_RATE):
    """Spoof probability from artifact CNN; None if no trained checkpoint."""
    m = _load()
    if m is None:
        return None
    feats = lfcc(wav, sr)
    feats = (feats - feats.mean(axis=1, keepdims=True)) / (feats.std(axis=1, keepdims=True) + 1e-6)
    x = torch.from_numpy(feats).float().unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(m(x), dim=-1)[0]
    return float(p[1])
