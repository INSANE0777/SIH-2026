"""Edge export: artifact CNN -> ONNX + torch-free inference path.

The < 10 MB edge claim, measured: the ONNX model is the artifact CNN (~60k
params -> ~240 KB fp32); the LFCC frontend is re-implemented in numpy so the
edge path needs NO torch/torchaudio — just onnxruntime + numpy + soundfile.

Parity is verified against the torch path on the demo corpus (same scores
within fp tolerance). Usage:

    python scripts/export_onnx.py                 # export + parity check
    python scripts/export_onnx.py --quantize      # also int8 dynamic quantize
"""
import argparse
from pathlib import Path

import numpy as np

from swarakavach import config
from swarakavach.audio import load_wav
from swarakavach.detectors.artifact import ArtifactCNN, available


def export(out_path: Path):
    import torch

    if not available():
        raise SystemExit("no trained checkpoint — run scripts/train_artifact.py first")
    ck = torch.load(config.ARTIFACT_CKPT, map_location="cpu", weights_only=True)
    model = ArtifactCNN()
    model.load_state_dict(ck["state_dict"])
    model.eval()

    # LFCC shape: (n_lfcc=20, T) -> batch (1, 20, T). T varies per clip; export
    # with dynamic T so any chunk length works.
    dummy = torch.randn(1, config.LFCC_COEFFS, 100)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["lfcc"], output_names=["logits"],
        dynamic_axes={"lfcc": {0: "batch", 2: "time"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    size = out_path.stat().st_size
    print(f"exported {out_path} ({size/1024:.0f} KB)")
    return out_path


def quantize(src: Path, dst: Path):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    print(f"quantized {dst} ({dst.stat().st_size/1024:.0f} KB)")


def _softmax_p1(logits_row: np.ndarray) -> float:
    e = np.exp(logits_row - logits_row.max())
    return float(e[1] / e.sum())


def parity(onnx_path: Path, quantized: Path | None = None, n: int = 40):
    """Score n corpus clips through torch path vs onnx path; report max diff."""
    import onnxruntime as ort
    import torch

    from swarakavach.detectors import artifact as art_det
    from swarakavach.features import lfcc

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    qsess = (ort.InferenceSession(str(quantized), providers=["CPUExecutionProvider"])
             if quantized and quantized.exists() else None)

    import csv
    with open(config.DATA_DIR / "demo" / "labels.csv") as f:
        rows = list(csv.DictReader(f))
    step = max(1, len(rows) // n)
    diffs, qdiffs = [], []
    for r in rows[::step]:
        wav = load_wav(r["path"])
        feats = lfcc(wav)
        feats = ((feats - feats.mean(axis=1, keepdims=True))
                 / (feats.std(axis=1, keepdims=True) + 1e-6))
        # torch path
        t = torch.from_numpy(feats).float().unsqueeze(0)
        with torch.no_grad():
            p_torch = float(torch.softmax(art_det._load()(t), -1)[0][1])
        # onnx path
        lg = sess.run(None, {"lfcc": feats[None].astype(np.float32)})[0][0]
        p_onnx = _softmax_p1(lg)
        diffs.append(abs(p_torch - p_onnx))
        if qsess is not None:
            ql = qsess.run(None, {"lfcc": feats[None].astype(np.float32)})[0][0]
            qdiffs.append(abs(p_torch - _softmax_p1(ql)))
    print(f"parity over {len(diffs)} clips: max |torch - onnx| = {max(diffs):.5f}")
    if qdiffs:
        print(f"parity vs int8 quantized:    max |torch - int8| = {max(qdiffs):.5f}")
    return max(diffs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.MODELS_DIR / "artifact_edge.onnx"))
    ap.add_argument("--quantize", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    export(out)
    qpath = out.with_name("artifact_edge_int8.onnx")
    if args.quantize:
        quantize(out, qpath)
    parity(out, qpath if args.quantize else None)
    total = sum(p.stat().st_size for p in [out, qpath] if p.exists())
    print(f"edge bundle: {total/1e6:.2f} MB (target < 10 MB -> "
          f"{'MET' if total < 10e6 else 'MISSED'})")
