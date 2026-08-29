"""Audio I/O, resampling, chunking, PCM16 conversion, energy VAD."""
import io

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF


def load_wav(path, sr: int = 16000) -> np.ndarray:
    """Load any audio file as float32 mono numpy array at target rate."""
    wav, orig = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if orig != sr:
        wav = AF.resample(wav, orig, sr)
    return wav.squeeze(0).numpy().astype(np.float32)


def load_wav_bytes(data: bytes, sr: int = 16000) -> np.ndarray:
    """Decode in-memory audio bytes (wav/mp3/flac/ogg) to mono float32 at target rate."""
    import soundfile as sf

    wav, orig = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if orig != sr:
        wav = AF.resample(torch.from_numpy(wav).unsqueeze(0), orig, sr).squeeze(0).numpy()
    return wav.astype(np.float32)


def save_wav(path, wav: np.ndarray, sr: int = 16000) -> None:
    t = torch.from_numpy(np.asarray(wav, dtype=np.float32).reshape(1, -1))
    torchaudio.save(str(path), t, sr)


def pcm16_to_float(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(x: np.ndarray) -> bytes:
    q = np.round(np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0) * 32768.0)
    q = np.clip(q, -32768, 32767).astype("<i2")
    return q.tobytes()


def iter_chunks(x: np.ndarray, chunk: int = 16000, overlap: int = 8000):
    """Yield overlapping chunks; a short signal is yielded once."""
    x = np.asarray(x)
    if len(x) <= chunk:
        yield x
        return
    hop = chunk - overlap
    for i in range(0, len(x) - overlap, hop):
        yield x[i : i + chunk]
        if i + chunk >= len(x):
            break


def frame_rms(x: np.ndarray, sr: int, frame_s: float = 0.025, hop_s: float = 0.010) -> np.ndarray:
    frame, hop = int(sr * frame_s), int(sr * hop_s)
    if len(x) < frame:
        return np.array([float(np.sqrt(np.mean(x**2) + 1e-12))])
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return np.sqrt(np.mean(x[idx] ** 2, axis=1) + 1e-12)


def vad_speech_ratio(x: np.ndarray, sr: int, thresh_db: float = -45.0) -> float:
    """Fraction of frames above energy threshold (simple VAD)."""
    rms = frame_rms(x, sr)
    thresh = 10 ** (thresh_db / 20.0)
    return float(np.mean(rms > thresh))
