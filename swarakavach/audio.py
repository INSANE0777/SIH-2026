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


# ---- telephony degradation simulation (eval/robustness) ----

def bandlimit_8k(x: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Narrowband telephone channel: resample 16k -> 8k -> 16k."""
    t = torch.from_numpy(np.asarray(x, dtype=np.float32)).reshape(1, -1)
    t = AF.resample(t, sr, sr // 2)
    t = AF.resample(t, sr // 2, sr)
    return t.squeeze(0).numpy().astype(np.float32)


def mulaw_roundtrip(x: np.ndarray) -> np.ndarray:
    """G.711 mu-law companding + 8-bit quantization roundtrip."""
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    mag = np.log1p(255.0 * np.abs(x)) / np.log(256.0)
    q = np.round(mag * 255.0) / 255.0            # 8-bit quantize
    mag = (np.expm1(q * np.log(256.0))) / 255.0
    return (np.sign(x) * mag).astype(np.float32)


def add_noise(x: np.ndarray, snr_db: float = 20.0, seed: int = 0) -> np.ndarray:
    """Additive white noise at target SNR."""
    rng = np.random.default_rng(seed)
    sig = float(np.mean(x ** 2)) + 1e-12
    noise = rng.standard_normal(len(x)).astype(np.float32)
    noise *= np.sqrt(sig / (10 ** (snr_db / 10.0)))
    out = x + noise
    return (out / (np.abs(out).max() + 1e-9)).astype(np.float32) * min(1.0, np.abs(x).max())


def packet_loss(x: np.ndarray, frac: float = 0.02, seed: int = 0) -> np.ndarray:
    """Drop random 20 ms frames (concealment = hold last sample)."""
    out = np.asarray(x, dtype=np.float32).copy()
    n = len(out)
    frame = 320  # 20 ms @ 16k
    rng = np.random.default_rng(seed)
    for i in range(0, n - frame, frame):
        if rng.random() < frac:
            out[i:i + frame] = out[i - 1] if i > 0 else 0.0
    return out


def degrade(x: np.ndarray, level: str, sr: int = 16000) -> np.ndarray:
    """Named degradation levels for the telephony robustness eval."""
    if level == "clean":
        return np.asarray(x, dtype=np.float32)
    if level == "narrowband":
        return bandlimit_8k(x, sr)
    if level == "mulaw":
        return mulaw_roundtrip(bandlimit_8k(x, sr))
    if level == "noisy":
        return add_noise(mulaw_roundtrip(bandlimit_8k(x, sr)), snr_db=20.0)
    if level == "pl":
        return packet_loss(add_noise(mulaw_roundtrip(bandlimit_8k(x, sr)), snr_db=20.0), frac=0.02)
    raise ValueError(f"unknown degradation level: {level}")
