"""Generate the 2026 clone corpus — the real attack generator families.

Families (for leave-one-generator-out evaluation):
  - edge     Microsoft Edge neural TTS (en-IN, hi-IN, en-US, en-GB voices)
  - piper    local neural TTS (hi-IN pratham/priyamvada, en-US lessac)
  - sapi     Windows SAPI concatenative TTS
  - sarvam   Sarvam AI Bulbul v3 — INDIAN TTS, 11 languages, 30+ voices
             (env-gated: set SARVAM_API_KEY; the 'made in India, tested against
             made in India' attack family — API verified: client.text_to_speech
             .convert(model='bulbul:v3', text=..., language_code='hi-IN',
             speaker=..., speech_sample_rate=16000, output_audio_codec='wav'))
  - xtts     XTTS-v2 zero-shot VOICE CLONES of real speakers (the key attack;
             CPML non-commercial — training-corpus use only, documented)

Attack scripts: social-engineering transfer demands (EN + HI Devanagari),
matching the fraud scenario the PS describes.

Usage: python make_clone_corpus.py [--out data/clone-corpus] [--skip-xtts]
Output: <out>/<family>/<voice>_<i>.wav + manifest.csv (path, family, label=1)
Bonafide human audio: reused from SIH-2026/data/human (label=0) via --human-dir.
"""
import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000

SCRIPTS = [
    "Hello, this is the regional manager calling. I need you to approve the "
    "fifty lakh rupees transfer immediately. It is extremely urgent.",
    "This is the bank manager. Please share the OTP and your PIN right now "
    "so we can complete the urgent transfer.",
    "नमस्ते, मैं क्षेत्रीय प्रबंधक बोल रहा हूँ। पचास लाख रुपये का ट्रांसफर तुरंत स्वीकृत करें।",
    "कृपया अभी OTP और अपना पिन नंबर बताइए, यह बहुत जरूरी है।",
]

EDGE_VOICES = ["en-IN-NeerjaNeural", "en-IN-PrabhatNeural", "hi-IN-MadhurNeural",
               "hi-IN-SwaraNeural", "en-US-GuyNeural", "en-GB-SoniaNeural"]

PIPER_VOICES = {  # onnx name -> (path, langs supported)
    "pratham": "hi",
    "priyamvada": "hi",
    "lessac": "en",
}

HUMAN_DIR = Path("C:/SIH/SIH-2026/data/human")


def load_resample(p: Path, sr: int = SR) -> np.ndarray:
    import torch
    import torchaudio.functional as AF
    wav, orig = __import__("torchaudio").load(str(p))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if orig != sr:
        wav = AF.resample(wav, orig, sr)
    return wav.squeeze(0).numpy().astype(np.float32)


def gen_edge(out: Path) -> list[Path]:
    import asyncio
    import edge_tts
    files = []
    for voice in EDGE_VOICES:
        hi = voice.startswith("hi-IN")
        for i, text in enumerate(SCRIPTS):
            hi_texts = [2, 3]
            if (i in hi_texts) != hi:  # hi voices speak hi scripts only
                continue
            d = out / f"{voice.split('-')[2].lower().removesuffix('neural')}_{i}.wav"
            if not d.exists():
                asyncio.run(edge_tts.Communicate(text, voice).save(str(d)))
            files.append(d)
    return files


def gen_piper(out: Path, piper_dir: Path) -> list[Path]:
    from piper import PiperVoice
    files = []
    for name, lang in PIPER_VOICES.items():
        v = PiperVoice.load(str(piper_dir / f"{name}.onnx"))
        for i, text in enumerate(SCRIPTS):
            if (lang == "hi") != (i in (2, 3)):
                continue
            d = out / f"{name}_{i}.wav"
            if not d.exists():
                chunks = list(v.synthesize(text))
                audio = np.concatenate([c.audio_float_array for c in chunks])
                sf.write(str(d), audio, chunks[0].sample_rate)
            files.append(d)
    return files


def gen_sapi(out: Path) -> list[Path]:
    files = []
    for i, text in enumerate(SCRIPTS[:2]):  # SAPI en voice, English scripts
        d = out / f"sapi_{i}.wav"
        if not d.exists():
            subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Add-Type -AssemblyName System.Speech;"
                 "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                 f"$s.SetOutputToWaveFile('{d}');"
                 f"$s.Speak('{text}');"
                 "$s.Dispose()"],
                check=True, capture_output=True, timeout=60)
        files.append(d)
    return files


def gen_sarvam(out: Path) -> list[Path]:
    """Sarvam AI Bulbul v3 — the Indian TTS attack family (11 Indic languages).
    Env-gated: requires SARVAM_API_KEY. Skips with a message otherwise."""
    import os
    key = os.environ.get("SARVAM_API_KEY")
    if not key:
        print("  [sarvam] skipped (set SARVAM_API_KEY to add the Indian TTS family)")
        return []
    import base64

    from sarvamai import SarvamAI

    client = SarvamAI(api_subscription_key=key)
    speakers = {"shubh": "hi-IN", "ritu": "hi-IN", "aditya": "en-IN", "priya": "en-IN"}
    files = []
    for voice, lang in speakers.items():
        hi_texts = [2, 3]
        for i, text in enumerate(SCRIPTS):
            if (i in hi_texts) != lang.startswith("hi"):
                continue
            d = out / f"sarvam_{voice}_{i}.wav"
            if not d.exists():
                r = client.text_to_speech.convert(
                    model="bulbul:v3", text=text, language_code=lang, speaker=voice,
                    speech_sample_rate=16000, output_audio_codec="wav")
                d.write_bytes(base64.b64decode(r.audios[0]))
            files.append(d)
    return files


def gen_xtts(out: Path, skip: bool) -> list[Path]:
    if skip:
        print("  [xtts] skipped (--skip-xtts)")
        return []
    import torch
    # torch 2.6 defaults torch.load to weights_only=True; the official Coqui
    # release checkpoints pickle config objects and fail the allowlist whack-
    # a-mole. Trusted source (Coqui release channel) -> force weights_only=False.
    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
    from TTS.api import TTS
    files = []
    humans = sorted(HUMAN_DIR.glob("*.wav")) if HUMAN_DIR.exists() else []
    if not humans:
        print("  [xtts] no human reference speakers found")
        return []
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dev)
    print(f"  [xtts] loaded on {dev}, cloning {len(humans)} speakers x {len(SCRIPTS[:2])} scripts")
    for h in humans:
        try:
            spk = load_resample(h)
            ref = out / f"_tmp_ref_{h.stem}.wav"
            sf.write(str(ref), spk, SR)
            for i, text in enumerate(SCRIPTS[:2]):  # EN clones (xtts en is strongest)
                d = out / f"xtts_{h.stem}_{i}.wav"
                if not d.exists():
                    tts.tts_to_file(text=text, speaker_wav=str(ref), language="en",
                                    file_path=str(d))
                files.append(d)
            ref.unlink(missing_ok=True)
        except Exception as e:
            print(f"  [xtts] {h.stem} failed: {str(e)[:300]}")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="C:/SIH/SIH-2026/data/clone-corpus")
    ap.add_argument("--piper-dir", default="C:/SIH/clone-corpus/generators/piper")
    ap.add_argument("--skip-xtts", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []

    for fam, fn in [("edge", lambda: gen_edge(out / "edge")),
                    ("piper", lambda: gen_piper(out / "piper", Path(args.piper_dir))),
                    ("sapi", lambda: gen_sapi(out / "sapi")),
                    ("sarvam", lambda: gen_sarvam(out / "sarvam")),
                    ("xtts", lambda: gen_xtts(out / "xtts", args.skip_xtts))]:
        (out / fam).mkdir(exist_ok=True)
        print(f"generating family: {fam}")
        try:
            for f in fn():
                manifest.append((str(f), fam, 1))
        except Exception as e:
            print(f"  [{fam}] family failed: {str(e)[:120]}")

    # bonafide: real human speech, label 0
    if HUMAN_DIR.exists():
        for h in sorted(HUMAN_DIR.glob("*.wav")):
            manifest.append((str(h), "human", 0))

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "family", "label"])
        w.writerows(manifest)

    fams = {}
    for _, fam, lab in manifest:
        fams.setdefault(fam, [0, 0])[lab] += 1
    print(f"manifest: {len(manifest)} entries")
    for fam, (n0, n1) in sorted(fams.items()):
        print(f"  {fam:<8} bonafide={n0} attack={n1}")


if __name__ == "__main__":
    main()
