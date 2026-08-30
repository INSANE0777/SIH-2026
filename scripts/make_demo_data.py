"""Generate the bootstrap demo corpus + finale scenario files.

Real class  : real human speech (public samples) chunked to call length.
Attack class: real neural TTS (edge-tts — free, no API key; incl. Indian
              English voices), the mel-vocoder clone proxy (same voice,
              vocoder artifacts), and vocoded TTS.

Downloads human samples on first run (network needed once):
  - whisper.cpp samples/jfk.wav          (public domain speech)
  - speechbrain test single-mic samples  (Apache-2.0)
Replace with ASVspoof5 + a 2026 clone corpus (F5-TTS, XTTS-v2, RVC, ...) for
real training — see docs/architecture.md.

Output: data/demo/labels.csv (path,label  0=bonafide, 1=spoof)
        data/demo/scenario/manager_real.wav, manager_clone.wav, attacker_tts.wav
"""
import argparse
import asyncio
import csv
import subprocess
import urllib.request
from pathlib import Path

from swarakavach import config
from swarakavach.audio import iter_chunks, load_wav, save_wav
from swarakavach.synth import SR, vocode

HUMAN_SOURCES = [
    ("jfk.wav", "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"),
    ("example1.wav", "https://github.com/speechbrain/speechbrain/raw/develop/tests/samples/single-mic/example1.wav"),
    ("example5.wav", "https://github.com/speechbrain/speechbrain/raw/develop/tests/samples/single-mic/example5.wav"),
    ("example6.wav", "https://github.com/speechbrain/speechbrain/raw/develop/tests/samples/single-mic/example6.wav"),
] + [
    (f"sb_spk{sp}_snt{n}.wav",
     f"https://github.com/speechbrain/speechbrain/raw/develop/tests/samples/ASR/spk{sp}_snt{n}.wav")
    for sp in (1, 2) for n in range(1, 7)
]

TTS_SCRIPT = (
    "Hello, this is the regional manager calling. I need you to approve the "
    "fifty lakh rupee transfer immediately. It is extremely urgent. Please do "
    "not discuss this with anyone else in the office."
)

# Hindi attack script: hits the intent rules (transfer/lakh, urgency, secrecy, authority)
HINDI_SCRIPT = (
    "नमस्ते, मैं क्षेत्रीय प्रबंधक बोल रहा हूँ। पचास लाख रुपये का ट्रांसफर "
    "तुरंत स्वीकृत करें। यह बात किसी से न कहें।"
)

# English attacker script: OTP + PIN harvesting (credentials rules)
ATTACKER_SCRIPT = (
    "This is the bank manager calling. Please share the OTP and your PIN "
    "right now so we can complete the urgent rupees transfer."
)

EDGE_VOICES = [
    "en-IN-NeerjaNeural",   # Indian English female
    "en-IN-PrabhatNeural",  # Indian English male
    "hi-IN-MadhurNeural",   # Hindi male (multilingual attack)
    "hi-IN-SwaraNeural",    # Hindi female (multilingual attack)
    "en-US-GuyNeural",
    "en-GB-SoniaNeural",
]


def fetch_human(out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    wavs = []
    for name, url in HUMAN_SOURCES:
        dest = out / name
        if not dest.exists() or dest.stat().st_size < 1000:
            print(f"fetching {name} ...")
            urllib.request.urlretrieve(url, dest)
        if dest.stat().st_size > 1000:
            wavs.append(dest)
    if not wavs:
        raise SystemExit("no human samples — check network or drop wav files into data/human/")
    return wavs


def edge_tts_wav(text: str, voice: str, dest: Path) -> bool:
    """Real neural TTS sample via edge-tts (Microsoft Edge read-aloud, free)."""
    try:
        import edge_tts

        async def gen():
            await edge_tts.Communicate(text, voice).save(str(dest))

        asyncio.run(gen())
        return dest.exists() and dest.stat().st_size > 1000
    except Exception as e:
        print(f"edge-tts {voice} unavailable ({str(e)[:60]}) — skipped")
        return False


def sapi_tts(dest: Path, rate: int = 0) -> bool:
    """Windows SAPI TTS (fallback attack source, Windows only)."""
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Add-Type -AssemblyName System.Speech;"
             "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
             f"$s.Rate = {rate};"
             f"$s.SetOutputToWaveFile('{dest}');"
             f"$s.Speak('{TTS_SCRIPT}');"
             "$s.Dispose()"],
            check=True, capture_output=True, timeout=60)
        return dest.exists() and dest.stat().st_size > 1000
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.DATA_DIR / "demo"))
    args = ap.parse_args()

    out = Path(args.out)
    (out / "real").mkdir(parents=True, exist_ok=True)
    (out / "attack").mkdir(parents=True, exist_ok=True)
    (out / "scenario").mkdir(parents=True, exist_ok=True)

    humans = fetch_human(config.DATA_DIR / "human")

    # ---- real class: chunked human speech (2 s windows, 1 s hop) ----
    rows, real_chunks = [], []
    for hw in humans:
        wav = load_wav(hw)
        for i, ch in enumerate(iter_chunks(wav, chunk=2 * SR, overlap=SR)):
            if len(ch) < 1.5 * SR:
                continue
            p = out / "real" / f"{hw.stem}_{i:02d}.wav"
            save_wav(p, ch, SR)
            rows.append([str(p), 0])
            real_chunks.append(p)

    # ---- attack class 1: vocoder clone proxy (same voice, TTS artifacts) ----
    for p in real_chunks:
        q = out / "attack" / f"voc_{p.stem}.wav"
        save_wav(q, vocode(load_wav(p)), SR)
        rows.append([str(q), 1])

    # ---- attack class 2: real neural TTS (multiple voices, EN + HI) ----
    tts_files = []
    for voice in EDGE_VOICES:
        text = HINDI_SCRIPT if voice.startswith("hi-IN") else TTS_SCRIPT
        vname = voice.split("-")[2].lower().removesuffix("neural")  # neerja, madhur, ...
        d = out / "attack" / f"edge_{vname}.wav"
        if edge_tts_wav(text, voice, d):
            rows.append([str(d), 1])
            tts_files.append(d)
            # chunk long TTS clips to match training window length
            for i, ch in enumerate(iter_chunks(load_wav(d), chunk=2 * SR, overlap=SR)):
                if len(ch) < 1.5 * SR:
                    continue
                c = out / "attack" / f"edge_{vname}_{i:02d}.wav"
                save_wav(c, ch, SR)
                rows.append([str(c), 1])
    if not tts_files:  # no network for edge-tts — Windows SAPI fallback
        d = out / "attack" / "sapi_tts.wav"
        if sapi_tts(d):
            rows.append([str(d), 1])
            tts_files.append(d)
            for i, ch in enumerate(iter_chunks(load_wav(d), chunk=2 * SR, overlap=SR)):
                if len(ch) < 1.5 * SR:
                    continue
                c = out / "attack" / f"sapi_tts_{i:02d}.wav"
                save_wav(c, ch, SR)
                rows.append([str(c), 1])

    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows(rows)

    # ---- finale scenario ----
    # genuine call = real human (the 'regional manager' voiceprint to enroll)
    manager = load_wav(humans[0])[: 6 * SR]
    save_wav(out / "scenario" / "manager_real.wav", manager, SR)
    # cloned manager = neural TTS attack claiming to be the manager (the real
    # threat class; the Griffin-Lim proxy is corpus-only training variety)
    clone = out / "scenario" / "manager_clone.wav"
    if not edge_tts_wav(TTS_SCRIPT, "en-IN-PrabhatNeural", clone):
        # offline fallback: vocoder proxy of the manager (weaker attack class)
        save_wav(clone, vocode(manager), SR)
    # attacker with an OTP-harvesting English TTS voice
    if tts_files:
        atk = out / "scenario" / "attacker_tts.wav"
        if not edge_tts_wav(ATTACKER_SCRIPT, "en-IN-NeerjaNeural", atk):
            save_wav(atk, load_wav(tts_files[0]), SR)
        # Hindi attacker (multilingual neural TTS attack)
        hindi = [f for f in tts_files if "madhur" in f.name or "swara" in f.name]
        if hindi:
            save_wav(out / "scenario" / "attacker_hindi.wav", load_wav(hindi[0]), SR)

    print(f"wrote {len(rows)} clips ({sum(1 for _, l in rows if l == 0)} real / "
          f"{sum(1 for _, l in rows if l == 1)} attack) + scenario to {out}")


if __name__ == "__main__":
    main()
