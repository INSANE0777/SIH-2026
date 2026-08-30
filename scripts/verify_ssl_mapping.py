"""Validate the wav2vec2 SSL checkpoint's spoof-index mapping.

Probes the loaded SSL layer with real human speech (jfk.wav, must score LOW)
vs real TTS (SAPI + edge + formant/vocoder proxies, must score HIGH).
Run after any SSL model change (SWK_SSL_MODEL env var).

History: this probe is what caught two broken checkpoints — DavidCombei/
wav2vec2-base-ASVSpoof5_TUC-N returned constant logits for every input,
and Hemgg/Deepfake-audio-detection misses Edge neural TTS at call level
(0.08) while catching SAPI (0.98) — the live single-detector-failure story.
"""
import sys

import numpy as np

from swarakavach import config
from swarakavach.audio import load_wav
from swarakavach.detectors import ssl_detector

UA = {"User-Agent": "Mozilla/5.0"}


def fetch(url, dest):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def human_sample(dest):
    """LibriSpeech train.100 row audio via the HF datasets-server asset API."""
    api = ("https://datasets-server.huggingface.co/assets?"
           "dataset=openslr%2Flibrispeech_asr&config=clean&split=train.100&row=0&audio_col=audio")
    with urllib.request.urlopen(urllib.request.Request(api, headers=UA), timeout=60) as r:
        url = json.load(r)["url"]
    return fetch(url, dest)


def main():
    from pathlib import Path

    vdir = Path("data/verify")
    vdir.mkdir(parents=True, exist_ok=True)
    probes = []
    human = Path("data/human/jfk.wav")
    if human.exists():
        probes.append(("human", "real human speech", load_wav(human)))
    sapi = Path("data/verify/sapi_tts.wav")
    if sapi.exists():
        probes.append(("sapi_tts", "Windows SAPI TTS", load_wav(sapi)))
    else:
        from swarakavach.synth import synth_phrase
        probes.append(("formant_synth", "formant synth (synthetic)",
                       synth_phrase(seed=7, dur=3.0)))
    for p_ in ["data/demo/scenario/attacker_tts.wav",
               "data/demo/scenario/manager_clone.wav"]:
        q = Path(p_)
        if q.exists():
            probes.append((q.stem, f"scenario {q.stem}", load_wav(q)))

    if not ssl_detector.available():
        sys.exit("SSL layer unavailable")

    print(f"ssl model: {config.SSL_MODEL_ID} | spoof_idx: {ssl_detector._state['spoof_idx']}")
    scores = {}
    for key, name, wav in probes:
        s = ssl_detector.score(wav)
        scores[key] = s
        print(f"  {name:<28} p[spoof]={s:.3f}")

    ok = "human" in scores and scores["human"] < 0.5 and \
        any(v > 0.5 for k, v in scores.items() if k != "human")
    print(f"\n>>> {'MAPPING OK' if ok else 'MAPPING UNCLEAR — inspect scores above'}")


if __name__ == "__main__":
    main()
