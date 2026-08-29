"""Fetch real human-speech verification samples from public HF-hosted assets,
then validate the wav2vec2 spoof-index mapping: human (bonafide) must score
LOW, real TTS + vocoder proxies (spoof) must score HIGH."""
import json
import sys
import urllib.request

import numpy as np

from swarakavach.audio import load_wav
from swarakavach.detectors import ssl_detector
from swarakavach.synth import synth_phrase, vocode

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
    sapi = vdir / "sapi_tts.wav"
    human_paths = []
    try:
        human_paths.append(human_sample(vdir / "human_lj.flac"))
        print("fetched human speech sample")
    except Exception as e:
        print(f"WARNING: could not fetch human sample: {e}")

    if not ssl_detector.available():
        sys.exit("SSL layer unavailable")

    def p(name, wav):
        s = ssl_detector.score(wav)
        print(f"{name:<28} p[spoof]={s:.3f}")
        return s

    scores = {}
    for hp in human_paths:
        scores["human"] = p(f"human ({hp.name})", load_wav(hp))
    scores["sapi_tts"] = p("Windows SAPI TTS", load_wav(sapi))
    scores["formant_synth"] = p("formant synth (proxy)", synth_phrase(seed=7, dur=3.0))
    scores["vocoder"] = p("Griffin-Lim vocoded (proxy)", vocode(synth_phrase(seed=7, dur=3.0)))

    if "human" in scores and scores["human"] < 0.5 and scores["sapi_tts"] > 0.5:
        print("\n>>> MAPPING OK: human=bonafide, TTS=spoof, spoof_idx="
              f"{ssl_detector._state['spoof_idx']}")
    else:
        print("\n>>> mapping UNCLEAR — inspect scores above and set SWK_SSL_SPOOF_IDX")


if __name__ == "__main__":
    main()
