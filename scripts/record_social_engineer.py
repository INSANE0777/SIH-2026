"""Record the 'real human social engineer' demo call (finale call #3).

The one scenario no synthesizer can honestly produce: a REAL HUMAN voice
saying the scam script. Voice authenticity must score SAFE (it is human);
speaker layer abstains or flags mismatch (unfamiliar caller); context is HIGH;
intent rules fire CRITICAL — proving SwaraKavach catches fraud that is NOT
deepfake.

Team usage (10 seconds, any laptop mic):
    python scripts/record_social_engineer.py          # live mic recording
    python scripts/record_social_engineer.py --check  # verify the scenario works

Reads the script aloud (or paste any social-engineering lines you prefer),
writes data/demo/scenario/social_engineer_real.wav, then eval_ablation.py and
run_demo.py pick it up automatically.
"""
import argparse
import subprocess
import sys
import wave

from swarakavach import config

SCRIPT_LINES = [
    "Hi, quick one — this is Ravi from the head office audit team.",
    "We're reconciling a vendor payment and I need you to approve it right now,",
    "the fifty lakh transfer is already flagged urgent.",
    "Also read me the OTP on your phone so I can finish the verification.",
    "Please don't loop in anyone else yet, it's confidential.",
]


def record(out, seconds: int = 14):
    """Record from the default mic via ffmpeg (dshow) at 16 kHz mono."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "dshow", "-i", "audio=Microphone (Realtek High Definition Audio)",
        "-t", str(seconds), "-ar", "16000", "-ac", "1", str(out),
    ]
    print("recording", seconds, "s — READ ALOUD:")
    for line in SCRIPT_LINES:
        print("   ", line)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # device name varies; list devices and retry with the first audio input
        ls = subprocess.run(["ffmpeg", "-hide_banner", "-list_devices", "true",
                             "-f", "dshow", "-i", "dummy"], capture_output=True)
        import re

        m = re.search(r'"audio=([^"]+)"', ls.stderr.decode(errors="ignore"))
        if not m:
            print("no microphone found — plug one in and retry")
            sys.exit(1)
        cmd[cmd.index("audio=Microphone (Realtek High Definition Audio)")] = f"audio={m.group(1)}"
        print("recording with:", m.group(1))
        subprocess.run(cmd, check=True)
    print(f"\nwrote {out}")


def check():
    """Verify the recorded scenario produces the expected verdict shape."""
    from swarakavach.asr import transcribe
    from swarakavach.audio import iter_chunks, load_wav
    from swarakavach.detectors import artifact, intent
    from swarakavach.fusion import fuse, level_for

    import numpy as np

    p = config.DATA_DIR / "demo" / "scenario" / "social_engineer_real.wav"
    if not p.exists():
        print("no recording yet — run without --check first")
        sys.exit(1)
    wav = load_wav(p)
    tr = transcribe(wav)
    print("ASR heard:", tr[:120])
    traj = [fuse({"artifact": artifact.score(c), "intent": intent.score(tr)})
            for c in iter_chunks(wav)]
    lvl = level_for(float(np.median(traj)))
    print(f"voice layer (A) mean: {np.mean([artifact.score(c) for c in iter_chunks(wav)]):.3f} (want LOW=human)")
    print(f"intent rules:", intent.matched_rules(tr))
    print(f"fused verdict (A+E only): {lvl}")
    print("expected: A low (real human) + intent HIGH/CRITICAL (dangerous ask)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=14)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        check()
    else:
        record(config.DATA_DIR / "demo" / "scenario" / "social_engineer_real.wav", args.seconds)


if __name__ == "__main__":
    main()
