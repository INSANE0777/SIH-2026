"""WS-path verification: stream PCM16 chunks like the browser demo does.

Confirms the backend returns correct verdicts over the exact WS protocol the
fixed demo player uses (Int16 chunks — the Float32 send was the all-LOW bug).

Usage: python scripts/ws_verify.py
"""
import asyncio
import json

import numpy as np
import websockets

from swarakavach.audio import load_wav

CALLS = [
    ("genuine", "data/demo/scenario/manager_real.wav"),
    ("clone", "data/demo/scenario/manager_clone.wav"),
    ("hindi attacker", "data/demo/scenario/attacker_hindi.wav"),
]
URL = ("ws://127.0.0.1:8000/v1/stream?speaker_id=regional_manager"
       "&context=%7B%22caller_id_match%22%3A%22false%22%2C%22enrolled_contact%22%3A%22true%22%2C%22tx_amount%22%3A5000000%7D")


async def test(name, path):
    wav = load_wav(path)
    async with websockets.connect(URL, max_size=50 * 1024 * 1024) as ws:
        final = None
        for i in range(0, len(wav), 16000):
            ch = wav[i:i + 16000]
            if len(ch) < 8000:
                break
            pcm16 = (np.clip(ch, -1, 1) * 32767).astype("<i2")
            await ws.send(pcm16.tobytes())
            msg = json.loads(await ws.recv())
            if msg.get("final"):
                final = msg
                break
        if final is None:
            await ws.send("END")
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 20))
                if msg.get("final"):
                    final = msg
                    break
    v = final["verdict"] if "verdict" in final else final
    print(f"{name:15s} -> {v['level']:9s} peak={v['risk_peak']}  actions={v['actions'][:2]}")
    return v


async def main():
    for name, path in CALLS:
        await test(name, path)


if __name__ == "__main__":
    asyncio.run(main())
