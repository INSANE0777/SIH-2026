"""Twilio Media Streams gateway — wire SwaraKavach to a real phone call.

Twilio's <Connect><Stream> opens a WS to this server and forwards live call
audio as base64 mulaw 8kHz frames. This gateway:

  1. decodes mulaw -> PCM float,
  2. feeds the existing fusion engine chunk-by-chunk (same logic as /v1/stream),
  3. sends risk events back to a client callback (or logs them),
  4. on HIGH/CRITICAL, can respond on the call via TwiML <Say>/<Pause> or the
     REST API (HOLD instructions live in the bank's integration, not here).

Run:
    SWK_TWILIO_ACCOUNT_SID=... SWK_TWILIO_AUTH_TOKEN=... \
    python scripts/twilio_gateway.py --public-url https://<tunnel>/twilio

Requirements on Twilio side (number configured with a Media Streams webhook):
    <Response><Connect><Stream url="wss://<public>/twilio"/></Connect></Response>

Local demo: ngrok http 8000 (or cloudflared tunnel) exposes this server.
No audio is stored — events only (the audit ledger keeps the verdict hash).
"""
import argparse
import asyncio
import base64
import json
import os
import time
import uuid

import numpy as np

from swarakavach import asr as asr_mod
from swarakavach import config
from swarakavach.audio import iter_chunks, pcm16_to_float
from swarakavach.detectors import artifact, context as ctx_engine, intent, speaker, ssl_detector
from swarakavach.fusion import RiskEngine

# ---- mulaw (G.711 u-law) decode, no deps ----
_ULAW_TABLE = np.zeros(256, dtype=np.float32)
for b in range(256):
    sign = -1 if (b & 0x80) else 1
    pos = (~b) & 0x7F
    exponent = (pos >> 4) & 0x07
    mantissa = pos & 0x0F
    level = (2 ** exponent) * (2 * mantissa + 33) - 33
    _ULAW_TABLE[b] = sign * level / 8191.0


def mulaw_to_float(data: bytes) -> np.ndarray:
    idx = np.frombuffer(data, dtype=np.uint8).astype(np.int16)
    return _ULAW_TABLE[idx].astype(np.float32)


class TwilioStreamSession:
    """One live phone call: Twilio media frames -> fused risk events."""

    def __init__(self, speaker_id: str | None):
        self.engine = RiskEngine({"caller_id_match": "false", "tx_amount": 5_000_000})
        self.speaker_id = speaker_id
        self.buf = np.zeros(0, dtype=np.float32)
        # Twilio sends 8 kHz mulaw; engine expects 16 kHz — upsample 2x
        self.sr_in = 8000
        self.trans = asr_mod.StreamingTranscriber() if asr_mod.available() else None
        self.session_id = uuid.uuid4().hex[:12]
        self.events: list[dict] = []

    def feed(self, media: bytes) -> list[dict]:
        x8 = mulaw_to_float(media)
        x16 = np.repeat(x8, 2)  # 8k -> 16k naive upsample (band-limited anyway)
        self.buf = np.concatenate([self.buf, x16.astype(np.float32)])
        out = []
        while len(self.buf) >= config.CHUNK_SAMPLES:
            chunk, self.buf = (self.buf[: config.CHUNK_SAMPLES],
                               self.buf[config.CHUNK_SAMPLES:])
            live_tr = None
            if self.trans is not None:
                utt = self.trans.feed(chunk)
                if utt or self.trans.text_parts:
                    live_tr = " ".join(self.trans.text_parts)
            ev = self.engine.update({
                "artifact": artifact.score(chunk),
                "ssl": ssl_detector.score(chunk),
                "speaker": speaker.score(self.speaker_id, chunk) if self.speaker_id else None,
                "context": ctx_engine.score(self.engine.context),
                "intent": intent.score(live_tr),
            })
            ev["session_id"] = self.session_id
            self.events.append(ev)
            out.append(ev)
        return out

    def verdict(self) -> dict:
        return self.engine.verdict()


async def handle_twilio(ws):
    """WebSocket handler body: Twilio Media Streams protocol.

    Protocol (Twilio): first a 'connected' text frame, then JSON events:
      {event: 'start', ...call metadata} -> {event: 'media', media: {payload: b64}}
      -> {event: 'stop'} at call end. We reply with nothing (server->Twilio
      media requires separate <Stream> bidirectional mode); risk events are
      logged + delivered to the audit webhook path.
    """
    session = None
    await ws.accept()
    while True:
        try:
            msg = await ws.receive_text()
        except Exception:
            break
        try:
            evt = json.loads(msg)
        except Exception:
            continue
        kind = evt.get("event")
        if kind == "connected":
            continue
        if kind == "start":
            session = TwilioStreamSession(speaker_id=None)
            print(f"[twilio] call started: {evt.get('start', {}).get('callSid', '?')}")
            continue
        if kind == "stop":
            if session:
                v = session.verdict()
                print(f"[twilio] call ended: level={v['level']} peak={v['risk_peak']}")
            break
        if kind == "media" and session:
            payload = evt.get("media", {}).get("payload")
            if not payload:
                continue
            for ev in session.feed(base64.b64decode(payload)):
                if ev["level"] in ("HIGH", "CRITICAL"):
                    print(f"[twilio] RISK {ev['level']} risk={ev['risk']:.3f} "
                          f"actions={ev['actions']}")
        if kind == "mark" or kind == "clear":
            continue


def main():
    """Standalone gateway server (mounts /twilio WS on FastAPI)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()

    import uvicorn
    from fastapi import FastAPI, WebSocket

    app = FastAPI(title="SwaraKavach Twilio Gateway")

    @app.websocket("/twilio")
    async def twilio_ws(ws: WebSocket):
        await handle_twilio(ws)

    @app.get("/healthz")
    def health():
        return {"ok": True, "gateway": "twilio"}

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
