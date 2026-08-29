"""SwaraKavach API server: REST + WebSocket streaming + live dashboard.

Endpoints
---------
GET  /                      live dashboard (web/index.html)
GET  /healthz               liveness + layer availability
POST /v1/analyze            body = WAV bytes (raw) → full verdict
POST /v1/enroll?speaker_id= body = WAV bytes → enroll caller voiceprint
WS   /v1/stream             binary PCM16LE 16k mono chunks → JSON risk events
GET  /v1/sessions/{id}      replay a session's verdict
"""
import asyncio
import json
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from . import config
from .audio import iter_chunks, load_wav_bytes, pcm16_to_float
from .detectors import artifact, context as ctx_engine, speaker, ssl_detector
from .fusion import RiskEngine

app = FastAPI(title="SwaraKavach", version="0.1.0",
              description="Real-time voice cloning impersonation detection & prevention — SIH 2026 PS26104")

SESSIONS: dict[str, dict] = {}


def run_session(wav: np.ndarray, context: dict | None, speaker_id: str | None,
                session_id: str | None = None) -> dict:
    engine = RiskEngine(context)
    for chunk in iter_chunks(wav, config.CHUNK_SAMPLES, config.CHUNK_OVERLAP_SAMPLES):
        engine.update({
            "artifact": artifact.score(chunk),
            "ssl": ssl_detector.score(chunk),
            "speaker": speaker.score(speaker_id, chunk) if speaker_id else None,
            "context": ctx_engine.score(context),
        })
    verdict = engine.verdict()
    sid = session_id or uuid.uuid4().hex[:12]
    SESSIONS[sid] = {
        "session_id": sid, "ts": time.time(), "context": context or {},
        "speaker_id": speaker_id, "verdict": verdict,
        "trajectory": engine.trajectory, "layers": engine.layer_history,
    }
    if "INCIDENT_WEBHOOK" in verdict["actions"] and config.WEBHOOK_URL:
        _webhook(SESSIONS[sid])
    return SESSIONS[sid]


def _webhook(payload: dict):
    try:
        req = urllib.request.Request(
            config.WEBHOOK_URL, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:  # alerting must never take the call path down
        print(f"[webhook] failed: {e}")


@app.get("/", response_class=FileResponse)
def index():
    return FileResponse(Path(config.WEB_DIR) / "index.html")


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "layers": {
            "artifact": artifact.available(),
            "ssl": ssl_detector.available(),
            "speaker_enrollments": len(list((config.DATA_DIR / "enrollments").glob("*.npz")))
            if (config.DATA_DIR / "enrollments").exists() else 0,
        },
        "ssl_model": config.SSL_MODEL_ID if ssl_detector.available() else None,
    }


@app.post("/v1/analyze")
async def analyze(body: bytes = Body(...), speaker_id: str | None = Query(None),
                  context: str | None = Query(None)):
    try:
        wav = load_wav_bytes(body)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"bad audio: {e}"})
    ctx = json.loads(context) if context else None
    return run_session(wav, ctx, speaker_id)


@app.post("/v1/enroll")
async def enroll(body: bytes = Body(...), speaker_id: str = Query(...)):
    try:
        wav = load_wav_bytes(body)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"bad audio: {e}"})
    return speaker.enroll(speaker_id, wav)


@app.get("/v1/sessions/{sid}")
def session(sid: str):
    if sid not in SESSIONS:
        return JSONResponse(status_code=404, content={"error": "unknown session"})
    return SESSIONS[sid]


@app.websocket("/v1/stream")
async def stream(ws: WebSocket, speaker_id: str | None = Query(None),
                 context: str | None = Query(None)):
    """Live stream protocol: client sends binary PCM16LE 16k-mono frames;
    server replies per-chunk risk events; text frame 'END' finalizes."""
    await ws.accept()
    engine = RiskEngine(json.loads(context) if context else None)
    sid = uuid.uuid4().hex[:12]
    buf = np.zeros(0, dtype=np.float32)
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if (text := msg.get("text")) is not None:
                if text.strip().upper() == "END":
                    break
                continue
            data = msg.get("bytes")
            if not data:
                continue
            buf = np.concatenate([buf, pcm16_to_float(data)])
            while len(buf) >= config.CHUNK_SAMPLES:
                chunk, buf = buf[: config.CHUNK_SAMPLES], buf[config.CHUNK_SAMPLES:]
                ev = engine.update({
                    "artifact": artifact.score(chunk),
                    "ssl": await asyncio.to_thread(ssl_detector.score, chunk),
                    "speaker": speaker.score(speaker_id, chunk) if speaker_id else None,
                    "context": ctx_engine.score(engine.context),
                })
                await ws.send_text(json.dumps({"session_id": sid, **ev}))
        verdict = engine.verdict()
        SESSIONS[sid] = {"session_id": sid, "ts": time.time(), "context": engine.context,
                         "speaker_id": speaker_id, "verdict": verdict,
                         "trajectory": engine.trajectory, "layers": engine.layer_history}
        if "INCIDENT_WEBHOOK" in verdict["actions"] and config.WEBHOOK_URL:
            _webhook(SESSIONS[sid])
        await ws.send_text(json.dumps({"session_id": sid, "final": True, **verdict}))
    except WebSocketDisconnect:
        pass
