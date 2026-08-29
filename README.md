# SwaraKavach — Real-Time Voice Clone Impersonation Detection & Prevention

**SIH 2026 · Problem Statement ID 26104 · Organization: AICTE · Theme: Blockchain & Cybersecurity**

> AI voice-cloning is being used to impersonate executives and family members over
> phone calls to authorize fraudulent transactions. Indians lost **₹2,140 crore to
> digital-arrest scams alone in the first 10 months of 2024** (MHA/I4C). SwaraKavach
> gives call centers, banks, and telephony platforms a **real-time risk engine** that
> scores every live call for voice-clone impersonation while the conversation is still
> ongoing — and **prevents** the fraud, not just detects it (transaction HOLD, MFA
> step-up, liveness challenge, supervisor escalation).

---

## How it works

Four independent detector layers run per 1-second chunk of live audio. A weighted
late-fusion engine calibrates and smooths their outputs into a **dynamic risk score
(EMA trajectory)**, mapped to risk levels and a response-action policy:

```
        live call audio (PCM16, 16 kHz)          call metadata (caller ID, tx amount)
                │                                        │
   ┌────────────┼────────────┐                          │
   ▼            ▼            ▼                          ▼
[A] Artifact  [B] SSL      [C] Speaker               [D] Context
 LFCC + CNN    wav2vec2     prosody vs enrolled      deterministic
 (60k params,  deepfake     voiceprint (cosine)      risk rules
  edge-ready)  classifier
   └────────────┴────────────┴──────────┬───────────────┘
                                       ▼
                    weighted late fusion → temperature calibration
                                       ▼
                        EMA risk trajectory (LOW / MID / HIGH / CRITICAL)
                                       ▼
        policy: CAUTION_PROMPT · STEP_UP_MFA · HOLD_TRANSACTION ·
                LIVENESS_CHALLENGE · SUPERVISOR_ALERT · INCIDENT_WEBHOOK
```

**Defense in depth is the point** — no single detector survives contact with modern
neural TTS. In our own benchmark, the SSL layer flags Windows SAPI TTS (p=0.98)
but misses Microsoft Edge neural TTS (p=0.08); the artifact layer catches both.
Fusion keeps the verdict alive when any one layer is blind.

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# 1. build the bootstrap demo corpus (downloads public human-speech samples,
#    generates real neural-TTS attacks via edge-tts + vocoder clone proxies)
python scripts/make_demo_data.py

# 2. train the layer-A artifact CNN (bootstrap corpus; ~1 min CPU)
python scripts/train_artifact.py

# 3. run the three-call finale demo (genuine → cloned → TTS attacker)
python scripts/run_demo.py

# 4. benchmarks (per-layer + fused EER)
python scripts/eval_benchmarks.py

# 5. API + live dashboard
uvicorn swarakavach.server:app --port 8000     # open http://localhost:8000
```

**Expected demo output:** genuine manager call → LOW, transaction proceeds;
cloned manager → HIGH, transaction HOLD + liveness challenge + supervisor alert;
TTS attacker → CRITICAL, all actions + incident webhook.

## API

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | liveness + which detector layers are active |
| `POST /v1/enroll?speaker_id=X` | body = WAV/MP3 bytes → enroll caller voiceprint |
| `POST /v1/analyze?speaker_id=X&context={json}` | body = WAV/MP3 bytes → full verdict + trajectory |
| `WS /v1/stream` | send binary PCM16LE 16k-mono frames → per-chunk risk events; text `END` finalizes |
| `GET /v1/sessions/{id}` | replay any session verdict |

The dashboard at `/` supports microphone streaming, WAV file analysis, and
enrollment — zero build step, single HTML file.

## What is real vs. bootstrap (honest status)

| Component | Status |
|---|---|
| Fusion + risk engine + policy actions | **Working**, unit-tested |
| REST + WebSocket API + dashboard | **Working**, end-to-end tested |
| Layer A artifact CNN | **Working** — trained on the bootstrap corpus in `scripts/make_demo_data.py` (real human speech + real neural TTS + vocoder proxies), EER 0.00 on that corpus |
| Layer B SSL detector | **Working** — `Hemgg/Deepfake-audio-detection` (wav2vec2, ASVspoof-based); validated: human 0.03 / SAPI TTS 0.98 / Edge neural TTS 0.08 (misses modern neural TTS — documented, layer A covers it) |
| Layer C speaker consistency | **Working** — prosody+LFCC voiceprint, cosine distance; abstains without enrollment |
| Layer D context rules | **Working** — deterministic metadata risk rules |
| ASVspoof 2019/2021/5 training | **Planned** — `train_artifact.py --asvspoof-dir` already accepts the ASVspoof2019 LA layout |
| Real 2026 clone corpus (F5-TTS, XTTS-v2, RVC, Fish Speech) | **Planned** — see `docs/architecture.md` |
| Telephony realism (8 kHz μ-law, AMR/EVS/Opus codec sim, RIR, MUSAN) | **Planned** |

We deliberately ship the bootstrap corpus generator instead of committed audio so
every number in this repo is reproducible from code.

## Repository layout

```
swarakavach/            core package
  audio.py              I/O, resampling, chunking, VAD, PCM16
  features.py           LFCC, F0 tracking, prosody vectors, speaker stats
  synth.py              formant synthesis + mel-vocoder clone proxy (data bootstrap)
  detectors/            A artifact · B ssl · C speaker · D context
  fusion.py             late fusion, calibration, EMA, policy verdicts
  server.py             FastAPI REST + WebSocket + dashboard
scripts/                corpus gen, training, benchmarks, demo, SSL mapping check
web/index.html          live dashboard (zero-build)
tests/                  pytest suite (9 tests)
docs/                   architecture + demo runbook
```

## Team & license

SIH 2026 submission. Code: MIT. Model checkpoints keep their upstream licenses —
the SSL layer is a wav2vec2 fine-tune (Apache-2.0 base); any XTTS-v2-based clone
corpus tooling is **non-commercial (CPML) — training/research use only** and will
be flagged accordingly.

## Roadmap

1. **Week 1–2:** real training — ASVspoof 2019/2021 LA + own 2026 clone corpus
   (F5-TTS, XTTS-v2, OpenVoice 2, RVC, Fish Speech at 3 s/10 s enrollment).
2. **Week 3–4:** telephony-realistic augmentation (μ-law 8 kHz, AMR/EVS/Opus
   codec simulation, MUSAN noise, RIR reverb, packet loss) — attacks the known
   in-the-wild degradation of lab-trained detectors.
3. **Week 5–6:** Indic language data (AI4Bharat IndicVoices/Shrutilipi),
   streaming latency budget (p95 < 300 ms/chunk), gRPC + Twilio Media Streams demo.
4. **Week 7–8:** ONNX edge export (< 10 MB bundle), leave-one-tool-out evaluation
   (the generalization moat), dashboard polish, demo freeze.
