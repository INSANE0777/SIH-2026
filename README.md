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

Five independent detector layers run per 1-second chunk of live audio. A weighted
late-fusion engine calibrates and smooths their outputs into a **dynamic risk score
(EMA trajectory)**, mapped to risk levels and a response-action policy. Each layer
answers one independent question — that separation is the architecture's defense:

```
        live call audio (PCM16, 16 kHz)          call metadata          transcript
                │                             (caller ID, tx amount)      (from ASR)
   ┌────────────┼────────────┐                        │                    │
   ▼            ▼            ▼                        ▼                    ▼
[A] Artifact  [B] SSL      [C] Speaker             [D] Context         [E] Intent
 LFCC + CNN    wav2vec2     voiceprint vs          deterministic       rule-based social-
 (60k params,  deepfake     enrollment             metadata risk      engineering signals
  edge-ready)  classifier   (cosine)               rules               (EN + HI, auditable)
   └────────────┴────────────┴────────────┬──────────┴─────────┬────────────┘
                                        ▼
                    weighted late fusion → temperature calibration
                                        ▼
                EMA risk trajectory → median-robust level assignment
                (LOW / MID / HIGH / CRITICAL) + explainable breakdown
                                        ▼
        policy: CAUTION_PROMPT · STEP_UP_MFA · HOLD_TRANSACTION ·
                LIVENESS_CHALLENGE · SUPERVISOR_ALERT · INCIDENT_WEBHOOK
                                        ▼
              hash-chained incident ledger (tamper-evident audit —
              risk, policy, action, model version; never audio)
```

**The three questions are deliberately separate:** A/B — *is this voice synthetic?* ·
C — *is this the claimed person?* · D/E — *is this call/request dangerous?* A voice
clone of the right person defeats C but not A/B; a real voice with a fraudulent
script defeats A/B but not E. Defense in depth is the design point — in our own
benchmark, layer B flags Windows SAPI TTS (p=0.98) but misses Microsoft Edge neural
TTS (p=0.08); layer A catches both.

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
cloned manager (neural TTS) → CRITICAL, transaction HOLD + liveness challenge +
supervisor alert; OTP-harvesting attacker (English) and Hindi attacker (Devanagari
intent rules fire) → CRITICAL, all actions + incident webhook. Every verdict prints
an explainable per-dimension breakdown (voice authenticity / speaker mismatch /
call context / social engineering).

## API

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | liveness + which detector layers are active |
| `POST /v1/enroll?speaker_id=X` | body = WAV/MP3 bytes → enroll caller voiceprint |
| `POST /v1/analyze?speaker_id=X&context={json}&transcript=T` | body = WAV/MP3 bytes → verdict + trajectory + explainable breakdown |
| `WS /v1/stream` | binary PCM16LE 16k-mono frames → per-chunk risk events; text `END` finalizes |
| `GET /v1/sessions/{id}` | replay any session verdict |
| `GET /v1/audit` | hash-chained incident ledger + chain integrity proof |
| `POST /v1/audit/{incident_id}/decision` | append supervisor decision (append-only) |

The dashboard at `/` supports microphone streaming, WAV file analysis, enrollment,
a transcript field (feeds the intent rules), and the incident ledger view — zero
build step, single HTML file.

## What is real vs. bootstrap (honest status)

| Component | Status |
|---|---|
| Fusion + risk engine + policy actions + explainable breakdown | **Working**, unit-tested |
| REST + WebSocket API + dashboard + incident ledger | **Working**, end-to-end tested |
| Layer A artifact CNN | **Working** — trained on the reproducible corpus (real human speech × 6 speakers + real neural TTS × 6 voices incl. Hindi + vocoder proxies) with augmentation and speaker-disjoint validation. **Honest numbers:** val acc 0.74 speaker-disjoint (bonafide 0.56 / spoof 0.84); corpus EER 0.18 (0.17 excluding the vocoder-proxy class) |
| Layer B SSL detector | **Working** — `Hemgg/Deepfake-audio-detection` (wav2vec2, ASVspoof-based); validated: human 0.03 / SAPI TTS 0.98 / Edge neural TTS 0.08 (misses modern neural TTS — documented; layer A covers it) |
| Layer C speaker consistency | **Working** — prosody+LFCC voiceprint, cosine distance; abstains without enrollment |
| Layer D context rules | **Working** — deterministic metadata risk rules |
| Layer E intent rules | **Working** — bilingual (EN/HI, Devanagari + romanized) social-engineering rule engine with per-rule explainability; transcript comes from ASR in production |
| Telephony robustness | **Measured** — `scripts/eval_telephony.py` sweeps clean → 8kHz narrowband → μ-law → noise → packet loss and reports the EER cost of each (layer A: 0.18 clean → 0.26 worst-case) |
| Audit chain | **Working** — hash-chained, tamper-evident incident ledger + supervisor decisions; chain verification unit-tested |
| ASVspoof 2019/2021/5 training | **Planned** — `train_artifact.py --asvspoof-dir` already accepts the ASVspoof2019 LA layout |
| Real 2026 clone corpus (F5-TTS, XTTS-v2, RVC, Fish Speech) + leave-one-generator-out eval | **Planned** — the core ML contribution; see `docs/architecture.md` |
| Indic-language ASR → live intent pipeline | **Planned** — AI4Bharat IndicVoices/Shrutilipi data already in the roadmap |

We deliberately ship the corpus generator instead of committed audio so every
number in this repo is reproducible from code — and we label which numbers are
bootstrap-level vs. which are claims we do not yet make. No benchmark number on
this tiny corpus is evidence of real-world performance; the speaker-disjoint and
cross-generator protocols exist so the numbers can't lie when the real data lands.

## Repository layout

```
swarakavach/            core package
  audio.py              I/O, resampling, chunking, VAD, PCM16, telephony degradation
  features.py           LFCC, F0 tracking, prosody vectors, speaker stats
  synth.py              formant synthesis + mel-vocoder clone proxy (data bootstrap)
  detectors/            A artifact · B ssl · C speaker · D context · E intent
  fusion.py             late fusion, calibration, EMA, median verdicts, breakdown
  audit.py              hash-chained tamper-evident incident ledger
  server.py             FastAPI REST + WebSocket + dashboard + audit endpoints
scripts/                corpus gen, training, benchmarks, telephony eval, demo, SSL check
web/index.html          live dashboard (zero-build)
tests/                  pytest suite (13 tests)
docs/                   architecture + demo runbook
```

## Team & license

SIH 2026 submission. Code: MIT. Model checkpoints keep their upstream licenses —
the SSL layer is a wav2vec2 fine-tune (Apache-2.0 base); any XTTS-v2-based clone
corpus tooling is **non-commercial (CPML) — training/research use only** and will
be flagged accordingly.

## Roadmap

1. **Week 1–2:** real training — ASVspoof 2019/2021 LA + own 2026 clone corpus
   (F5-TTS, XTTS-v2, OpenVoice 2, RVC, Fish Speech at 3 s/10 s enrollment),
   **leave-one-generator-out evaluation** as the headline ML claim: train on all
   clone tools except one, test on the held-out generator — that is the number
   that predicts in-the-wild performance.
2. **Week 3–4:** telephony-realistic augmentation (μ-law 8 kHz, AMR/EVS/Opus
   codec simulation, MUSAN noise, RIR reverb, packet loss) folded into training
   — the eval_telephony deltas show exactly what it buys.
3. **Week 5–6:** Indic language data (AI4Bharat IndicVoices/Shrutilipi/Kathbath),
   ASR → live intent pipeline, streaming latency budget (p95 < 300 ms/chunk),
   gRPC + Twilio Media Streams demo.
4. **Week 7–8:** ONNX edge export (< 10 MB bundle), dashboard polish, demo freeze.
