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

# 1. demo corpus + scenario (public human speech, neural TTS attacks)
python scripts/make_demo_data.py
# 2. attack generator families for LOTO (Edge/Piper/SAPI/XTTS-v2; GPU helps)
python scripts/make_clone_corpus.py
# 3. 40-speaker LibriSpeech bonafide diversity
python scripts/prepare_libri.py
# 4. train layer A on everything (speaker-disjoint validation)
python scripts/train_artifact.py --labels data/demo/labels.csv \
    --extra-labels data/clone-corpus/chunks/labels.csv,data/libri-human/chunks/labels.csv
# 5. four-call finale demo (genuine -> REAL XTTS clone -> OTP attacker -> Hindi)
python scripts/run_demo.py
# 6. honest evaluations
python scripts/eval_loto.py            # leave-one-generator-out (the core claim)
python scripts/eval_benchmarks.py      # per-layer + fused EER, per-class
python scripts/eval_telephony.py       # channel degradation sweep
# 7. API + live dashboard
uvicorn swarakavach.server:app --port 8000     # open http://localhost:8000
```

**Expected demo output:** genuine manager call → LOW, transaction proceeds;
cloned manager (**a real XTTS-v2 zero-shot clone of the enrolled manager's
voice**) → CRITICAL, transaction HOLD + liveness challenge + supervisor alert;
OTP-harvesting attacker (English) and Hindi attacker (Devanagari intent rules
fire) → CRITICAL, all actions + incident webhook. Every verdict prints an
explainable per-dimension breakdown (voice authenticity / speaker mismatch /
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
| Layer A artifact CNN | **Working, codec-trained** — trained on the reproducible corpus: real human speech (16 curated + 40 LibriSpeech speakers) + 4 attack families (Edge, Piper, SAPI, **XTTS-v2 zero-shot clones**) + **AMR-NB/Opus codec copies of everything**. Honest numbers: speaker-disjoint val 0.78 (bonafide 0.88 / spoof 0.72); demo-corpus fused EER **0.07**; **LOTO mean held-out EER 0.13**; telephony worst-case fused EER **0.19** (was 0.32 before codec training) |
| **Leave-one-generator-out** | **Measured** — `scripts/eval_loto.py`: for each family, train on the other three + humans, test on the held-out family. This is the protocol that predicts in-the-wild performance and the core ML claim. |
| Layer B SSL detector | **Working** — `Hemgg/Deepfake-audio-detection` (wav2vec2, ASVspoof-based); validated: human 0.03 / SAPI TTS 0.98 / Edge neural TTS 0.08 (misses modern neural TTS — documented; layer A covers it) |
| Layer C speaker consistency | **Working** — prosody+LFCC voiceprint, cosine distance; abstains without enrollment |
| Layer D context rules | **Working** — deterministic metadata risk rules |
| Layer E intent rules | **Working** — bilingual (EN/HI, Devanagari + romanized) social-engineering rule engine with per-rule explainability; transcript comes from ASR in production |
| Telephony robustness | **Trained-for + measured** — real ffmpeg codec pipelines (AMR-NB 4.75k cellular, Opus 8k/16k VoIP, packet loss, EVS-style profile) are encoded into training corpora (`scripts/make_codec_corpora.py`). Fused EER: clean 0.08 → worst case 0.19 (pre-codec-training it was 0.15 → 0.32) |
| Audit chain | **Working** — hash-chained, tamper-evident incident ledger + supervisor decisions; chain verification unit-tested |
| ASVspoof 2019/2021/5 training | **Planned** — `train_artifact.py --asvspoof-dir` already accepts the ASVspoof2019 LA layout |
| Real 2026 clone corpus (F5-TTS, RVC, Fish Speech) + broader LOTO | **Next** — 4 generator families implemented (Edge, Piper, SAPI, XTTS-v2); adding F5-TTS/RVC strengthens the LOTO claim further |
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

1. **Week 1–2:** done for the corpus (Edge/Piper/SAPI/XTTS-v2 + LibriSpeech diversity + LOTO protocol); next: add F5-TTS, RVC, OpenVoice — every new family strengthens leave-one-generator-out eval. ASVspoof 2019 LA remains the baseline anchor when storage allows (65 GB).
2. **Week 3–4:** telephony-realistic augmentation (μ-law 8 kHz, AMR/EVS/Opus
   codec simulation, MUSAN noise, RIR reverb, packet loss) folded into training
   — the eval_telephony deltas show exactly what it buys.
3. **Week 5–6:** Indic language data (AI4Bharat IndicVoices/Shrutilipi/Kathbath),
   ASR → live intent pipeline, streaming latency budget (p95 < 300 ms/chunk),
   gRPC + Twilio Media Streams demo.
4. **Week 7–8:** ONNX edge export (< 10 MB bundle), dashboard polish, demo freeze.
