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
| Layer A artifact CNN | **Working, codec-trained, multilingual** — trained on real human speech (16 curated + 40 LibriSpeech + **9 real Hindi speaker groups**) + 4 attack families (Edge, Piper, SAPI, **XTTS-v2 zero-shot clones**) + AMR-NB/Opus codec copies. Honest numbers: speaker-disjoint val 0.80; demo-corpus fused EER **0.06**; telephony worst-case fused EER **0.14**; Hindi false-alarm rate on real Hindi speech: **12.9%** (was 44.6% before Hindi training data) with Hindi TTS catch-rate still **1.00** |
| **Leave-one-generator-out** | **Measured, two-sided protocol** — `scripts/eval_loto.py`: for each family, train on all other corpora, test on the held-out family's attacks **plus never-seen human speakers** (both classes unseen — the deployment condition). **5 attack families incl. the Indian TTS (Sarvam Bulbul v3): mean held-out EER 0.225** — with **sarvam held-out at 0.082**: when the Indian TTS is excluded from training, the detector still catches it at ~8% EER (the formal version of our live 'caught it unseen' demo). Neural-vocoder families (edge/xtts) remain hardest (~0.3); classic SAPI and Hindi-Indian neural TTS hold out best (~0.08–0.09). Fold-level run-to-run variance is documented in `experiments/LOG.md` (EXP-8) — along with the protocol's own audit trail (an earlier 0.113 reading was flattered by in-training humans in the test set; we fixed it and publish the honest number). |
| Layer B SSL detector | **Working** — `Hemgg/Deepfake-audio-detection` (wav2vec2, ASVspoof-based); validated: human 0.03 / SAPI TTS 0.98 / Edge neural TTS 0.08 (misses modern neural TTS — documented; layer A covers it) |
| Layer C speaker consistency | **Working** — prosody+LFCC voiceprint, cosine distance; abstains without enrollment |
| Layer D context rules | **Working** — deterministic metadata risk rules |
| Layer E intent rules | **Working — fed by live ASR** — bilingual (EN/HI, Devanagari + romanized) social-engineering rule engine with per-rule explainability. ASR: whisper medium fp16 CUDA (RTF 0.11), auto-detects Hindi/English; Devanagari keywords match fuzzily against ASR-mangled TTS Hindi (edit distance ≤2 + homophone normalization) — the Hindi attacker demo fires all four rule categories on real ASR output |
| Telephony robustness | **Trained-for + measured** — real ffmpeg codec pipelines (AMR-NB 4.75k cellular, Opus 8k/16k VoIP, packet loss, EVS-style profile) encoded into training. Fused EER: clean 0.06 → worst case **0.14** |
| Audit chain | **Working** — hash-chained, tamper-evident incident ledger + supervisor decisions; chain verification unit-tested |
| ASVspoof 2019/2021/5 training | **Planned** — `train_artifact.py --asvspoof-dir` already accepts the ASVspoof2019 LA layout |
| Real 2026 clone corpus (F5-TTS, RVC, Fish Speech) + broader LOTO | **Next** — 4 generator families implemented (Edge, Piper, SAPI, XTTS-v2); adding F5-TTS/RVC strengthens the LOTO claim further |
| Indic-language ASR → live intent pipeline | **Working** — `swarakavach/asr.py` (faster-whisper, whisper medium): REST auto-transcribes when no transcript is given; WS path uses a rolling 4 s window transcriber streaming `transcript_delta` events; the finale demo runs entirely on live ASR |
| Real Hindi human speech in training | **Working** — `scripts/prepare_hindi.py`: 9 Hindi speaker groups (603 chunks + codec copies) via HF datasets-server; closed a 44.6% Hindi false-alarm hole |
| Full-system ablation | **Measured** — `scripts/eval_ablation.py`: layer-by-layer scenario table (A alone already flags the synthetic attacks; B/C/D/E each refine the verdict — and layers E+D are what catch the *real-human* social-engineering call, not just deepfakes) + corpus EER ablation. The 'five independent questions' claim is experimentally grounded |
| Hindi false-alarm, honest split | **chunk-level 12.9%** (p90 spikes on unseen speakers) but **call-level 0.0%** — 0/27 real Hindi calls reached even MID under the median decision rule; no legitimate Hindi transaction gets held. Chunk spikes decay; the decision rule is call-median |
| Edge ONNX deployment | **Shipped, measured** — `scripts/export_onnx.py` + `swarakavach/edge.py`: torch-free edge path (numpy LFCC with exact torchaudio parity — max diff 4e-4 — + onnxruntime). **Bundle 0.49 MB** vs the < 10 MB target (fp32 416 KB + int8 115 KB optional + LFCC constants 132 KB); score parity vs torch 0.00001; ~2 ms/chunk on CPU. Layer A runs on-device, B/C/E stay server-side by design |
| Twilio Media Streams gateway | **Working, verified on simulated frames** — `scripts/twilio_gateway.py`: Twilio mulaw 8kHz WS frames → decode → upsample → same fusion engine (genuine MID / clone HIGH / Hindi attacker CRITICAL in offline simulation); needs a public tunnel + Twilio number for the live-call demo |
| Streaming latency | **Measured** — `scripts/eval_latency.py`: fused chunk pipeline **p95 = 108 ms** (target < 300 ms; artifact 2 ms, speaker 5 ms, wav2vec2 85 ms dominates); ASR RTF 0.11 |

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

## Roadmap — status

**Done (all measured, scripts in repo):**

1. **Corpus + generalization protocol** — **5 attack generator families** (Edge/Piper/SAPI/**Sarvam Bulbul v3 — the Indian TTS**/XTTS-v2 zero-shot clones) + real human diversity (16 curated + 40 LibriSpeech + 9 Hindi speaker groups), leave-one-generator-out eval (**mean held-out EER 0.225**, two-sided; **sarvam held-out 0.082** — Indian TTS caught at ~8% EER when excluded from training), speaker-disjoint validation.
2. **Codec-aware training** — real AMR-NB/Opus ffmpeg pipelines in training; worst-case channel fused EER 0.32 → **0.14**.
3. **Live ASR intent pipeline** — whisper medium (RTF 0.11), bilingual fuzzy intent rules; **fused chunk pipeline p95 = 108 ms** against the 300 ms target (artifact 2 ms · speaker 5 ms · wav2vec2 85 ms · rest ~16 ms).
4. **Telephony integration** — Twilio Media Streams gateway verified on simulated frames.

**Next (in priority order):**

5. ~~**Full-system ablation**~~ — **done**: `scripts/eval_ablation.py` (layer-subset scenario table + corpus EER; the 'five independent questions' claim is measured).
6. **Live Twilio phone-call demo** — gateway is ready; needs a public tunnel + Twilio number.
7. **More generator families** (F5-TTS, RVC, OpenVoice) — each strengthens the LOTO claim.
8. **ONNX edge export** — **done**: 0.49 MB torch-free bundle, parity-guarded in tests.
9. **Demo freeze** — this is it: the 5-call finale below is the demo script.

ASVspoof 2019/2021 LA is optional supporting evidence (65 GB, storage-gated) — it strengthens the story but does not gate it.
