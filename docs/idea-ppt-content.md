# SwaraKavach — SIH Idea Presentation Content (PS26104)

**Purpose:** every field of the official SIH PPT template, filled and paste-ready.
**Template rules (from the template's own instructions slide):**
- Max **6 slides INCLUDING the title slide**
- **Points / diagrams / infographics — NO paragraphs**
- Precise and easy to understand
- Upload as **PDF only** on the portal (no PPT/Word)
- Delete the instructions slide before uploading

---

## SLIDE 1 — TITLE PAGE

| Field | Value |
|---|---|
| Problem Statement ID | **PS26104** |
| Problem Statement Title | **AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks** |
| Theme | **Blockchain & Cybersecurity** |
| PS Category | **Software** |
| Organization / Department | **AICTE — Cyber Security Cell** |
| Team ID | **[FILL: from SIH portal after idea registration]** |
| Team Name (Registered on portal) | **[FILL: exact name as registered]** |

> Suggested title slide subtitle: **SwaraKavach — Real-Time Voice Clone Fraud Prevention for India's Call Channels**
> (SwaraKavach = "voice armor" in Sanskrit)

---

## SLIDE 2 — IDEA TITLE: Proposed Solution

### One-liner (put at top of slide)
**SwaraKavach listens to every live call, scores voice-clone fraud risk every second, and BLOCKS the fraudulent transaction before it clears — in Hindi and English, on real phone-call quality, in 108 milliseconds.**

### The solution (points)
- **5 independent AI detectors — one question each** (no single point of failure):
  - **A — Artifact CNN:** "Is this voice synthetic?" — 105k-param spectral fingerprint model, runs on-device (0.49 MB)
  - **B — SSL wav2vec2:** "Is this voice synthetic?" — second opinion from a pretrained speech model
  - **C — Speaker consistency:** "Is this the CLAIMED person?" — enrolled voiceprint vs live voice
  - **D — Context engine:** "Is this call risky?" — caller-ID mismatch, transaction value, velocity rules
  - **E — Intent engine:** "Is the request dangerous?" — live ASR transcript → bilingual (Hindi + English) social-engineering rules: OTP/credential harvesting, urgent transfers, secrecy demands
- **Fusion + policy:** calibrated risk score (LOW / MID / HIGH / CRITICAL) →
  **transaction HOLD · liveness challenge (random phrase) · MFA step-up · supervisor alert · incident webhook**
- **Tamper-evident audit ledger** (hash-chained, blockchain-style): every blocked call = un-editable forensic record — risk score, policy triggered, timestamp, supervisor decision. **Stores metadata only — never audio** (privacy by design, DPDP-aligned)
- **Explainable verdicts:** every decision shows WHY (voice authenticity 80 · speaker mismatch 38 · context 55 · social engineering 95 → FINAL 84 CRITICAL)

### How it addresses the problem (points — map to PS26104's asks)
- PS asks for **detection DURING an ongoing call** → we score every 1-second chunk, p95 = **108 ms** (3× under the 300 ms real-time target)
- PS asks for **prevention, not just detection** → transaction HOLD is a policy action wired into the payment flow
- PS asks for **robustness to real telephony** → trained on real AMR-NB (cellular) and Opus (VoIP) codec simulations via ffmpeg
- PS asks for **multilingual support** → Hindi + English end-to-end: Hindi ASR, Devanagari intent rules, real Hindi speakers in training

### Innovation & uniqueness (points)
- **Leave-one-generator-out evaluation** — we train WITHOUT one attack tool and still catch it (mean held-out EER 0.20). In the wild, attackers use tools we've never seen — this is the only protocol that measures that.
- **Caught an Indian TTS it had never seen** — live-tested against Sarvam AI Bulbul v3 (Made-in-India TTS): our detector flagged it (0.79 mean) with zero retraining
- **Tested against REAL zero-shot clones** — XTTS-v2 clones of our own enrolled speakers (the exact attack in the PS), plus Edge neural TTS, Piper, SAPI, Sarvam — 5 attack generator families
- **Honest evaluation, published** — speaker-disjoint validation, per-class metrics, full audit trail in the repo (we even corrected our own benchmark when we found it flattering)
- **Small-AI edge design** — 0.49 MB torch-free ONNX model runs inside the bank's wall; audio never leaves the premises

---

## SLIDE 3 — TECHNICAL APPROACH

### Technologies used (points)
- **Languages/frameworks:** Python, PyTorch, FastAPI (REST + WebSocket), NumPy/SciPy
- **AI models:** wav2vec2 (speech SSL), custom 105k-param CNN (ONNX Runtime for edge), faster-whisper (Hindi/English ASR), rule + fusion engine
- **Telephony:** Twilio Media Streams gateway (μ-law 8 kHz PCM), ffmpeg codec simulation (AMR-NB / Opus)
- **Attack corpus tooling:** XTTS-v2, Edge neural TTS, Piper, SAPI, Sarvam Bulbul v3 (attack generation for training)
- **Frontend:** zero-build live dashboard (risk gauge, per-layer bars, live transcript, incident ledger)

### Methodology / flow (describe as diagram)
```
LIVE CALL (Twilio / WebRTC / file)
        ↓  16 kHz PCM chunks, 1 s windows
[A] Spectral CNN ──┐
[B] wav2vec2 ──────┤   each layer answers ONE question
[C] Voiceprint ────┤→  WEIGHTED FUSION → calibration →
[D] Context rules ─┤   EMA risk trajectory → level
[E] ASR → intent ──┘        ↓
                     POLICY ENGINE: HOLD / MFA / LIVENESS / ALERT / WEBHOOK
                             ↓
                HASH-CHAINED AUDIT LEDGER (tamper-evident, metadata only)
```

### Working prototype — measured numbers (all reproducible from repo)
- Fused chunk pipeline **p95 = 108 ms** (target < 300 ms)
- Edge bundle **0.49 MB** (target < 10 MB) — torch-free ONNX + numpy
- Channel robustness: trained on real codecs — worst-case error halved (0.32 → 0.14)
- Hindi call-level false alarms: **0.0%** (real Hindi speakers never blocked)
- Generalization: catches unseen TTS generators (LOTO mean EER 0.20; Sarvam caught live, untrained)
- 14/14 automated tests; live 5-call demo (genuine → clone → human social engineer → EN attacker → Hindi attacker)

---

## SLIDE 4 — FEASIBILITY AND VIABILITY

### Feasibility (points)
- **Working prototype TODAY** — not a concept: API + dashboard + Twilio gateway + edge model, all in the repo, every metric reproducible from code
- **Runs on commodity hardware** — laptop GPU/CPU; edge layer needs no GPU at all
- **No dependency on external audio data** at deployment — models ship with the system
- ₹100 Sarvam free credits / open-source stack = **near-zero build cost**

### Potential challenges & risks → Strategies (table)
| Risk | Strategy |
|---|---|
| New clone generators appear constantly | Leave-one-generator-out training + corpus refresh pipeline (add a family in one command); live-proven: caught Sarvam unseen |
| False alarms could block genuine customers | Median decision rule (call-level FA 0.0% on real Hindi/EN speakers); first response is non-blocking (MFA step-up), HOLD only on sustained high risk |
| Large SSL models memorize speakers at small data scale (measured negative result) | Deliberately small 105k edge CNN (measured: bigger = worse generalization); ASVspoof-scale data = documented upgrade path |
| Privacy / regulatory (DPDP, RBI) | On-device layer A; **no audio stored or transmitted out** — ledger holds metadata only; hash-chained for tamper evidence |
| Model drift over time | Monthly recalibration protocol + drift dashboard (roadmap, ownership defined) |
| Adoption friction in banks | Drops into existing telephony (Twilio/SIP) — no change to the caller's experience |

### Viability (points)
- **Market exists:** real-time voice analytics (L4) sells at **₹1.50–4.50 per analyzed minute** in India; buyers = banks, NBFCs, insurance, enterprise finance desks
- **Economics:** one prevented ₹50 lakh fraud > annual cost of the system; ₹22,495 crore lost in 2025 alone = the addressable pain
- **Compliance pull:** RBI's real-time fraud-monitoring directives push banks toward exactly this control layer

---

## SLIDE 5 — IMPACT AND BENEFITS

### The scale of the problem (sourced points — put as infographic)
- **₹22,495 crore** lost to cyber fraud in India in **2025** — up 24% over 2024 (I4C/MHA data, 28.15 lakh cases)
- **₹52,976 crore** lost over the last 6 years (I4C)
- **47% of Indian adults** have been a victim of — or know a victim of — an **AI voice-cloning/deepfake scam** (McAfee 2025; ~2× the global average of 25%)
- **83% of AI voice-scam victims in India lost money; ~half lost over ₹50,000** (McAfee 2025)
- Real cases: Indore teacher ₹1L via cloned cousin's voice (Jan 2026, MP's first confirmed AI voice-clone case) · Hyderabad homemaker ₹1.97L · corporate employee ₹55L after a WhatsApp voice impersonation of the owner

### Impact on target audience (points)
- **Banks / fintechs / NBFCs:** fraud blocked BEFORE the transaction clears — not investigated after; forensic ledger supports LEA handover
- **Ordinary citizens:** elderly and vulnerable groups shielded from family-impersonation scams; genuine callers are never blocked (0.0% call-level false alarms)
- **Law enforcement (I4C/cyber cells):** every blocked call produces a tamper-evident evidence record — currently almost impossible to obtain
- **Enterprises:** CEO/CFO impersonation ("urgent vendor payment") intercepted at the call layer

### Benefits (points)
- **Social:** restores trust in the phone channel; protects those least able to detect clones (elderly, low digital literacy)
- **Economic:** direct loss prevention at ₹/crore scale; supports RBI fraud-directive compliance; new "voice fraud intelligence" product layer for Indian BFSI
- **Strategic (Atmanirbhar):** tested against Made-in-India attacks (Sarvam Bulbul), Hindi-first design, deployable fully on-prem — no foreign cloud dependency
- **Environmental:** software-only; 0.49 MB edge model runs on existing infrastructure — negligible incremental energy footprint

---

## SLIDE 6 — RESEARCH AND REFERENCES

### Project
- Working prototype + all measured metrics: **https://github.com/INSANE0777/SIH-2026**
- Problem statement: SIH 2026 Portal — PS26104 (AICTE, Cyber Security Cell)

### Statistics & studies (impact slide sources)
- ₹22,495 crore lost to cyber fraud 2025, 24% YoY spike — I4C data via ThePrint/Reuters: theprint.in/india/cybercrime-saw-24-spike-in-2025-indians-lost-rs-22495-crore-mainly-in-investment-scams/2859930
- McAfee 2025 AI voice scam study (47% exposure, 83% monetary loss) — via ORF: orfonline.org/expert-speak/deepfakes-and-financial-cybercrime-india-s-multi-layered-response
- Digital arrest scam data (₹1,918 crore in 2024, declining in 2025) — MHA via ThePrint: theprint.in/india/governance/after-465-spike-in-2024-mha-data-shows-digital-arrest-scams-are-on-a-decline-in-india/2857002
- AI voice scam case reports — Indian Express (Hyderabad ₹1.97L): indianexpress.com/article/technology/artificial-intelligence/ai-scams-surge-in-india-voice-cloning-deepfakes-and-otp-frauds-leave-victims-helpless-10232064; News18 (Indore, Jan 2026): news18.com/india/scammers-clone-cousins-voice-using-ai-trick-indore-teacher-out-of-rs-1-lakh-ws-l-9822675.html

### Technical foundations
- wav2vec2 (speech representation learning) — Facebook AI Research
- ASVspoof challenge corpora (anti-spoofing benchmark) — zenodo.org
- XTTS-v2 (voice cloning, CPML non-commercial — used for attack corpus only) — Coqui
- Sarvam AI Bulbul v3 & Saaras v4 (Indian TTS/ASR used in testing) — docs.sarvam.ai
- faster-whisper (Indic-capable ASR) — OpenAI Whisper, CTranslate2 runtime
- ONNX Runtime (edge deployment)

---

## FINAL CHECKLIST BEFORE UPLOAD
- [ ] Team ID + Team Name filled (from portal) on slide 1
- [ ] 6 slides total (title included) — delete the instructions slide
- [ ] No paragraphs — points / diagram / infographic only
- [ ] Convert to **PDF** before upload
- [ ] Verify every number matches the repo (they do as of commit 47b7e20 — if you retrain the model, re-run `python scripts/eval_benchmarks.py` and update)
