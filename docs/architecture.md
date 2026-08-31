# Architecture

## Threat model

An attacker calls a victim (bank agent, family member, employee at a payment desk)
using a real-time voice clone of a trusted person, and pressures for an urgent
transfer. The clone may be:
- a **zero-shot TTS render** (ElevenLabs, XTTS-v2, F5-TTS) from a short sample,
- a **voice-conversion clone** (RVC, OpenVoice) in a live loop,
- a plain **different-voice TTS imposter** (social engineering, no clone at all).

Defender constraints: decision must be **streaming** (per-chunk, ~1 s), robust to
telephone band and codecs, and never block genuine callers — missed-detection and
false-alarm both cost money.

## Detector layers

### A — Spectral artifact detector (LFCC + 1D CNN)
`swarakavach/detectors/artifact.py`, trained by `scripts/train_artifact.py`.

- 20-coefficient LFCC spectrogram (per-chunk, per-coefficient normalized),
- 4-layer 1D CNN (~60k params, ~240 KB fp32) → softmax over {bonafide, spoof},
- cheap enough for edge/ONNX export; class-weighted loss for imbalanced corpora,
- accepts either the bootstrap `labels.csv` or the **ASVspoof 2019 LA** protocol
  layout via `--asvspoof-dir`.

### B — SSL deepfake classifier (wav2vec2)
`swarakavach/detectors/ssl_detector.py`.

- HuggingFace audio-classification checkpoint, lazy-loaded, CPU inference in a
  worker thread (server) so the event loop never blocks,
- spoof index auto-detected from `id2label` (keywords: spoof/fake/aivoice/synthetic),
- **validated empirically** (`scripts/verify_ssl_mapping.py`): human 0.03,
  SAPI TTS 0.98, Edge neural TTS 0.08. The Edge miss is exactly why fusion exists —
  layer A catches what layer B misses on modern neural TTS.

### C — Speaker consistency (enrolled voiceprint)
`swarakavach/detectors/speaker.py`.

- Enrollment stores a 26-dim voiceprint: 6 prosody stats (F0 mean/std, jitter,
  shimmer proxy, speech ratio, pause density — normalized) + 20-dim LFCC mean,
- per-chunk cosine similarity → risk via the SPEAKER_COS_REF / SPAN calibration,
- **abstains (None) when the caller is not enrolled** — fusion renormalizes over
  available layers, so anonymous-call analysis still works,
- voice-conversion clones preserve prosody → layer C stays low; that is correct
  behavior: C answers "is this the enrolled voice", not "is it synthetic".

### D — Context risk engine
`swarakavach/detectors/context.py` — deterministic, auditable rules over call
metadata: caller-ID/SIM mismatch, enrolled-contact status, known-fraud flag,
international origin, transaction amount (scaled at ₹50 L), call frequency.
Returns None when no metadata is provided.

### E — Conversation-intent risk
`swarakavach/detectors/intent.py` — the fourth question: **is what they are
asking for dangerous?** Rule-based, bilingual (English + Hindi, romanized and
Devanagari) categories: fund transfer, credentials/OTP harvesting, urgency,
secrecy, authority impersonation. Every score carries the matched rules and
keywords (explainability payload). Transcript input comes from ASR in production
(honest labeling: the demo passes the transcript explicitly); the keyword rules
are auditable by design — swap in a fine-tuned IndicBERT classifier if keyword
precision becomes the ceiling.

### Codec-aware training (week 3, done)

`swarakavach/codecs.py` + `scripts/make_codec_corpora.py`: every training clip
is encoded through REAL ffmpeg codec pipelines (AMR-NB 4.75 kbps — the worst
cellular rate — plus Opus 8k/16k VoIP), materialized once on disk and trained
alongside the clean copies. The detector learns codec artifacts as part of
'bonafide channel noise' instead of confusing them with spoof artifacts.
Measured effect: worst-case channel fused EER halves (0.32 → 0.19). NB: this
ffmpeg build has no AMR-WB/EVS encoders — Opus profiles cover wideband VoIP
and the EVS-style level is a documented approximation (AMR-NB + noise).

### Incident audit chain
`swarakavach/audit.py` — every HIGH/CRITICAL verdict appends a hash-chained
record (SHA-256, each record embeds the previous hash) to an append-only JSONL
ledger: incident ID, risk, level, model version, policy triggered, transaction
action, supervisor decision, timestamp. **Metadata only — never audio, never
transcripts** (privacy). `verify_chain()` detects any after-the-fact edit; the
tamper test is unit-tested. This is the legitimate blockchain-shaped component
for the theme: tamper-evident fraud forensics, not decorative crypto.

## Fusion & policy
`swarakavach/fusion.py`

1. Weighted late fusion over **available** layers (missing → renormalize):
   A 0.50 · B 0.28 · C 0.05 · D 0.09 · E 0.08 (bootstrap-calibrated; see Calibration).
2. Temperature-softened sigmoid calibration (T = 1.3) — the fused mean is treated
   as a logit-of-evidence, softening overconfident single layers.
3. EMA smoothing (α = 0.35) → the **dynamic risk trajectory** the PS asks for.
4. Levels: LOW < 0.25 ≤ MID < 0.45 ≤ HIGH < 0.65 ≤ CRITICAL. The verdict level
   comes from the **median** of the trajectory — robust to single-chunk detector
   wobble on genuine calls (a one-chunk CNN false positive cannot block a
   transaction), while sustained attack patterns still drive the level up. Peak
   is kept as an informational/alert field.
5. Actions: CAUTION_PROMPT, STEP_UP_MFA (MID) · HOLD_TRANSACTION, LIVENESS_CHALLENGE,
   SUPERVISOR_ALERT (HIGH) · + INCIDENT_WEBHOOK (CRITICAL). HIGH/CRITICAL also
   append a hash-chained incident record to the audit ledger.

## Explainability

Every verdict carries a **breakdown** — the judge-facing risk decomposition the
policy acted on (0–100 per dimension):

```
voice authenticity      80   ################ voice is synthetic
speaker mismatch        38   #######           not the enrolled person
call context            55   ###########       suspicious call setup
social engineering      70   ##############    dangerous request
────────────────────────────
FINAL RISK               86   CRITICAL → HOLD_TRANSACTION
```

A/B average → voice authenticity; C → speaker mismatch; D → call context;
E → social engineering. The demo prints this per call; the dashboard and
`/v1/analyze` return it in JSON.

## Calibration

All constants (layer weights, temperature, EMA α, speaker cosine ref/span) live in
`swarakavach/config.py` and are **bootstrap-calibrated on the reproducible demo
corpus**. They are placeholders for proper calibration on ASVspoof 2021/5 dev
sets (isotonic/temperature scaling per layer before weight fitting). Every number
in this repo is reproducible from code — no committed audio.

## Data plan (real training)

| Data | Use |
|---|---|
| ASVspoof 2019 LA (Zenodo) | layer A/B training baseline — loader already implemented |
| ASVspoof 2021 LA / 2025 (Zenodo 14498691) | codec/narrowband robustness eval |
| Own 2026 clone corpus: F5-TTS, XTTS-v2 (CPML, non-commercial), OpenVoice 2, RVC, Fish Speech, ElevenLabs — 3 s and 10 s enrollment settings | the actual threat; **leave-one-tool-out** evaluation = the generalization moat |
| AI4Bharat IndicVoices / Shrutilipi / Kathbath | Indic-language bonafide coverage |
| MUSAN + RIR convolutions + μ-law 8 kHz + AMR/EVS/Opus codec simulation + packet loss | telephony-realistic augmentation |

## Streaming & latency

- 1 s chunks, 0.5 s overlap (config). WS path: audio buffer → full chunk → all
  layers → event JSON (< 300 ms p95 target on GPU, CPU acceptable for demo).
- REST `/v1/analyze` runs the same engine over a complete file — identical code
  path for batch and streaming.
- Session store is in-memory (dict); production swap = Redis (single-line change
  behind `SESSIONS`).

## Evaluation — honest protocols

The numbers below are **measured on our own 2026 clone corpus**, not copied
from anyone's paper:

- **Leave-one-generator-out (LOTO)** (`scripts/eval_loto.py`): for each attack
  family, train layer A on the other three families + human speech, test on the
  held-out family. **Mean held-out EER 0.11** (edge 0.16 · piper 0.03 · sapi 0.06 ·
  xtts 0.21). A detector that generalizes to generators it never saw is the
  whole point — a lab-only benchmark never tests this.
- **Speaker-disjoint validation** (`scripts/train_artifact.py`): whole speaker
  groups held out. Val acc 0.85 (bonafide 0.91 / spoof 0.57) on unseen speakers
  — the bonafide side is what protects genuine callers from false blocks.
- **Per-class EER** (`scripts/eval_benchmarks.py`): fused EER 0.10 excluding the
  vocoder-proxy class (a corpus training variety, not a claimed capability).
- **Telephony degradation sweep** (`scripts/eval_telephony.py`): REAL ffmpeg
  codec roundtrips (AMR-NB 4.75k, Opus 8k/16k, packet loss, EVS-style profile).
  Codec-trained fused EER: clean 0.08 → worst case 0.19 (the same model before
  codec training measured 0.15 → 0.32 — codec-aware training halves the
  channel penalty).
- **Known limitation, documented not hidden:** the SSL layer misses modern Edge
  neural TTS at call level (0.08); the artifact layer covers it. Fusion exists
  precisely because single layers fail differently.

| Metric | Current (own corpus) | Target |
|---|---|---|
| LOTO mean held-out EER | **0.11** | < 0.06 (more generator families) |
| Speaker-disjoint val acc | **0.78** (bonafide 0.88) | > 0.90 (more human speakers) |
| Fused EER (demo corpus) | **0.07** | — |
| Telephony worst-case fused EER | **0.19** (pre-codec: 0.32) | < 0.10 |
| p95 chunk latency | — | < 300 ms |
| Edge bundle (ONNX, layer A) | — | < 10 MB |
