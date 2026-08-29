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

## Fusion & policy
`swarakavach/fusion.py`

1. Weighted late fusion over **available** layers (missing → renormalize):
   A 0.55 · B 0.30 · C 0.05 · D 0.10 (bootstrap-calibrated; see Calibration).
2. Temperature-softened sigmoid calibration (T = 1.3) — the fused mean is treated
   as a logit-of-evidence, softening overconfident single layers.
3. EMA smoothing (α = 0.35) → the **dynamic risk trajectory** the PS asks for.
4. Levels: LOW < 0.25 ≤ MID < 0.45 ≤ HIGH < 0.65 ≤ CRITICAL.
5. Actions: CAUTION_PROMPT, STEP_UP_MFA (MID) · HOLD_TRANSACTION, LIVENESS_CHALLENGE,
   SUPERVISOR_ALERT (HIGH) · + INCIDENT_WEBHOOK (CRITICAL). Final verdict takes
   max(final, 0.85 × peak) — an early spike is never forgotten by trailing audio.

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

## Evaluation targets

| Metric | Target |
|---|---|
| EER on ASVspoof 2021/5 LA (dev) | < 2% |
| EER on in-the-wild 2024+ deepfake sets | < 8% |
| EER leave-one-tool-out (own corpus) | < 6% |
| p95 chunk latency | < 300 ms |
| Edge bundle (ONNX, layer A) | < 10 MB |
