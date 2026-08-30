# Demo runbook (finale rehearsal)

Full end-to-end demo of SwaraKavach against four calls. Everything below is
reproducible from a clean clone in ~5 minutes.

## 0. Setup

```bash
pip install -r requirements.txt && pip install -e .
python scripts/make_demo_data.py      # corpus + scenario files (needs network once)
python scripts/train_artifact.py      # layer A training (~2 min CPU, speaker-disjoint val)
```

## 1. Four-call demo (the finale script)

```bash
python scripts/run_demo.py
```

| Call | Audio | Transcript intent rules | Expected verdict | Actions |
|---|---|---|---|---|
| Genuine manager | real human speech | none matched | **LOW** (peak ≈ 0.21) | none — transaction proceeds |
| Cloned manager | neural TTS claiming to be the manager | transfer, urgency, authority | **CRITICAL** (peak ≈ 0.86) | HOLD, LIVENESS, SUPERVISOR, WEBHOOK |
| OTP attacker (English) | neural TTS, different voice | transfer, **credentials**, urgency, authority | **CRITICAL** (peak ≈ 0.84) | all actions |
| Hindi attacker | hi-IN neural TTS, Devanagari | ट्रांसफर/लाख, तुरंत, किसी से न, प्रबंधक | **CRITICAL** (peak ≈ 0.85) | all actions |

Each verdict prints the **explainable breakdown** — voice authenticity, speaker
mismatch, call context, social engineering (0–100) — so the audience sees *why*
the system acted, not just a score.

Talking points:
1. The SSL layer **misses** modern Edge neural TTS (p≈0.08) but the fused verdict
   is still CRITICAL — show `layer_means` in the verdict to prove defense-in-depth.
2. The Hindi call fires the bilingual intent rules — the multilingual requirement
   of the PS, live, in Devanagari.
3. "SwaraKavach doesn't just detect a fake voice; it **prevents** the high-risk
   action" — transaction HOLD is a policy action, not an alert.

## 2. Live API + dashboard

```bash
uvicorn swarakavach.server:app --port 8000
# open http://localhost:8000
```

Dashboard: enroll a speaker, paste a transcript (feeds the intent rules),
analyze any file or stream the microphone live — risk gauge, per-layer bars
labeled with their question (synthetic? this person? dangerous request?),
trajectory sparkline, action chips, and the **incident ledger** panel.

REST equivalent:

```bash
curl -X POST "http://localhost:8000/v1/enroll?speaker_id=regional_manager" \
     --data-binary @data/demo/scenario/manager_real.wav
curl -X POST "http://localhost:8000/v1/analyze?speaker_id=regional_manager&transcript=approve%20the%20fifty%20lakh%20transfer%20immediately" \
     --data-binary @data/demo/scenario/manager_clone.wav
```

The analyze response includes the breakdown. HIGH/CRITICAL verdicts append an
incident to the **hash-chained ledger**:

```bash
curl http://localhost:8000/v1/audit          # chain_valid: true + records
curl -X POST "http://localhost:8000/v1/audit/INC-XXXX/decision?decision=BLOCKED&supervisor=name"
```

Tamper demo (judges love this): edit any line of `data/audit/ledger.jsonl`,
re-run `GET /v1/audit` → `chain_valid: false, broken_at: N`.

## 3. Benchmarks & robustness

```bash
python scripts/eval_benchmarks.py
# EER artifact 0.18 · ssl 0.63 · fused 0.23 (full corpus)
# excluding vocoder-proxy class: artifact 0.17 · fused 0.20

python scripts/eval_telephony.py
# level    EER art  EER ssl  EER fused
# clean    0.18     0.63     0.23
# narrowband 0.26 ...          ← the cost of codec-blind training, quantified
```

The SSL EER on this corpus is high **on purpose** — the corpus is dominated by
modern neural TTS that the SSL layer misses; that is the single-detector-failure
story the fusion fixes. All numbers are bootstrap corpus numbers — the
speaker-disjoint validation and leave-one-generator-out protocols are what make
the real-data numbers trustworthy when they land (roadmap).

## 4. Optional: real attack replays

Replace `data/demo/scenario/manager_clone.wav` with any real TTS/clone render
(edge-tts, ElevenLabs, XTTS) of a recorded enrollment phrase and re-run
`run_demo.py`. Report leave-one-generator-out numbers for unseen families —
that is the honest generalization claim.

## Judge FAQ

- *"Is the risk score real-time?"* — yes, per 1-second chunk with 0.5 s overlap;
  the WS endpoint streams verdicts live; the dashboard shows the trajectory.
- *"What prevents the fraud?"* — the policy engine: transaction HOLD, MFA step-up,
  random-phrase liveness challenge, supervisor escalation, incident webhook —
  and the median-robust level means a single glitchy chunk cannot block a genuine
  customer, but sustained attacks cannot slip through.
- *"Why five layers?"* — each layer answers a different question (synthetic? this
  person? dangerous call? dangerous request?); each fails differently (we show a
  live failure of layer B on neural TTS); fusion over independent failure modes
  is the robust design.
- *"Where's the blockchain?"* — the tamper-evident incident ledger: every blocked
  transaction is a hash-chained forensic record (risk, policy, model version,
  supervisor decision, timestamp) that cannot be edited after the fact. No audio
  is stored — privacy by design.
- *"Is this trained on ASVspoof?"* — the bootstrap model in the repo trains on a
  reproducible public-speech + neural-TTS corpus with speaker-disjoint
  validation; the ASVspoof2019 loader is implemented and is week-1 work
  (see docs/architecture.md#data-plan).
- *"Does it work for Indian languages?"* — the Hindi attacker call fires the
  Devanagari intent rules live; Indic-language audio coverage is the AI4Bharat
  week-5 roadmap item.
