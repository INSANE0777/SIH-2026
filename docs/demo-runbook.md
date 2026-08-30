# Demo runbook (finale rehearsal)

Full end-to-end demo of SwaraKavach against four calls. Everything below is
reproducible from a clean clone in ~5 minutes.

## 0. Setup

```bash
pip install -r requirements.txt && pip install -e .
python scripts/make_demo_data.py      # demo corpus + scenario (network once)
python scripts/make_clone_corpus.py   # 4 attack families + XTTS clones (GPU ~6 min)
python scripts/prepare_libri.py       # 40-speaker bonafide diversity (~4000 chunks)
python scripts/train_artifact.py --labels data/demo/labels.csv \
    --extra-labels data/clone-corpus/chunks/labels.csv,data/libri-human/chunks/labels.csv
```

## 1. Four-call demo (the finale script)

```bash
python scripts/run_demo.py
```

| Call | Audio | Transcript intent rules | Expected verdict | Actions |
|---|---|---|---|---|
| Genuine manager | real human speech | none matched | **LOW** (peak ≈ 0.31) | none — transaction proceeds |
| Cloned manager | **REAL XTTS-v2 zero-shot clone of the enrolled manager's voice** | transfer, urgency, authority | **CRITICAL** (peak ≈ 0.67) | HOLD, LIVENESS, SUPERVISOR, WEBHOOK |
| OTP attacker (English) | neural TTS, different voice | transfer, **credentials**, urgency, authority | **CRITICAL** (peak ≈ 0.82) | all actions |
| Hindi attacker | hi-IN neural TTS, Devanagari | ट्रांसफर/लाख, तुरंत, किसी से न, प्रबंधक | **CRITICAL** (peak ≈ 0.84) | all actions |

Each verdict prints the **explainable breakdown** — voice authenticity, speaker
mismatch, call context, social engineering (0–100) — so the audience sees *why*
the system acted, not just a score.

Talking points:
1. Call 2 is a **real zero-shot voice clone** — XTTS-v2 was given a 6-second
   sample of the manager's voice and generated the attack call. Same voice,
   synthetic artifacts: the speaker layer stays calm (it IS his voiceprint) but
   the artifact layers catch it — exactly the attack the PS describes.
2. The SSL layer **misses** modern Edge neural TTS (p≈0.08) but the fused verdict
   is still CRITICAL — show `layer_means` in the verdict to prove defense-in-depth.
3. The Hindi call fires the bilingual intent rules — the multilingual requirement
   of the PS, live, in Devanagari.
4. "SwaraKavach doesn't just detect a fake voice; it **prevents** the high-risk
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
# EER artifact 0.13 · ssl 0.51 · fused 0.15 (full corpus incl. proxies)
# excluding vocoder-proxy class: fused 0.10

python scripts/eval_loto.py     # leave-one-generator-out — the core claim
# held-out family EER: edge 0.16 · piper 0.05 · sapi 0.09 · xtts 0.19, mean 0.12

python scripts/eval_telephony.py
# clean fused EER 0.15 -> 0.32 worst case (narrowband/mu-law/noise/packet loss)
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
- *"How do you know it generalizes?"* — leave-one-generator-out: we train
  WITHOUT one attack family and it still catches that family (mean held-out EER
  0.12 today, target < 0.06 as families grow). In-the-wild attackers are, by
  definition, generators we have never seen — this is the only protocol that
  measures that.
- *"Is this trained on ASVspoof?"* — the bootstrap model in the repo trains on a
  reproducible public-speech + neural-TTS corpus with speaker-disjoint
  validation; the ASVspoof2019 loader is implemented and is week-1 work
  (see docs/architecture.md#data-plan).
- *"Does it work for Indian languages?"* — the Hindi attacker call fires the
  Devanagari intent rules live; Indic-language audio coverage is the AI4Bharat
  week-5 roadmap item.
