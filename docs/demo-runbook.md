# Demo runbook (finale rehearsal)

Full end-to-end demo of SwaraKavach against three calls. Everything below is
reproducible from a clean clone in ~5 minutes.

## 0. Setup

```bash
pip install -r requirements.txt && pip install -e .
python scripts/make_demo_data.py      # corpus + scenario files (needs network once)
python scripts/train_artifact.py      # layer A bootstrap training (~1 min)
```

## 1. Three-call demo (the finale script)

```bash
python scripts/run_demo.py
```

| Call | Audio | Expected verdict | Actions |
|---|---|---|---|
| Genuine manager | real human speech (`manager_real.wav`) | **LOW** (peak ≈ 0.21) | none — transaction proceeds |
| Cloned manager | same voice through the vocoder proxy (`manager_clone.wav`) | **HIGH** (peak ≈ 0.59) | HOLD_TRANSACTION, LIVENESS_CHALLENGE, SUPERVISOR_ALERT |
| TTS attacker | different neural-TTS voice (`attacker_tts.wav`, Edge en-IN) | **CRITICAL** (peak ≈ 0.89) | all + INCIDENT_WEBHOOK |

Talking point: the SSL layer **misses** the Edge neural TTS call (p≈0.08) but the
fused verdict is still CRITICAL — show `layer_means` in the verdict to prove
defense-in-depth works.

## 2. Live API + dashboard

```bash
uvicorn swarakavach.server:app --port 8000
# open http://localhost:8000
```

Dashboard: enroll a speaker (record or upload a WAV), then analyze any file or
stream the microphone live — risk gauge, per-layer bars, trajectory sparkline,
and the action log update per chunk.

REST equivalent:

```bash
curl -X POST "http://localhost:8000/v1/enroll?speaker_id=regional_manager" \
     --data-binary @data/demo/scenario/manager_real.wav
curl -X POST "http://localhost:8000/v1/analyze?speaker_id=regional_manager&context=%7B%22caller_id_match%22%3A%22false%22%2C%22tx_amount%22%3A5000000%7D" \
     --data-binary @data/demo/scenario/manager_clone.wav
```

## 3. Benchmarks

```bash
python scripts/eval_benchmarks.py
# EER artifact 0.00 · EER ssl 0.39 · EER fused 0.00  (bootstrap corpus)
```

The SSL EER on this corpus is high **on purpose** — the bootstrap corpus is
dominated by modern neural TTS that the SSL layer misses; that is the single-
detector-failure story the fusion fixes. On ASVspoof-style corpora the SSL layer
carries its weight (validate with `--asvspoof-dir` when the dataset is licensed
in).

## 4. Optional: real attack replays

Replace `data/demo/scenario/manager_clone.wav` with any real TTS/clone render
(edge-tts, ElevenLabs, XTTS) of a recorded enrollment phrase and re-run
`run_demo.py`. The artifact layer generalizes across vocoder/TTS families seen
in training; report leave-one-tool-out numbers for unseen families.

## Judge FAQ

- *"Is the risk score real-time?"* — yes, per 1-second chunk with 0.5 s overlap;
  the WS endpoint streams verdicts live; the dashboard shows the trajectory.
- *"What prevents the fraud?"* — the policy engine: transaction HOLD, MFA step-up,
  random-phrase liveness challenge (a clone can't answer unpredicted phrases
  naturally), supervisor escalation, incident webhook.
- *"Why four layers?"* — each layer fails differently (we show a live failure of
  layer B on neural TTS); fusion over independent failure modes is the robust design.
- *"Is this trained on ASVspoof?"* — the bootstrap model in the repo trains on a
  reproducible synthetic+public-speech corpus; the ASVspoof2019 loader is
  implemented and is week-1 work (see docs/architecture.md#data-plan).
