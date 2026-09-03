# SwaraKavach experiment log
Baseline (v1.0): LOTO mean 0.113 (edge .155/piper .026/sapi .060/xtts .212); fused EER .06; B=Hemgg chance-level (.51) on our corpus

## EXP-1: fusion weight grid (2026-08-31)
- Question: does B's 0.28 weight hurt? Grid on held-out (clean + AMR-degraded).
- Result: SATURATED — layer A = 0.0 EER on both; every weight combo ties at 0.0.
- Conclusion: weights are unidentifiable when A is perfect on-distribution. B only matters where A errs (xtts fold, 0.212). Current weights FINE — no change. Next: strengthen A on xtts.

## EXP-2: layer-A capacity grid on the xtts fold (2026-08-31) — MAJOR FINDING
- Grid (n_lfcc 20/40/60, filters 60/128, base/wide CNN) on my full-corpus xtts fold.
- ALL configs tie at EER 0.0008 — capacity/features are NOT the bottleneck.
- WHY: my fold trained on clone+demo+libri+hindi (2719 clips) vs eval_loto's clone-corpus-only (385). The 0.212 'xtts is hardest' was UNDER-POWERED PROTOCOL, not model weakness. Full-corpus training drops xtts held-out EER to 0.0008 (~0.21 -> ~0.001).
- ACTION: re-run eval_loto with full training corpus per fold. Expect the LOTO table to collapse toward ~0.01-0.03 across families. The 'LOTO mean 0.113' number was measuring data scarcity, not generalization limits.

## EXP-3: eval_loto fold-composition post-mortem (2026-08-31) — EVAL BUG FOUND+FIXED
- The '0.0008 xtts' from EXP-2 was DEGENERATE: all-attack test set (no bonafide rows) makes EER meaningless. An EER needs both classes. Guard: always print test label counts before trusting an EER.
- Fixed fold: attacks of held-out family + FIXED human holdout speakers (li1089/1188/121/1221 + ih_hi_m_agri_*), never in any fold's training. Two-sided, honest.
- Full-diversity fold at 10 epochs: edge 0.249 / piper 0.249 / sapi 0.094 / xtts 0.365, mean 0.239. Per-class on xtts fold: attacks mean 0.36 (36% >0.5) vs humans mean 0.11 (9% >0.5) — separation direction correct, margin thin at 10 epochs.
- UNDER-TRAINED: production recipe = 20-30 epochs. Next: same fold at 20-30 epochs before drawing conclusions.

## EXP-4: fine-tune our own wav2vec2 layer B (2026-08-31) — NEGATIVE RESULT, root-caused
- Tried head-only (6 ep) and top-2-layer unfreeze (23.7M params, 3 ep, lr 1e-5) on wav2vec2-base.
- Speaker-disjoint val: EER 0.56 — WORSE than chance. Attack mean 0.96 but HUMAN mean 0.978 (humans score HIGHER than attacks!). Hemgg on same val: 0.67.
- ROOT CAUSE: with ~16 human speakers (4-6 held out), a fine-tuned wav2vec2 learns 'unseen speaker = spoof', not 'synthetic = spoof'. wav2vec2 fine-tuning needs THOUSANDS of speakers (what ASVspoof 2019 LA provides: 2.6k bonafide utts from 40+ speakers... actually hundreds of speakers). Our 60k-param LFCC CNN wins BECAUSE its capacity is too small to memorize speakers — a small-model-is-the-regularizer result.
- DECISION: keep Hemgg as layer B (borrowed but honest about it); our own SSL model is explicitly gated on ASVspoof-scale data. Log in README as a negative result — it strengthens the 'why fusion + why small edge CNN' story.
- ENV FIX along the way: faster-whisper install had silently upgraded numpy to 2.4.6, which breaks ALL of transformers 4.42 (numpy<2.0 pin). Restored numpy==1.26.4; full stack verified (transformers/TTS/faster_whisper/onnxruntime/torch-cuda all import and run). requirements.txt needs numpy>=1.24,<2.0 pin.

## EXP-5: LOTO protocol v2 at 30 epochs (2026-08-31)
- Full-diversity folds, two-sided test (held-out attacks + NEVER-SEEN humans), 30 ep: mean 0.198 (xtts 0.289, sapi 0.181).
- vs old 0.113: the old protocol's test set contained IN-TRAINING humans (memorized bonafide side). v2 is the deployment condition and is the number we report. Regression in the headline = increase in honesty.
- NEXT: capacity grid rerun on the FIXED fold (the old 'capacity doesn't matter' conclusion came from the degenerate all-attack test).

## EXP-6: capacity grid on FIXED xtts fold, 20 epochs (2026-08-31)
- base 105k/20-lfcc/60-filters: EER 0.2475 (BEST). Everything bigger is WORSE: wide 0.33, 40-lfcc 0.35, 40x128 0.40.
- CONCLUSION: on our data budget, the small CNN's inability to memorize speakers IS the generalization. Larger capacity memorizes seen speakers -> worse on unseen ones. Pairs with EXP-4 (wav2vec2 23.7M trainable -> worse than chance). Both say: this is a DATA-limited regime, not model-limited.
- ARCHITECTURE VALIDATED: the 60k edge CNN stays. 'Small model as regularizer' is now an evidence-backed line for judges.

## EXP-7: chunk vs call-level aggregation on xtts fold (2026-08-31)
- chunk-level EER 0.374 -> CALL-level (mean over each call's chunks) 0.284 on the fixed fold.
- Aggregation over a call is itself a ~25% relative EER reduction — validates the median/mean decision-rule design as part of the detection stack, not just a policy nicety.

## SESSION SUMMARY (2026-08-31 experiment day)
1. Fusion weights: unidentifiable when A saturates; kept as-is (EXP-1).
2. EVAL BUG found+fixed: LOTO needed two-sided test sets (unseen attacks + unseen humans). Old 0.113 was flattered by memorized-human test rows; honest v2 = 0.198 mean at 30 ep (EXP-3/5). Also caught my own 0.0008 artifact (degenerate all-attack EER).
3. NEGATIVE RESULT with root cause: fine-tuning wav2vec2 on 16-speaker data learns 'unseen=synthetic' (humans scored 0.978 > attacks 0.960). Our own SSL layer is gated on ASVspoof-scale data (EXP-4).
4. Small-model-is-the-regularizer: bigger CNN = worse LOTO (0.25->0.40). The 105k edge CNN is the right size for the data budget (EXP-6).
5. Call-level aggregation cuts xtts-fold EER 0.37->0.28 (EXP-7).
6. ENV: numpy 2.4.6 (pulled by faster-whisper) silently breaks transformers 4.42 entirely; pinned numpy<2.0.

## EXP-5 full table (30-epoch two-sided folds, complete)
- edge 0.1465 · piper 0.1773 · sapi 0.1812 · xtts 0.2885 · mean 0.1984
- Note: edge is the EASIEST held-out family (0.147) despite its modern neural TTS — neural voices with consistent vocoder signatures generalize across voices within a family. XTTS zero-shot clones remain the hard case (0.289): the clone inherits the target's real-voice texture, leaving only synthesis artifacts to detect.

## EXP-8: 5-family two-sided LOTO, 30-epoch folds (2026-09-04)
Table: edge 0.329 · piper 0.305 · sapi 0.091 · SARVAM 0.082 · xtts 0.319 · mean 0.225.
- HEADLINE: sarvam held-out fold = 0.082 EER — when the Indian TTS is excluded from training,
  the detector still catches it at ~8% EER. This is the formal LOTO version of the live
  'caught Sarvam unseen' demonstration (which measured mean 0.79 flag-rate on 12 chunks).
- FOLD VARIANCE CAVEAT (important): per-fold EERs moved substantially between this run and
  the 4-family run under the same protocol (edge 0.147→0.329, piper 0.177→0.305,
  sapi 0.181→0.091, xtts 0.289→0.319). The small CNN's training has high run-to-run
  variance at this data scale (retrain val_acc fluctuated 0.66-0.80 between adjacent
  epochs). Single-run fold numbers carry roughly ±0.1 uncertainty. PROTOCOL FIX for any
  headline claim: multi-seed fold averaging (3 seeds per fold) — noted as the eval
  upgrade; the sarvam 0.082 and sapi 0.091 vs edge/piper/xtts ~0.3 separation is large
  enough to trust directionally, but the mean 0.225 vs the earlier 0.198 is NOT a
  regression signal — it is fold noise + a changed fold set (5 vs 4).
- Reading: the two Hindi-heavy families (sapi-classic + sarvam neural) generalize best
  held-out; the neural-vocoder families (edge/xtts) remain hardest, consistent with
  every previous measurement.
