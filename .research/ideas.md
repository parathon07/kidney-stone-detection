# Ranked experiment backlog

1. **E025 cross-seed weight grid** — test fixed E019 weights 0.00, 0.05, ..., 1.00 against both OOF AUC and the unchanged transfer guard. E024's equal blend raised OOF to 0.856194 but failed KS 0.192547; E023 weight 0.00 remains the safe anchor.
2. **Site and patient-size robustness** — stratify E023 and any E025 transfer-safe candidate by acquisition site and slice-count strata without touching sealed data.
3. **Final-training protocol** — revisit only if a blended candidate improves the frontier; E021 showed that a single exact-epoch full-development retrain can fail KS despite passing spread and threshold prevalence.
4. **Controlled anti-instability intervention** — if blend evidence shows residual seed/fold instability, test one isolated change rather than broad hyperparameter sweeps.
5. **Normalization/precision stability** — only if repeated-seed evidence identifies a consistent numerical/training instability; preserve the kidney_cnn_v1 topology.
6. **External data** — KidneyStoneTR remains quarantined because patient mapping/provenance and exact-slice overlap are unresolved. Do not use it for training or evaluation.

Completed evidence: E019 independently replicated the patient-AUC selector on seed 44 but failed transfer KS; E023 replicated it on seed 43 and produced the first candidate passing all OOF and transfer gates. E024 equal cross-seed blending improved OOF but failed KS; E023 is current BEST.

Do not spend the main budget on tiny learning-rate or augmentation sweeps until repeated-seed evidence identifies the dominant failure mode.
