# Active research state

Status: GOOD_ENOUGH. Final verification passed; reconstruct from this file plus `config.json`, `best.json`, and the append-only `results.jsonl` if research is explicitly resumed.

## Current BEST

- E023, development-only patient OOF ROC-AUC 0.853414 (+0.049274 vs E000).
- Run: `D:/Documents/Desktop/kidney stone detection-isolated/runs/20260911_e023_seed43_patient_auc_cv`.
- Fold patient AUCs: 0.9375, 0.9375, 0.863971, 0.9125, 0.920833; mean 0.914461; worst 0.863971.
- OOF gates and development transfer guard pass (std 0.373026, KS 0.130435, threshold delta 0.062112).
- Final correctness checks pass: 47 tests and CLI smoke. This is the current verified development candidate; sealed-test evaluation remains prohibited.
- Final verification passed: OOF integrity, all five fold artifacts, manifest hash, frozen split hashes, OOF gates, transfer gates, and sealed-test exclusion all verified.

## Strongest rejected candidate

- E004, seed 43 complete CV: patient OOF ROC-AUC 0.854032, slice OOF ROC-AUC 0.829452, mean fold patient AUC 0.839265, worst fold 0.680147.
- All OOF acceptance gates pass and all five fold artifacts/provenance hashes are present.
- E005 final retrain/transfer verification failed decisively: probability std 0.103676 passed, but KS vs OOF was 0.726708 and locked-threshold positive-rate delta was 0.503106; every final-development patient was positive at the locked threshold.
- E004 therefore does not replace E000. Its OOF gain is retained as evidence, not treated as a deployable improvement.

## Completed in this session

- Backend-neutral smoke-test repair: 46/46 tests pass and CLI smoke passes. This is an infrastructure fix, not a model improvement.
- E002 fold-2 seed-43 diagnostic: patient AUC 0.680147, slice AUC 0.553994, best epoch 1. It modestly exceeds the same fold's seed-42 0.6618 but is not a global win.
- E003 fold-3 seed-43 diagnostic: patient AUC 0.866667, slice AUC 0.913462, best epoch 19. It exceeds the same fold's seed-42 patient AUC 0.741667, but the epoch/score spread shift makes this evidence of seed variance until a full CV confirms it.
- E006 epoch-1 final-retrain diagnostic: transfer failed in the opposite direction from E005, with probability std 0.047622, KS vs OOF 0.447205, and locked-threshold positive-rate delta 0.447205; only 4.97% of development patients were positive.
- E007 fixed-weight BN recalibration (three shuffled clean-development passes) restored probability spread and threshold prevalence (std 0.433687; threshold delta 0.031056), but produced a bimodal score shape with KS vs OOF 0.322981, so the transfer guard still failed.
- E008 one-pass clean BN recalibration was materially the same failure: std 0.435253, threshold delta 0.031056, KS vs OOF 0.316770. Calibration strength is not the main fix.
- E009 equal-probability averaging of all five E004 fold-best checkpoints nearly transferred: std 0.313090 and threshold delta 0.018634 passed, while KS 0.173913 narrowly missed the <0.15 gate. The ensemble is a serious near-miss, not BEST.
- E010 median aggregation of the same fold predictions was worse (std 0.393733, threshold delta 0.024845, KS 0.242236); the equal mean remains the better ensemble diagnostic but still fails.
- E011 fold-2 patient-balanced diagnostic was not material: patient AUC 0.683824 versus E004's 0.680147, while slice AUC fell to 0.539090 from 0.553994; both selected epoch 1.
- A fixed-prediction leave-one-fold-out audit found omitting fold 0 or fold 3 would narrowly pass KS (0.149068), while omitting other folds or using all five did not. This is an exploratory subset signal and requires independent-seed confirmation before any ensemble choice.
- E012 independent seed-42 ensemble audit falsified that subset signal: all-five mean KS 0.242236, omit-fold-0 0.236025, and omit-fold-3 0.260870; none passed the transfer gates.
- E013 seed-44 CV was externally interrupted by Windows user logoff at 05:41:14 after folds 0 and 1 completed and fold 2 had partial history. Partial patient AUCs were 0.555147 and 0.904412; no complete OOF metric exists. This is an execution interruption, not a model-quality result. E014 recovered folds 2-4 with the same seed/config/splits.
- E014 completed seed-44 CV: patient OOF ROC-AUC 0.804603, slice OOF ROC-AUC 0.763229, mean fold patient AUC 0.788529, worst fold 0.555147, and fold AUCs [0.555147, 0.904412, 0.908088, 0.908333, 0.666667]. OOF acceptance gates passed, but the result is only +0.000463 over E000 and 0.049428 below E004; it is rejected and E000 remains BEST.

## Active operation

- E014 is complete and rejected as a non-improvement. Its canonical OOF artifacts and metrics are preserved at `runs/20260910_e014_seed44_recovered`.
- E015 is complete and retained as a transfer near-miss: cross-seed OOF patient AUC 0.871486, worst fold 0.8125, but KS vs OOF 0.180124 failed the strict transfer gate. Its artifacts and ten-checkpoint development guard are preserved at `runs/20260910_e015_seed43_44_oof_ensemble`.
- E016 is complete and falsified the fixed convex-blend direction: no tested E004/E014 weight passed KS < 0.15. Its full grid is preserved at `runs/20260910_e015_seed43_44_oof_ensemble/blend_weight_grid.json`.
- E017 completed the fold-0 seed-44 diagnostic: patient-AUC selection reached 0.816176 at epoch 26 versus canonical E014 fold-0 0.555147, with identical data/model/seed/optimizer/augmentation. The slice AUC was 0.780709. This is promising but is not a full-CV result.
- E018 completed the independent fold-4 replication: patient-AUC selection reached 0.854167 at epoch 13 versus canonical E014 fold-4 0.666667, with slice AUC 0.686763. Both weak folds now improve under the aligned selector.
- E019 CV is complete: seed-44 patient-AUC-selected OOF patient ROC-AUC 0.834415, slice ROC-AUC 0.768757, mean fold patient AUC 0.877402, and worst fold 0.816176; all OOF gates pass. Fold best epochs are [26, 29, 23, 44, 13]. E020's equal-mean five-checkpoint transfer guard passed spread and threshold prevalence but failed KS (0.167702 > 0.15). E021's exact-26-epoch single full-development retrain failed KS decisively (0.378882) despite passing the other transfer gates. E022 independently rescued seed-43 fold 2 from 0.680147 to 0.863971 patient AUC (+0.183824). E023 seed-43 patient-AUC-selector CV and equal-mean five-checkpoint development transfer are complete and pass all gates; E023 is now BEST. E024 equal cross-seed blending increased OOF patient AUC to 0.856194 but failed transfer KS (0.192547), and E025's predeclared 21-point weight grid found no transfer-safe gain. The fixed blend direction is closed; next is site/patient-size robustness analysis before any new training run. No CLI finalization or sealed-test action is permitted.

## Evidence-backed discoveries

- The old 1024-item loader shuffle against a class-sorted manifest created long negative-only and late positive-heavy batches; full-dataset shuffle materially improved OOF.
- The shuffle-remediated model still has strong fold/epoch variability and a failed development transfer guard.
- The final full-development retrain is sensitive to the locked epoch budget: E006 epoch 1 was nearly all-negative and E005 epoch 15 nearly all-positive under eval-mode inference. On E005 clean images, the first 80-image probe had eval-mode mean/std 0.8024/0.1417 versus train-mode 0.3649/0.2469, implicating a BatchNorm moving-statistics/eval mismatch as a high-value implementation hypothesis.
- Site/format is a real confound: dev/test patient counts differ substantially by site, and a site-only baseline is about 0.562 AUC.
- Derivative-heavy training is not trusted; derivatives remain disabled.
- Read-only development audit: all 161 patients have consistent original labels, but negative patients average 18.89 slices versus 15.74 for positive patients; the top 10% of patients contribute 28.25% of slices. This motivates a patient-balanced sampler experiment.

## Important failures and boundaries

- The earlier sealed-test run collapsed to all-positive predictions (patient AUC 0.538462) and invalidated the sealed test for all future tuning.
- Never run `finalize`, reopen the sealed test, use KidneyStoneTR, regenerate frozen splits, or quote the incomplete 20260828 run as a complete result.

## Next research direction

1. Research is complete at the defined bar. If explicitly resumed, start with a fresh hypothesis rather than reopening the sealed test.
2. The fixed cross-seed blend direction is closed: E024 and the predeclared E025 grid produced no transfer-safe gain over E023.
3. Preserve E023's five-checkpoint ensemble and its development-only provenance; do not run `finalize` or reopen the sealed test.

## Resume instruction

Read `.research/config.json`, `.research/best.json`, this file, and the latest relevant result records. Inspect the current run artifacts before launching new work. Preserve the sealed-test boundary and checkpoint every completed experiment.
