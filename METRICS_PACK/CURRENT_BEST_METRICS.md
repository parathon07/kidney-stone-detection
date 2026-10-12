# Current best scorecard: E023

## 1. Main result

E023 is the current verified development candidate. It uses five patient-wise cross-validation folds and an equal-mean ensemble of the five fold-best checkpoints.

| Metric | Value |
|---|---:|
| Patient-level OOF ROC-AUC | **0.8534136546** |
| Slice-level OOF ROC-AUC | 0.8140357018 |
| Mean fold patient ROC-AUC | 0.9144607843 |
| Worst fold patient ROC-AUC | 0.8639705882 |
| Locked threshold | 0.4531617227 |
| Patient count | 161 |
| Development slice count | 2,796 |
| OOF rows | 2,796 |
| Best epochs by fold | 23, 16, 26, 25, 29 |
| Median best epoch | 25 |
| Seed | 43 |

## 2. Patient-level metrics at the locked threshold

The threshold was selected from development OOF predictions with Youden's J statistic. ROC-AUC does not depend on one threshold; these values describe one operating point chosen after the ranking score was measured.

| Metric | Value |
|---|---:|
| ROC-AUC | 0.8534136546 |
| PR-AUC | 0.8605877001 |
| Sensitivity / recall | 0.7179487179 |
| Specificity | 0.8915662651 |
| Precision | 0.8615384615 |
| F1 score | 0.7832167832 |
| Balanced accuracy | 0.8047574915 |
| True negatives | 74 |
| False positives | 9 |
| False negatives | 22 |
| True positives | 56 |

At a fixed threshold of 0.5, the same OOF predictions give sensitivity 0.666667, specificity 0.903614, precision 0.866667, F1 0.753623, and balanced accuracy 0.785141.

## 3. Slice-level metrics at the same locked threshold

Slice metrics are secondary diagnostics. The patient-level score remains the primary metric.

| Metric | Value |
|---|---:|
| Slice ROC-AUC | 0.8140357018 |
| Slice PR-AUC | 0.8028700421 |
| Sensitivity / recall | 0.6343648208 |
| Specificity | 0.8405612245 |
| Precision | 0.7570456754 |
| F1 score | 0.6902968542 |
| Balanced accuracy | 0.7374630227 |
| True negatives | 1,318 |
| False positives | 250 |
| False negatives | 449 |
| True positives | 779 |

## 4. Fold-by-fold patient results

| Fold | Patients | Patient ROC-AUC | Probability standard deviation |
|---:|---:|---:|---:|
| 0 | 33 | 0.937500 | 0.240461 |
| 1 | 33 | 0.937500 | 0.331679 |
| 2 | 33 | 0.863971 | 0.321003 |
| 3 | 31 | 0.912500 | 0.320205 |
| 4 | 31 | 0.920833 | 0.320377 |

No fold met the collapse criterion. The folds are not identical in size, so the mean fold score and the pooled patient OOF score answer slightly different questions.

## 5. Site breakdown

These are development-only patient AUCs by site/format label. They are useful for finding distribution differences, not for claiming independent clinical performance.

| Site label | Patients | Positive patients | Patient ROC-AUC |
|---|---:|---:|---:|
| FA | 42 | 15 | 0.782716 |
| ME | 20 | 8 | 0.812500 |
| RA | 99 | 55 | 0.869008 |

The small ME group makes its estimate less certain than the larger groups.

## 6. Transfer guard

The five fold-best checkpoints were also evaluated as an equal-mean ensemble on the full development set. This checks whether the score shape remains similar outside each fold's held-out predictions.

| Transfer check | Value | Gate |
|---|---:|---|
| Probability standard deviation | 0.373026 | pass; must be at least 0.10 |
| KS distance versus OOF score distribution | 0.130435 | pass; must be below 0.15 |
| OOF positive rate | 0.403727 | reference |
| Development positive rate | 0.465839 | reference |
| Absolute positive-rate difference | 0.062112 | pass; must be at most 0.10 |

## 7. Comparison with the baseline

| Metric | E000 baseline | E023 candidate | Change |
|---|---:|---:|---:|
| Patient OOF ROC-AUC | 0.804140 | **0.853414** | **+0.049274** |
| Slice OOF ROC-AUC | 0.796691 | 0.814036 | +0.017345 |
| Worst fold patient AUC | 0.661800 | 0.863971 | +0.202171 |
| Fold patient AUCs | 0.8676, 0.9301, 0.6618, 0.7417, 0.8625 | 0.9375, 0.9375, 0.8640, 0.9125, 0.9208 | more stable |

The comparison is development-only. The baseline's transfer guard failed, while E023 passed the three transfer gates.

## 8. Robustness audit versus the baseline

The E026 audit compared fixed development predictions at their respective thresholds. It is recorded separately from the official E023 OOF scorecard because it uses the robustness-audit artifact and its threshold metrics.

| Metric | Baseline | E023 candidate |
|---|---:|---:|
| Patient ROC-AUC | 0.804140 | 0.853414 |
| Sensitivity | 0.653846 | 0.705128 |
| Specificity | 0.819277 | 0.891566 |
| Precision | 0.772727 | 0.859375 |
| F1 | 0.708333 | 0.774648 |
| True positives | 51 | 55 |
| True negatives | 68 | 74 |
| False positives | 15 | 9 |
| False negatives | 27 | 23 |

## 9. Verification status

- Five-fold patient-wise OOF complete: pass.
- OOF integrity: pass.
- Development transfer guard: pass.
- Manifest hash: verified.
- Frozen split hashes: verified.
- All five fold artifacts present: verified.
- Correctness suite: 47 tests passed.
- CLI smoke check: pass.
- Sealed-test access in E023: false.
- Research status: GOOD_ENOUGH under the documented development-only bar.

This is a research result, not medical-grade deployment evidence.
