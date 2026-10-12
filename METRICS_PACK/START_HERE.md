# Kidney-stone model: metrics pack

This folder is a simple evidence pack for explaining the research project. It contains the current model's measurements, the complete experiment history, the graphs, and the main limitations.

## The short version

The current verified development candidate is **E023**.

| Measure | Result |
|---|---:|
| Patient-level ROC-AUC | **0.853414** |
| Original baseline patient-level ROC-AUC | 0.804140 |
| Improvement over baseline | +0.049274 |
| Slice-level ROC-AUC | 0.814036 |
| Mean patient AUC across five folds | 0.914461 |
| Weakest fold patient AUC | 0.863971 |
| Transfer-shape check, KS statistic | 0.130435 (pass; lower is better) |
| Locked-threshold sensitivity | 0.717949 |
| Locked-threshold specificity | 0.891566 |
| Correctness tests | 47 passed |
| Sealed test used for tuning | No |

These are development results, not medical validation and not a medical-grade deployment claim.

## What the model does

The model predicts a Stone probability for each CT slice. The slice probabilities are averaged for each patient. The patient-level average is the main result because the label belongs to the patient, not to every individual slice.

## Where to look

- CONFUSION_MATRIX_IMAGES.md - confusion-matrix image index, with patient OOF, slice OOF, transfer diagnostics, and the invalidated sealed-test audit.
- graphs/confusion_matrices/ - the individual PNG matrices and overview contact sheets.

- `CURRENT_BEST_METRICS.md` — the complete E023 scorecard in plain language.
- `ALL_EXPERIMENT_METRICS.md` — every recorded experiment, including failures and near-misses.
- `experiment_ledger.csv` — spreadsheet-friendly summary of all experiments.
- `DATASET_AND_LIMITATIONS.md` — what data was used and what it cannot prove.
- `METRICS_GLOSSARY.md` — simple explanations of ROC-AUC, sensitivity, specificity, KS, and the other terms.
- `PROJECT_EXPLAINER.html` — the interactive offline walkthrough of the whole project.
- `FINAL_VERIFICATION.md` — integrity and reproducibility checks.
- `graphs/` — charts and diagrams for presentation.
- `e023_evidence/` — machine-readable evidence copied from the winning development run.

## Important boundary

The 40-patient sealed test set was kept out of tuning. An earlier historical test run collapsed and permanently invalidated that test for future optimization. Therefore, this pack reports development-only evidence and does not claim external clinical generalization.

The base checkout at `D:/Documents/Desktop/kidney stone detection` was not modified. This pack belongs to the isolated research worktree.
