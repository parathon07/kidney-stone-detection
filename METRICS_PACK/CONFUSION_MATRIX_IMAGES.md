# Confusion-matrix image set

Open the overview images first:

- [Patient-level OOF overview](graphs/confusion_matrices/patient_oof_locked_overview.png) - the primary development comparison.
- [Slice-level OOF overview](graphs/confusion_matrices/slice_oof_locked_overview.png) - the secondary slice-level view.
- [Fixed-threshold 0.5 overview](graphs/confusion_matrices/patient_oof_fixed05_overview.png) - a common-threshold reference comparison.
- [Transfer-diagnostic overview](graphs/confusion_matrices/transfer_diagnostic_overview.png) - full-development diagnostics, not generalization results.

## How to read a matrix

Rows are the ground-truth class and columns are the model prediction:

| Cell | Meaning |
|---|---|
| TN | Actual negative, predicted negative |
| FP | Actual negative, predicted positive |
| FN | Actual positive, predicted negative |
| TP | Actual positive, predicted positive |

The main evidence is patient-level OOF because slices from the same patient are correlated. Slice-level matrices are included for transparency, not as a replacement for patient-level evaluation.

## Patient-level OOF at each run's locked threshold

| Run | Matrix image |
|---|---|
| Historical 20260826 | [open](graphs/confusion_matrices/patient_oof_locked__historical_20260826__locked.png) |
| E000 baseline | [open](graphs/confusion_matrices/patient_oof_locked__e000_baseline__locked.png) |
| E004 seed 43 | [open](graphs/confusion_matrices/patient_oof_locked__e004_seed43__locked.png) |
| E014 seed 44 | [open](graphs/confusion_matrices/patient_oof_locked__e014_seed44__locked.png) |
| E015 cross-seed | [open](graphs/confusion_matrices/patient_oof_locked__e015_cross_seed__locked.png) |
| E019 seed 44 selector | [open](graphs/confusion_matrices/patient_oof_locked__e019_seed44_selector__locked.png) |
| E023 current best | [open](graphs/confusion_matrices/patient_oof_locked__e023_current_best__locked.png) |
| E024 cross-seed blend | [open](graphs/confusion_matrices/patient_oof_locked__e024_cross_seed__locked.png) |

## Patient-level OOF at the common 0.5 threshold

| Run | Matrix image |
|---|---|
| Historical 20260826 | [open](graphs/confusion_matrices/patient_oof_fixed05__historical_20260826__fixed05.png) |
| E000 baseline | [open](graphs/confusion_matrices/patient_oof_fixed05__e000_baseline__fixed05.png) |
| E004 seed 43 | [open](graphs/confusion_matrices/patient_oof_fixed05__e004_seed43__fixed05.png) |
| E014 seed 44 | [open](graphs/confusion_matrices/patient_oof_fixed05__e014_seed44__fixed05.png) |
| E015 cross-seed | [open](graphs/confusion_matrices/patient_oof_fixed05__e015_cross_seed__fixed05.png) |
| E019 seed 44 selector | [open](graphs/confusion_matrices/patient_oof_fixed05__e019_seed44_selector__fixed05.png) |
| E023 current best | [open](graphs/confusion_matrices/patient_oof_fixed05__e023_current_best__fixed05.png) |
| E024 cross-seed blend | [open](graphs/confusion_matrices/patient_oof_fixed05__e024_cross_seed__fixed05.png) |

## Slice-level OOF at each run's locked threshold

| Run | Matrix image |
|---|---|
| Historical 20260826 | [open](graphs/confusion_matrices/slice_oof_locked__historical_20260826__locked.png) |
| E000 baseline | [open](graphs/confusion_matrices/slice_oof_locked__e000_baseline__locked.png) |
| E004 seed 43 | [open](graphs/confusion_matrices/slice_oof_locked__e004_seed43__locked.png) |
| E014 seed 44 | [open](graphs/confusion_matrices/slice_oof_locked__e014_seed44__locked.png) |
| E015 cross-seed | [open](graphs/confusion_matrices/slice_oof_locked__e015_cross_seed__locked.png) |
| E019 seed 44 selector | [open](graphs/confusion_matrices/slice_oof_locked__e019_seed44_selector__locked.png) |
| E023 current best | [open](graphs/confusion_matrices/slice_oof_locked__e023_current_best__locked.png) |
| E024 cross-seed blend | [open](graphs/confusion_matrices/slice_oof_locked__e024_cross_seed__locked.png) |

## Full-development transfer diagnostics

These are not independent holdout measurements. They show how each candidate behaved when applied to all development patients and were used only as a score-regime/transfer guard.

| Experiment | Matrix image |
|---|---|
| E005 final retrain | [open](graphs/confusion_matrices/transfer_diagnostic__e005_final_retrain.png) |
| E006 epoch 1 | [open](graphs/confusion_matrices/transfer_diagnostic__e006_epoch1.png) |
| E007 BN recalibration | [open](graphs/confusion_matrices/transfer_diagnostic__e007_bn.png) |
| E008 BN one-pass | [open](graphs/confusion_matrices/transfer_diagnostic__e008_bn_onepass.png) |
| E009 equal mean | [open](graphs/confusion_matrices/transfer_diagnostic__e009_equal_mean.png) |
| E010 median | [open](graphs/confusion_matrices/transfer_diagnostic__e010_median.png) |
| E020 seed 44 transfer | [open](graphs/confusion_matrices/transfer_diagnostic__e020_seed44_transfer.png) |
| E021 final retrain | [open](graphs/confusion_matrices/transfer_diagnostic__e021_final_retrain.png) |
| E023 transfer | [open](graphs/confusion_matrices/transfer_diagnostic__e023_transfer.png) |
| E024 transfer | [open](graphs/confusion_matrices/transfer_diagnostic__e024_transfer.png) |

## Historical sealed-test audit

The historical sealed-test result is retained only to document the failure. It is permanently invalidated for tuning and must not be presented as model generalization.

- [Invalidated patient-level sealed-test matrix](graphs/confusion_matrices/sealed_test_invalidated__patient.png)
- [Invalidated slice-level sealed-test matrix](graphs/confusion_matrices/sealed_test_invalidated__slice.png)

The exact source paths and machine-readable counts are in [confusion_matrix_manifest.json](confusion_matrix_manifest.json). The images were generated by [build_confusion_matrix_images.py](build_confusion_matrix_images.py).
