# Dataset and limitations

## Dataset structure

The active source is the Mendeley Data V2 kidney-stone CT dataset.

| Quantity | Count |
|---|---:|
| Total patients | 201 |
| Original source images | 3,364 |
| Original Non-Stone images | 1,787 |
| Original Stone images | 1,577 |
| Indexed derivative rows | 35,457 |
| Development patients | 161 |
| Sealed-test patients | 40 |
| Development original slices used by E023 | 2,796 |
| Sealed-test original slices | 568 |

The model used original CT slices only. Supplied derivatives were disabled because derivative-heavy training encouraged memorization and produced validation collapse. The derivatives are not independent patients.

## Evaluation design

1. Build a manifest that maps each original slice to a patient and label.
2. Freeze a patient-wise outer split: 161 development patients and 40 sealed-test patients.
3. Use five patient-wise development folds.
4. Train on original slices only.
5. Predict each held-out slice.
6. Average slice probabilities by patient.
7. Rank patients using patient-level ROC-AUC.

This prevents slices from the same patient appearing in both training and validation for a fold.

## Main limitations

- Only 201 patients are available. That is small for a medical-imaging generalization claim.
- The 3,364 original images are slices, not 3,364 independent patients.
- Patients contribute different numbers of slices. Negative patients average 18.89 slices and positive patients 15.74 slices.
- The largest 10% of patients contribute 28.25% of slices, so slice counts are uneven.
- Site/format is a confound. A site-only baseline is about 0.562 AUC, and E023's site scores vary: FA 0.782716, ME 0.812500, RA 0.869008.
- There are only 20 patients in the ME subgroup, so that estimate is uncertain.
- The sealed test was not used for tuning. It is permanently excluded after an earlier historical run collapsed to all-positive predictions.
- KidneyStoneTR was audited and quarantined. Its archive did not contain a verifiable patient-to-slice map, and its advertised holdout contained duplicate images across directories. It was not used.
- The current result is development evidence, not an external holdout result, clinical validation, or a medical-grade deployment claim.

## What the score does and does not mean

An ROC-AUC of 0.853414 means that, on these development OOF predictions, a randomly chosen positive patient tends to receive a higher score than a randomly chosen negative patient. It does not mean 85.34% diagnostic accuracy, and it does not establish performance on a new hospital or future patients.

The threshold metrics are one operating point on the development data. They should not be interpreted as prospective clinical sensitivity or specificity.

## External dataset audit

The dataset-finding thread considered several related sources. They were not automatically added because patient identity, acquisition independence, labels, licenses, or duplication could not be verified. `ALL_EXPERIMENT_METRICS.md` records the model experiments; the full data audit remains in `D:/Documents/Desktop/kidney stone detection-research/findings.md`.
