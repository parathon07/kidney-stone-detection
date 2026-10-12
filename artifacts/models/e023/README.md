# E023 current-best model ensemble

This directory contains the five fold-best `kidney_cnn_v1` checkpoints from the
development-only E023 run. Inference should average the Stone-class
probability from all five checkpoints and aggregate slice probabilities by
patient using the repository's canonical mean rule.

- Source run: `runs/20260911_e023_seed43_patient_auc_cv`
- Locked patient threshold: `0.4531617227`
- Patient OOF ROC-AUC: `0.8534136546`
- Slice OOF ROC-AUC: `0.8140357018`
- Development transfer guard: passed (score std `0.3730257534`, KS `0.1304347826`, threshold-rate delta `0.0621118012`)
- Sealed test: intentionally not reopened or evaluated

The model files are stored with Git LFS. The raw CT dataset is not copied into
this public repository; use the official [Mendeley Data V2 source](https://data.mendeley.com/datasets/fwhytt5mzd/2),
then run the repository's `prepare` command to regenerate local manifests.
