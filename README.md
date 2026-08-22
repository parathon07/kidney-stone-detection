# Kidney Stone Detection CNN

Canonical 2D binary classification pipeline (Non-Stone vs Stone) for axial CT imaging based on the Mendeley Data V2 benchmark.

## Dataset Reference
- **Source**: Abdalla et al., *"Kidney stone detection via axial CT imaging: A dataset for AI and deep learning applications,"* Data in Brief, vol. 59, 111446, 2025.
- **Mendeley Data V2 DOI**: [10.17632/fwhytt5mzd.2](https://data.mendeley.com/datasets/fwhytt5mzd/2)
- **Cohort**: 201 patients (3,364 original images: 1,577 Stone, 1,787 Non-Stone).

Download and extract the axial CT dataset to a local path (e.g., `D:/kidney_ct`).

---

## Quick Start & CLI Workflow

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Verify Pipeline Integrity (Smoke Test)
Verify model shapes, parameter counts ($4.79\text{M}$ params), checkpointing, and metric formulas on synthetic tensors:
```bash
python train.py smoke --config configs/default.yaml
```

### 3. Prepare Dataset & Freeze Splits
Scan raw dataset, parse patient IDs, generate `data/manifest.csv`, and freeze the $161$ development / $40$ sealed test patient split and 5 CV folds:
```bash
python train.py prepare --config configs/default.yaml --data-dir D:/kidney_ct
```

### 4. Run 5-Fold Cross-Validation & OOF Optimization
Train all 5 patient-wise CV folds, reload best validation checkpoints, aggregate out-of-fold (OOF) predictions, and compute the optimal operating threshold via Youden's $J$:
```bash
python train.py cv --config configs/default.yaml --data-dir D:/kidney_ct
```
*Optional: run a specific fold on a separate workstation:*
```bash
python train.py cv --config configs/default.yaml --data-dir D:/kidney_ct --fold 2
```

### 5. Final Retrain, Sealed Test Evaluation & Bundle Export
Retrain from scratch on all 161 development patients for the median CV best epoch budget, evaluate once on the sealed 40-patient test set, and export the deployment inference bundle:
```bash
python train.py finalize --run-dir runs/<run_id> --data-dir D:/kidney_ct
```

---

## Output Artifacts & Deployment Bundle

Every completed run produces an immutable directory in `runs/<run_id>/`:
```
runs/<run_id>/
├── config_resolved.yaml
├── metadata.json
├── data_manifest_hash.txt
├── split_hashes.json
├── fold_0/ ... fold_4/
│   ├── best.keras
│   ├── history.csv
│   ├── val_predictions.csv
│   └── fold_metrics.json
├── oof_predictions.csv
├── oof_metrics.json
├── threshold.json
├── plots/
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── confusion_matrix.png
│   └── learning_curves.png
├── locked_config.yaml
└── final/
    ├── model.keras
    ├── test_predictions.csv
    ├── test_metrics.json
    ├── plots/
    └── bundle/
        ├── model.keras
        ├── inference_config.json
        └── provenance.json
```

## Running Unit Tests
```bash
pytest tests/ -v
```
