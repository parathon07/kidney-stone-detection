"""Development-only transfer guard for a fold-checkpoint ensemble.

This script deliberately evaluates only the original development images.  It
averages the five fold-best checkpoint probabilities at slice level,
aggregates them by patient, and applies the same distributional transfer gates
used by the project evaluator against E019's frozen OOF patient predictions.
It never loads the sealed test split.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.data import create_dataset
from src.evaluate import _load_inference_model, aggregate_patient_level
from src.splits import load_and_validate_splits
from src.utils import load_config, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config) if args.config else (PROJECT_ROOT / "configs" / "default.yaml")
    config = load_config(str(config_path))
    if args.config is None:
        config["seed"] = 44
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "manifest.csv")
    merged = load_and_validate_splits(
        manifest,
        splits_dir=str(PROJECT_ROOT / "splits"),
    )
    dev_df = merged[
        (merged["split"] == "development")
        & (merged["is_original"] == True)
    ].copy().reset_index(drop=True)
    if dev_df.empty:
        raise ValueError("No original development images were found.")

    model_paths = [run_dir / f"fold_{fold}" / "best.keras" for fold in range(5)]
    missing = [str(path) for path in model_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing fold checkpoints: {missing}")

    data_dir = PROJECT_ROOT.parent / "kidney stone detection" / "data" / "dataset" / "Kindey Stone Dataset"
    if not data_dir.exists():
        raise FileNotFoundError(f"Frozen source-data directory not found: {data_dir}")
    dev_ds = create_dataset(
        dev_df,
        str(data_dir),
        config,
        split_type="val",
    )
    fold_probabilities = []
    for model_path in model_paths:
        model = _load_inference_model(str(model_path))
        raw_predictions = model.predict(dev_ds, verbose=0)
        if raw_predictions.shape != (len(dev_df), 2):
            raise ValueError(
                f"Unexpected prediction shape from {model_path}: "
                f"{raw_predictions.shape}"
            )
        fold_probabilities.append(np.asarray(raw_predictions[:, 1], dtype=float))
        del model

    slice_probabilities = np.mean(np.stack(fold_probabilities, axis=0), axis=0)
    if not np.isfinite(slice_probabilities).all():
        raise ValueError("Ensemble probabilities contain non-finite values.")

    prediction_columns = {
        "path": dev_df["path"].tolist(),
        "patient_id": dev_df["patient_id"].tolist(),
        "source_image_id": dev_df["source_image_id"].tolist(),
        "y_true": dev_df["label"].astype(int).tolist(),
        "stone_probability": slice_probabilities,
    }
    if "hospital" in dev_df.columns:
        prediction_columns["hospital"] = dev_df["hospital"].tolist()
    slice_predictions = pd.DataFrame(prediction_columns)
    patient_predictions = aggregate_patient_level(slice_predictions)
    patient_predictions.to_csv(
        output_dir / "development_predictions_patient_level.csv",
        index=False,
        float_format="%.17g",
    )

    oof_patient_path = run_dir / "oof_predictions_patient_level.csv"
    oof_metrics_path = run_dir / "oof_metrics.json"
    if not oof_patient_path.exists() or not oof_metrics_path.exists():
        raise FileNotFoundError("OOF predictions/metrics are required.")
    oof_patient = pd.read_csv(oof_patient_path)
    oof_metrics = json.loads(oof_metrics_path.read_text(encoding="utf-8"))
    threshold = float(oof_metrics["locked_threshold_metrics"]["threshold"])

    if set(patient_predictions["patient_id"]) != set(oof_patient["patient_id"]):
        raise ValueError("Development and OOF patient sets do not match.")
    oof_by_patient = oof_patient.set_index("patient_id")
    ordered_oof = oof_by_patient.loc[patient_predictions["patient_id"]]
    if not np.array_equal(
        patient_predictions["y_true"].to_numpy(dtype=int),
        ordered_oof["y_true"].to_numpy(dtype=int),
    ):
        raise ValueError("Development and OOF patient labels do not match.")

    oof_probabilities = ordered_oof["stone_probability"].to_numpy(dtype=float)
    dev_probabilities = patient_predictions["stone_probability"].to_numpy(dtype=float)
    oof_positive_rate = float(np.mean(oof_probabilities >= threshold))
    dev_positive_rate = float(np.mean(dev_probabilities >= threshold))
    gates = {
        "prob_std": {
            "value": float(np.std(dev_probabilities)),
            "passed": bool(np.std(dev_probabilities) >= 0.10),
        },
        "ks_vs_oof": {
            "value": float(ks_2samp(dev_probabilities, oof_probabilities).statistic),
            "passed": bool(ks_2samp(dev_probabilities, oof_probabilities).statistic < 0.15),
        },
        "threshold_transfer": {
            "oof_pos_rate": oof_positive_rate,
            "development_pos_rate": dev_positive_rate,
            "delta": abs(dev_positive_rate - oof_positive_rate),
            "passed": bool(abs(dev_positive_rate - oof_positive_rate) <= 0.10),
        },
    }
    result = {
        "patient_level": True,
        "ensemble": "equal_mean_of_five_fold_best_checkpoints",
        "n_patients": int(len(patient_predictions)),
        "n_development_slices": int(len(slice_predictions)),
        "probability_mean": float(np.mean(dev_probabilities)),
        "probability_std": float(np.std(dev_probabilities)),
        "ks_vs_oof": float(gates["ks_vs_oof"]["value"]),
        "locked_threshold": threshold,
        "gates": gates,
        "gates_passed": bool(all(gate["passed"] for gate in gates.values())),
        "sealed_test_used": False,
        "model_paths": [str(path) for path in model_paths],
    }
    save_json(result, str(output_dir / "development_transfer.json"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
