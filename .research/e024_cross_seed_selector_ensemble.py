"""Development-only cross-seed ensemble diagnostic for E019 and E023.

The two selector runs use the same frozen folds and original development
manifest.  Patient-level averaging is mathematically equivalent to averaging
the aligned slice predictions before aggregation, so this script reuses the
already verified prediction artifacts and does not retrain or access the
sealed test set.
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
from sklearn.metrics import roc_auc_score

from src.evaluate import (
    aggregate_patient_level,
    compute_metrics_at_threshold,
    compute_per_site_metrics,
    find_youden_j_threshold,
    validate_oof_predictions,
)
from src.splits import load_and_validate_splits
from src.utils import load_config, save_json


def _keyed_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_key"] = result["path"].astype(str).str.replace("\\", "/", regex=False)
    if result["_key"].duplicated().any():
        raise ValueError("Prediction artifact contains duplicate paths.")
    return result


def _average_aligned_predictions(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    left_keyed = _keyed_frame(left)
    right_keyed = _keyed_frame(right).set_index("_key")
    if set(left_keyed["_key"]) != set(right_keyed.index):
        raise ValueError("Prediction artifacts do not cover the same paths.")
    aligned_right = right_keyed.loc[left_keyed["_key"]]
    for column in ["patient_id", "source_image_id", "y_true", "fold"]:
        if column in left_keyed.columns and column in aligned_right.columns:
            if not np.array_equal(
                left_keyed[column].astype(str).to_numpy(),
                aligned_right[column].astype(str).to_numpy(),
            ):
                raise ValueError(f"Prediction identity mismatch in column {column!r}.")
    result = left_keyed.drop(columns=["_key"]).copy()
    result["stone_probability"] = (
        pd.to_numeric(left_keyed["stone_probability"], errors="raise").to_numpy(dtype=float)
        + pd.to_numeric(aligned_right["stone_probability"], errors="raise").to_numpy(dtype=float)
    ) / 2.0
    return result


def _load_oof(run_dir: Path) -> pd.DataFrame:
    frames = []
    for fold in range(5):
        path = run_dir / f"fold_{fold}" / "val_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def _fold_metrics(oof: pd.DataFrame) -> list[dict]:
    records = []
    for fold in range(5):
        patient = aggregate_patient_level(oof[oof["fold"] == fold].copy())
        records.append(
            {
                "fold": fold,
                "n_patients": int(len(patient)),
                "roc_auc": float(roc_auc_score(patient["y_true"], patient["stone_probability"])),
                "probability_std": float(np.std(patient["stone_probability"].to_numpy(dtype=float))),
            }
        )
    return records


def _oof_result(oof: pd.DataFrame, config: dict) -> tuple[dict, pd.DataFrame]:
    patient = aggregate_patient_level(oof.copy())
    y_true_patient = patient["y_true"].to_numpy(dtype=int)
    y_prob_patient = patient["stone_probability"].to_numpy(dtype=float)
    y_true_slice = oof["y_true"].to_numpy(dtype=int)
    y_prob_slice = oof["stone_probability"].to_numpy(dtype=float)
    threshold, youden_j = find_youden_j_threshold(y_true_patient, y_prob_patient)
    fold_metrics = _fold_metrics(oof)
    fold_aucs = [item["roc_auc"] for item in fold_metrics]
    acceptance_cfg = config.get("evaluation", {}).get("acceptance", {})
    min_patient_auc = float(acceptance_cfg.get("min_patient_roc_auc", 0.60))
    min_mean_auc = float(acceptance_cfg.get("min_mean_fold_roc_auc", 0.55))
    min_worst_auc = float(acceptance_cfg.get("min_worst_fold_roc_auc", 0.50))
    max_collapsed = int(acceptance_cfg.get("max_collapsed_folds", 0))
    collapsed = [
        item["fold"]
        for item in fold_metrics
        if item["probability_std"] < 0.03 or item["roc_auc"] < 0.45
    ]
    patient_auc = float(roc_auc_score(y_true_patient, y_prob_patient))
    mean_auc = float(np.mean(fold_aucs))
    worst_auc = float(np.min(fold_aucs))
    checks = {
        "patient_roc_auc": patient_auc >= min_patient_auc,
        "mean_fold_patient_roc_auc": mean_auc >= min_mean_auc,
        "worst_fold_patient_roc_auc": worst_auc >= min_worst_auc,
        "collapsed_fold_count": len(collapsed) <= max_collapsed,
    }
    result = {
        "total_oof_samples": int(len(oof)),
        "n_patients": int(len(patient)),
        "patient_mean_roc_auc": patient_auc,
        "slice_level_roc_auc": float(roc_auc_score(y_true_slice, y_prob_slice)),
        "locked_threshold_metrics": compute_metrics_at_threshold(
            y_true_patient, y_prob_patient, threshold=threshold
        ),
        "slice_level_locked_metrics": compute_metrics_at_threshold(
            y_true_slice, y_prob_slice, threshold=threshold
        ),
        "per_site_metrics": compute_per_site_metrics(patient),
        "best_epochs_per_fold": None,
        "mean_fold_patient_auc": mean_auc,
        "worst_fold_patient_auc": worst_auc,
        "fold_patient_auc": fold_aucs,
        "fold_patient_metrics": fold_metrics,
        "threshold": threshold,
        "youden_j": youden_j,
        "acceptance": {
            "passed": bool(all(checks.values())),
            "criteria": {
                "min_patient_roc_auc": min_patient_auc,
                "min_mean_fold_roc_auc": min_mean_auc,
                "min_worst_fold_roc_auc": min_worst_auc,
                "max_collapsed_folds": max_collapsed,
            },
            "checks": checks,
            "collapsed_folds": collapsed,
        },
        "acceptance_passed": bool(all(checks.values())),
    }
    return result, patient


def _average_patient_transfer(left_path: Path, right_path: Path) -> pd.DataFrame:
    left = pd.read_csv(left_path)
    right = pd.read_csv(right_path)
    left["_key"] = left["patient_id"].astype(str)
    right["_key"] = right["patient_id"].astype(str)
    if left["_key"].duplicated().any() or right["_key"].duplicated().any():
        raise ValueError("Development transfer artifacts contain duplicate patients.")
    right = right.set_index("_key")
    if set(left["_key"]) != set(right.index):
        raise ValueError("Development transfer artifacts do not cover the same patients.")
    aligned = right.loc[left["_key"]]
    for column in ["patient_id", "y_true"]:
        if not np.array_equal(left[column].astype(str), aligned[column].astype(str)):
            raise ValueError(f"Development transfer identity mismatch in {column!r}.")
    result = left.drop(columns=["_key"]).copy()
    result["stone_probability"] = (
        left["stone_probability"].to_numpy(dtype=float)
        + aligned["stone_probability"].to_numpy(dtype=float)
    ) / 2.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed44-run", required=True)
    parser.add_argument("--seed43-run", required=True)
    parser.add_argument("--seed44-transfer", required=True)
    parser.add_argument("--seed43-transfer", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(str(PROJECT_ROOT / "configs" / "default.yaml"))
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "manifest.csv")
    merged = load_and_validate_splits(manifest, splits_dir=str(PROJECT_ROOT / "splits"))
    dev_manifest = merged[merged["split"] == "development"].copy()

    seed44_oof = _load_oof(Path(args.seed44_run))
    seed43_oof = _load_oof(Path(args.seed43_run))
    blended_oof = _average_aligned_predictions(seed44_oof, seed43_oof)
    validate_oof_predictions(blended_oof, n_folds=5, expected_manifest_df=dev_manifest)
    oof_result, blended_patient = _oof_result(blended_oof, config)
    blended_oof.to_csv(output_dir / "oof_predictions.csv", index=False, float_format="%.17g")
    blended_patient.to_csv(
        output_dir / "oof_predictions_patient_level.csv", index=False, float_format="%.17g"
    )
    save_json(oof_result, str(output_dir / "oof_metrics.json"))

    seed44_transfer = Path(args.seed44_transfer) / "development_predictions_patient_level.csv"
    seed43_transfer = Path(args.seed43_transfer) / "development_predictions_patient_level.csv"
    blended_transfer = _average_patient_transfer(seed44_transfer, seed43_transfer)
    blended_transfer.to_csv(
        output_dir / "development_predictions_patient_level.csv",
        index=False,
        float_format="%.17g",
    )
    oof_probs = blended_patient["stone_probability"].to_numpy(dtype=float)
    dev_probs = blended_transfer["stone_probability"].to_numpy(dtype=float)
    threshold = float(oof_result["threshold"])
    oof_rate = float(np.mean(oof_probs >= threshold))
    dev_rate = float(np.mean(dev_probs >= threshold))
    transfer = {
        "patient_level": True,
        "ensemble": "equal_mean_of_e019_and_e023_fold_best_ensembles",
        "n_patients": int(len(blended_transfer)),
        "probability_mean": float(np.mean(dev_probs)),
        "probability_std": float(np.std(dev_probs)),
        "ks_vs_oof": float(ks_2samp(dev_probs, oof_probs).statistic),
        "locked_threshold": threshold,
        "oof_pos_rate": oof_rate,
        "development_pos_rate": dev_rate,
        "threshold_delta": abs(dev_rate - oof_rate),
        "gates": {
            "prob_std": {"value": float(np.std(dev_probs)), "passed": bool(np.std(dev_probs) >= 0.10)},
            "ks_vs_oof": {
                "value": float(ks_2samp(dev_probs, oof_probs).statistic),
                "passed": bool(ks_2samp(dev_probs, oof_probs).statistic < 0.15),
            },
            "threshold_transfer": {
                "oof_pos_rate": oof_rate,
                "development_pos_rate": dev_rate,
                "delta": abs(dev_rate - oof_rate),
                "passed": bool(abs(dev_rate - oof_rate) <= 0.10),
            },
        },
        "gates_passed": bool(
            np.std(dev_probs) >= 0.10
            and ks_2samp(dev_probs, oof_probs).statistic < 0.15
            and abs(dev_rate - oof_rate) <= 0.10
        ),
        "sealed_test_used": False,
    }
    save_json(transfer, str(output_dir / "development_transfer.json"))
    save_json(
        {
            "seed44_run": str(Path(args.seed44_run)),
            "seed43_run": str(Path(args.seed43_run)),
            "seed44_transfer": str(seed44_transfer),
            "seed43_transfer": str(seed43_transfer),
            "sealed_test_used": False,
        },
        str(output_dir / "provenance.json"),
    )
    print(json.dumps({"oof": oof_result, "transfer": transfer}, indent=2))


if __name__ == "__main__":
    main()
