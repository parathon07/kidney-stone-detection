"""Predeclared development-only weight grid for E019/E023 predictions."""

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
    find_youden_j_threshold,
    validate_oof_predictions,
)
from src.splits import load_and_validate_splits
from src.utils import load_config, save_json


def load_oof(run_dir: Path) -> pd.DataFrame:
    frames = []
    for fold in range(5):
        frames.append(pd.read_csv(run_dir / f"fold_{fold}" / "val_predictions.csv"))
    return pd.concat(frames, ignore_index=True)


def align(left: pd.DataFrame, right: pd.DataFrame, key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = left.copy()
    right = right.copy()
    left["_key"] = left[key].astype(str).str.replace("\\", "/", regex=False)
    right["_key"] = right[key].astype(str).str.replace("\\", "/", regex=False)
    if left["_key"].duplicated().any() or right["_key"].duplicated().any():
        raise ValueError(f"Duplicate {key} values in prediction artifacts.")
    right = right.set_index("_key").loc[left["_key"]]
    for column in ["patient_id", "source_image_id", "y_true", "fold"]:
        if column in left.columns and column in right.columns:
            if not np.array_equal(left[column].astype(str), right[column].astype(str)):
                raise ValueError(f"Identity mismatch in {column!r}.")
    return left, right


def weighted_predictions(left: pd.DataFrame, right: pd.DataFrame, weight_left: float, key: str) -> pd.DataFrame:
    left_aligned, right_aligned = align(left, right, key)
    result = left_aligned.drop(columns=["_key"]).copy()
    result["stone_probability"] = (
        weight_left * left_aligned["stone_probability"].to_numpy(dtype=float)
        + (1.0 - weight_left) * right_aligned["stone_probability"].to_numpy(dtype=float)
    )
    return result


def fold_metrics(oof: pd.DataFrame) -> list[float]:
    values = []
    for fold in range(5):
        patient = aggregate_patient_level(oof[oof["fold"] == fold].copy())
        values.append(float(roc_auc_score(patient["y_true"], patient["stone_probability"])))
    return values


def evaluate_weight(
    oof: pd.DataFrame,
    transfer: pd.DataFrame,
    config: dict,
    weight: float,
) -> dict:
    patient = aggregate_patient_level(oof.copy())
    y_true = patient["y_true"].to_numpy(dtype=int)
    oof_prob = patient["stone_probability"].to_numpy(dtype=float)
    dev_prob = transfer["stone_probability"].to_numpy(dtype=float)
    threshold, _ = find_youden_j_threshold(y_true, oof_prob)
    fold_auc = fold_metrics(oof)
    acceptance = config.get("evaluation", {}).get("acceptance", {})
    collapsed = [
        index for index, fold in enumerate(range(5))
        if float(np.std(aggregate_patient_level(oof[oof["fold"] == fold])["stone_probability"])) < 0.03
        or fold_auc[index] < 0.45
    ]
    oof_auc = float(roc_auc_score(y_true, oof_prob))
    oof_acceptance = bool(
        oof_auc >= float(acceptance.get("min_patient_roc_auc", 0.60))
        and float(np.mean(fold_auc)) >= float(acceptance.get("min_mean_fold_roc_auc", 0.55))
        and float(np.min(fold_auc)) >= float(acceptance.get("min_worst_fold_roc_auc", 0.50))
        and len(collapsed) <= int(acceptance.get("max_collapsed_folds", 0))
    )
    ks_value = float(ks_2samp(dev_prob, oof_prob).statistic)
    oof_rate = float(np.mean(oof_prob >= threshold))
    dev_rate = float(np.mean(dev_prob >= threshold))
    std_value = float(np.std(dev_prob))
    delta = abs(dev_rate - oof_rate)
    transfer_passed = bool(std_value >= 0.10 and ks_value < 0.15 and delta <= 0.10)
    return {
        "weight_e019": float(weight),
        "weight_e023": float(1.0 - weight),
        "patient_oof_auc": oof_auc,
        "slice_oof_auc": float(roc_auc_score(oof["y_true"], oof["stone_probability"])),
        "fold_patient_auc": fold_auc,
        "mean_fold_patient_auc": float(np.mean(fold_auc)),
        "worst_fold_patient_auc": float(np.min(fold_auc)),
        "oof_acceptance_passed": oof_acceptance,
        "locked_threshold": float(threshold),
        "probability_std": std_value,
        "ks_vs_oof": ks_value,
        "oof_positive_rate": oof_rate,
        "development_positive_rate": dev_rate,
        "threshold_delta": delta,
        "development_transfer_gates_passed": transfer_passed,
        "gates_passed": bool(oof_acceptance and transfer_passed),
        "sealed_test_used": False,
    }


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

    e019_oof = load_oof(Path(args.seed44_run))
    e023_oof = load_oof(Path(args.seed43_run))
    e019_transfer = pd.read_csv(Path(args.seed44_transfer) / "development_predictions_patient_level.csv")
    e023_transfer = pd.read_csv(Path(args.seed43_transfer) / "development_predictions_patient_level.csv")
    aligned_transfer_left, aligned_transfer_right = align(e019_transfer, e023_transfer, "patient_id")

    weights = [round(index / 20.0, 2) for index in range(21)]
    records = []
    candidates = {}
    for weight in weights:
        blended_oof = weighted_predictions(e019_oof, e023_oof, weight, "path")
        validate_oof_predictions(blended_oof, n_folds=5, expected_manifest_df=dev_manifest)
        blended_transfer = weighted_predictions(
            aligned_transfer_left,
            aligned_transfer_right.reset_index(drop=True),
            weight,
            "patient_id",
        )
        record = evaluate_weight(blended_oof, blended_transfer, config, weight)
        records.append(record)
        candidates[weight] = (blended_oof, blended_transfer)

    eligible = [record for record in records if record["gates_passed"]]
    selected = max(eligible, key=lambda record: (record["patient_oof_auc"], -record["weight_e019"])) if eligible else None
    grid = {
        "weights": weights,
        "records": records,
        "selected": selected,
        "selection_rule": "max patient_oof_auc among candidates passing OOF acceptance and all transfer gates; ties prefer lower E019 weight",
        "sealed_test_used": False,
    }
    save_json(grid, str(output_dir / "blend_weight_grid.json"))
    if selected is not None:
        selected_weight = float(selected["weight_e019"])
        selected_oof, selected_transfer = candidates[selected_weight]
        selected_oof.to_csv(output_dir / "selected_oof_predictions.csv", index=False, float_format="%.17g")
        selected_transfer.to_csv(
            output_dir / "selected_development_predictions_patient_level.csv",
            index=False,
            float_format="%.17g",
        )
    print(json.dumps(grid, indent=2))


if __name__ == "__main__":
    main()
