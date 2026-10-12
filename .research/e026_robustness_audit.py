"""Read-only robustness audit for the current BEST versus E000."""

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
from sklearn.metrics import roc_auc_score

from src.evaluate import compute_metrics_at_threshold
from src.utils import save_json


def metrics(frame: pd.DataFrame, probability: str, threshold: float) -> dict:
    y_true = frame["y_true"].to_numpy(dtype=int)
    y_prob = frame[probability].to_numpy(dtype=float)
    auc = None if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, y_prob))
    result = {
        "n_patients": int(len(frame)),
        "n_positive": int(np.sum(y_true == 1)),
        "roc_auc": auc,
        "probability_mean": float(np.mean(y_prob)) if len(y_prob) else None,
        "probability_std": float(np.std(y_prob)) if len(y_prob) else None,
    }
    if len(y_prob):
        threshold_metrics = compute_metrics_at_threshold(y_true, y_prob, threshold=threshold)
        result["threshold_metrics"] = {
            "threshold": threshold,
            "sensitivity": threshold_metrics["sensitivity_recall"],
            "specificity": threshold_metrics["specificity"],
            "precision": threshold_metrics["precision"],
            "f1": threshold_metrics["f1_score"],
            "tp": threshold_metrics["confusion_matrix"]["tp"],
            "tn": threshold_metrics["confusion_matrix"]["tn"],
            "fp": threshold_metrics["confusion_matrix"]["fp"],
            "fn": threshold_metrics["confusion_matrix"]["fn"],
        }
    return result


def grouped(frame: pd.DataFrame, group_column: str, probability: str, threshold: float) -> dict:
    result = {}
    for key, group in frame.groupby(group_column, dropna=False, sort=True):
        result[str(key)] = metrics(group, probability, threshold)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-threshold", type=float, required=True)
    parser.add_argument("--candidate-threshold", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(args.baseline)
    candidate = pd.read_csv(args.candidate)
    required = {"patient_id", "y_true", "n_slices", "stone_probability", "site"}
    if not required.issubset(baseline.columns) or not required.issubset(candidate.columns):
        raise ValueError("Patient prediction artifacts are missing required audit columns.")
    baseline = baseline.rename(columns={"stone_probability": "baseline_probability"})
    candidate = candidate.rename(columns={"stone_probability": "candidate_probability"})
    merged = baseline.merge(candidate, on="patient_id", how="outer", suffixes=("_baseline", "_candidate"))
    if merged.isna().any().any():
        raise ValueError("Baseline and candidate do not cover identical patients.")
    for column in ["y_true", "n_slices", "site"]:
        if not np.array_equal(
            merged[f"{column}_baseline"].astype(str), merged[f"{column}_candidate"].astype(str)
        ):
            raise ValueError(f"Patient metadata mismatch in {column!r}.")
    frame = pd.DataFrame(
        {
            "patient_id": merged["patient_id"],
            "y_true": merged["y_true_baseline"].astype(int),
            "n_slices": merged["n_slices_baseline"].astype(int),
            "site": merged["site_baseline"],
            "baseline_probability": merged["baseline_probability"].astype(float),
            "candidate_probability": merged["candidate_probability"].astype(float),
        }
    )
    frame["slice_bin"] = pd.cut(
        frame["n_slices"],
        bins=[0, 10, 20, 30, np.inf],
        labels=["1-10", "11-20", "21-30", "31+"],
        include_lowest=True,
    ).astype(str)
    frame["candidate_minus_baseline"] = frame["candidate_probability"] - frame["baseline_probability"]
    frame.to_csv(output_dir / "patient_comparison.csv", index=False, float_format="%.17g")

    summary = {
        "n_patients": int(len(frame)),
        "baseline": metrics(frame, "baseline_probability", args.baseline_threshold),
        "candidate": metrics(frame, "candidate_probability", args.candidate_threshold),
        "by_site": {
            "baseline": grouped(frame, "site", "baseline_probability", args.baseline_threshold),
            "candidate": grouped(frame, "site", "candidate_probability", args.candidate_threshold),
        },
        "by_slice_bin": {
            "baseline": grouped(frame, "slice_bin", "baseline_probability", args.baseline_threshold),
            "candidate": grouped(frame, "slice_bin", "candidate_probability", args.candidate_threshold),
        },
        "score_shift": {
            "mean": float(frame["candidate_minus_baseline"].mean()),
            "std": float(frame["candidate_minus_baseline"].std(ddof=0)),
            "mean_absolute": float(frame["candidate_minus_baseline"].abs().mean()),
        },
        "sealed_test_used": False,
        "source_baseline": str(Path(args.baseline)),
        "source_candidate": str(Path(args.candidate)),
    }
    save_json(summary, str(output_dir / "robustness_audit.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
