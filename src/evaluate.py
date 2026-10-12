import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
from typing import Dict, Any, Tuple, List, Optional
from src.utils import save_json, save_config, compute_sha256, get_metadata, load_json
from src.data import create_dataset
from src.model import Fp16BatchNorm


def _load_inference_model(model_path: str):
    """Load a model without optimizer/loss state for inference-only use."""
    return tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"Fp16BatchNorm": Fp16BatchNorm},
    )


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute comprehensive binary classification metrics per Section 8.1.
    Works for slice-level or patient-level arrays alike.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Safe AUC computations
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = float("nan")

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0
    balanced_accuracy = float((sensitivity + specificity) / 2.0)

    return {
        "threshold": float(threshold),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
        "balanced_accuracy": balanced_accuracy,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


def find_youden_j_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Determine threshold maximizing Youden's J = sensitivity + specificity - 1 per Section 8.2.
    Inf thresholds are clamped BEFORE argmax so a degenerate ROC can never silently
    select the predict-all-negative boundary; degenerate outcomes are reported loudly.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    thresholds = np.asarray(thresholds, dtype=float)
    thresholds = np.where(~np.isfinite(thresholds), 1.0, thresholds)

    # Youden's J = tpr - fpr = sensitivity + specificity - 1
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))

    best_threshold = float(np.clip(thresholds[best_idx], 0.0, 1.0))
    best_j = float(j_scores[best_idx])

    if best_j <= 0.0:
        print(
            f"WARNING: Degenerate ROC detected (max Youden's J = {best_j:.4f} <= 0). "
            "The model ranks worse than chance on this set; threshold is not meaningful."
        )
    return best_threshold, best_j


def aggregate_patient_level(
    df: pd.DataFrame,
    prob_col: str = "stone_probability",
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate slice-level predictions to patient level.

    Mean aggregation is the primary unit: with weak slice-level labels, averaging
    across a patient's slices outperforms max/top-k (measured: patient-mean OOF
    AUC 0.635 vs max 0.577 on the failed run; bootstrap CI excludes zero).
    """
    required = {"patient_id", "y_true", prob_col}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Prediction data is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Cannot aggregate an empty prediction data frame.")
    if agg not in {"mean", "median", "max", "min"}:
        raise ValueError(f"Unsupported patient aggregation: {agg}")
    if not np.isfinite(df[prob_col].to_numpy(dtype=float)).all():
        raise ValueError(f"Prediction column '{prob_col}' contains non-finite values.")

    label_counts = df.groupby("patient_id")["y_true"].nunique()
    if (label_counts > 1).any():
        bad = label_counts[label_counts > 1].index.tolist()
        raise ValueError(f"Conflicting slice labels within patients: {bad[:5]}")

    agg_df = df.groupby("patient_id").agg(
        y_true=("y_true", "first"),
        n_slices=(prob_col, "size"),
        stone_probability=(prob_col, agg),
    ).reset_index()

    agg_df["site"] = "unknown"
    if "hospital" in df.columns:
        hospital_by_patient = df.groupby("patient_id")["hospital"].first().astype(str)
        agg_df["site"] = agg_df["patient_id"].map(hospital_by_patient).fillna("unknown")
    elif "path" in df.columns:
        # Patient IDs are often just 'P001'; infer the site from the filename
        # only when the manifest-derived hospital column was not persisted.
        known_sites = {"FA", "ME", "RA"}

        def site_from_path(path: Any) -> str:
            filename = os.path.basename(str(path).replace("\\", "/"))
            for token in filename.rsplit(".", 1)[0].split("_"):
                token = token.upper()
                if token in known_sites:
                    return token
            return "unknown"

        sites = df.groupby("patient_id")["path"].first().map(site_from_path)
        agg_df["site"] = agg_df["patient_id"].map(sites).fillna("unknown")
    return agg_df


def compute_per_site_metrics(agg_df: pd.DataFrame) -> Dict[str, Any]:
    """Patient-level AUC per hospital/site code (e.g. FA/ME/RA) when both classes exist."""
    site_metrics: Dict[str, Any] = {}
    for site, sub in agg_df.groupby("site"):
        if sub["y_true"].nunique() < 2:
            site_metrics[str(site)] = {"n_patients": int(len(sub)), "roc_auc": None}
            continue
        try:
            auc = float(roc_auc_score(sub["y_true"].values, sub["stone_probability"].values))
        except Exception:
            auc = None
        site_metrics[str(site)] = {
            "n_patients": int(len(sub)),
            "n_positive": int(sub["y_true"].sum()),
            "roc_auc": auc,
        }
    return site_metrics


def run_collapse_warnings(y_true: np.ndarray, y_prob: np.ndarray, fold_ids: Optional[np.ndarray] = None) -> None:
    """Emit warnings when probability distributions look collapsed or near-chance."""
    pooled_std = float(np.std(y_prob))
    if pooled_std < 0.10:
        print(f"WARNING: Pooled probability std = {pooled_std:.4f} < 0.10 — outputs look collapsed toward the prior.")
    if fold_ids is not None:
        for fold in np.unique(fold_ids):
            mask = fold_ids == fold
            f_std = float(np.std(y_prob[mask]))
            if f_std < 0.03:
                print(f"WARNING: Fold {fold} probability std = {f_std:.4f} < 0.03 — fold collapsed.")
            try:
                f_auc = roc_auc_score(y_true[mask], y_prob[mask])
                if f_auc < 0.45:
                    print(f"WARNING: Fold {fold} AUC = {f_auc:.4f} < 0.45 — at or below chance.")
            except Exception:
                pass


def validate_oof_predictions(
    oof_df: pd.DataFrame,
    n_folds: int = 5,
    expected_manifest_df: Optional[pd.DataFrame] = None,
) -> None:
    """Assert that OOF predictions cover each expected validation row once."""
    required = {
        "path", "patient_id", "source_image_id", "y_true", "stone_probability", "fold"
    }
    missing = sorted(required.difference(oof_df.columns))
    if missing:
        raise ValueError(f"OOF predictions are missing required columns: {missing}")
    if oof_df.empty:
        raise ValueError("OOF predictions are empty.")
    if oof_df["path"].isna().any() or oof_df["path"].duplicated().any():
        raise ValueError("OOF predictions must contain each path exactly once.")
    if oof_df[["patient_id", "source_image_id", "y_true", "fold"]].isna().any().any():
        raise ValueError("OOF predictions contain null identity, label, or fold values.")

    y_true = pd.to_numeric(oof_df["y_true"], errors="coerce")
    folds = pd.to_numeric(oof_df["fold"], errors="coerce")
    probs = pd.to_numeric(oof_df["stone_probability"], errors="coerce")
    if y_true.isna().any() or not y_true.astype(int).isin([0, 1]).all():
        raise ValueError("OOF labels must be exactly 0 or 1.")
    if folds.isna().any() or (folds != folds.astype(int)).any() or not folds.astype(int).isin(range(n_folds)).all():
        raise ValueError(f"OOF fold values must be integers in 0..{n_folds - 1}.")
    if probs.isna().any() or not np.isfinite(probs.to_numpy()).all() or not probs.between(0.0, 1.0).all():
        raise ValueError("OOF stone probabilities must be finite values in [0, 1].")

    normalized_paths = oof_df["path"].astype(str).str.replace("\\", "/", regex=False)
    oof_df = oof_df.copy()
    oof_df["_normalized_path"] = normalized_paths
    if oof_df["_normalized_path"].duplicated().any():
        raise ValueError("OOF predictions contain duplicate normalized paths.")

    for fold in range(n_folds):
        fold_df = oof_df[folds.astype(int) == fold]
        if fold_df.empty:
            raise ValueError(f"OOF predictions are missing fold {fold}.")
        if not (pd.to_numeric(fold_df["fold"]) == fold).all():
            raise ValueError(f"OOF predictions in fold {fold} contain mismatched fold values.")

    patient_fold_counts = oof_df.groupby("patient_id")["fold"].nunique()
    if (patient_fold_counts > 1).any():
        bad = patient_fold_counts[patient_fold_counts > 1].index.tolist()
        raise ValueError(f"Patients appear in multiple OOF folds: {bad[:5]}")

    patient_source_counts = oof_df.groupby(["patient_id", "source_image_id"]).size()
    if (patient_source_counts > 1).any():
        raise ValueError("OOF predictions contain duplicate patient/source-image rows.")

    if expected_manifest_df is not None:
        required_manifest = {
            "path", "patient_id", "source_image_id", "label", "is_original", "split", "fold"
        }
        missing_manifest = sorted(required_manifest.difference(expected_manifest_df.columns))
        if missing_manifest:
            raise ValueError(f"Expected manifest is missing OOF columns: {missing_manifest}")
        expected = expected_manifest_df[
            (expected_manifest_df["split"] == "development")
            & (expected_manifest_df["is_original"] == True)
        ].copy()
        expected["_normalized_path"] = expected["path"].astype(str).str.replace("\\", "/", regex=False)
        if expected["_normalized_path"].duplicated().any():
            raise ValueError("Expected validation manifest contains duplicate paths.")
        expected_paths = set(expected["_normalized_path"])
        actual_paths = set(oof_df["_normalized_path"])
        missing_paths = expected_paths - actual_paths
        extra_paths = actual_paths - expected_paths
        if missing_paths or extra_paths:
            raise ValueError(
                "OOF coverage does not exactly match original development images "
                f"(missing={sorted(missing_paths)[:5]}, extra={sorted(extra_paths)[:5]})"
            )

        expected_by_path = expected.set_index("_normalized_path")
        actual_by_path = oof_df.set_index("_normalized_path")
        for expected_col, actual_col in (
            ("patient_id", "patient_id"),
            ("source_image_id", "source_image_id"),
            ("label", "y_true"),
            ("fold", "fold"),
        ):
            expected_values = expected_by_path[expected_col]
            actual_values = actual_by_path[actual_col].reindex(expected_values.index)
            if expected_col in {"label", "fold"}:
                matches = (
                    pd.to_numeric(expected_values, errors="coerce").to_numpy()
                    == pd.to_numeric(actual_values, errors="coerce").to_numpy()
                )
                matches = bool(np.all(matches))
            else:
                matches = expected_values.astype(str).equals(actual_values.astype(str))
            if not matches:
                raise ValueError(f"OOF metadata mismatch in column '{actual_col}'.")


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str) -> None:
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC Curve (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="#7f7f7f", linestyle="--", lw=1.5, label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str) -> None:
    """Plot and save Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_val = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="#2ca02c", lw=2, label=f"PR Curve (AUC = {pr_auc_val:.4f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix_heatmap(
    cm_dict: Dict[str, int],
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save Confusion Matrix heatmap."""
    cm_array = np.array([
        [cm_dict["tn"], cm_dict["fp"]],
        [cm_dict["fn"], cm_dict["tp"]],
    ])
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_cv_learning_curves(run_dir: str, n_folds: int, save_path: str) -> None:
    """Plot and save loss and accuracy across all CV folds."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for fold in range(n_folds):
        hist_path = os.path.join(run_dir, f"fold_{fold}", "history.csv")
        if os.path.exists(hist_path):
            df = pd.read_csv(hist_path)
            epochs = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)

            axes[0].plot(epochs, df["loss"], linestyle="--", alpha=0.6, label=f"F{fold} Train")
            axes[0].plot(epochs, df["val_loss"], linestyle="-", label=f"F{fold} Val")

            acc_col = "accuracy" if "accuracy" in df.columns else ("acc" if "acc" in df.columns else None)
            val_acc_col = "val_accuracy" if "val_accuracy" in df.columns else ("val_acc" if "val_acc" in df.columns else None)
            if acc_col and val_acc_col:
                axes[1].plot(epochs, df[acc_col], linestyle="--", alpha=0.6, label=f"F{fold} Train")
                axes[1].plot(epochs, df[val_acc_col], linestyle="-", label=f"F{fold} Val")

    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(visible=True, linestyle=":", alpha=0.6)
    handles0, _ = axes[0].get_legend_handles_labels()
    if handles0:
        axes[0].legend()

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(visible=True, linestyle=":", alpha=0.6)
    handles1, _ = axes[1].get_legend_handles_labels()
    if handles1:
        axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate_oof_and_select_threshold(
    run_dir: str,
    config: Dict[str, Any],
    n_folds: int = 5,
    expected_manifest_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Concatenate 5 validation folds, aggregate to patient level (mean of slice
    probabilities), select Youden's J threshold on patient-level OOF, and save
    OOF artifacts. Slice-level metrics are retained as secondary reference.
    """
    oof_dfs = []
    best_epochs = []

    for fold in range(n_folds):
        pred_path = os.path.join(run_dir, f"fold_{fold}", "val_predictions.csv")
        metrics_path = os.path.join(run_dir, f"fold_{fold}", "fold_metrics.json")

        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Missing fold {fold} predictions at {pred_path}")
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"Missing fold {fold} metrics at {metrics_path}")

        oof_dfs.append(pd.read_csv(pred_path))
        f_metrics = load_json(metrics_path)
        if f_metrics.get("fold", fold) != fold:
            raise ValueError(
                f"Integrity Error: fold_metrics.json at '{metrics_path}' has fold="
                f"{f_metrics.get('fold')!r}, expected {fold}"
            )
        val_best_epoch = f_metrics.get("best_epoch")
        if (
            val_best_epoch is None
            or isinstance(val_best_epoch, bool)
            or not isinstance(val_best_epoch, int)
            or val_best_epoch < 1
        ):
            raise ValueError(
                f"Integrity Error: fold_metrics.json at '{metrics_path}' must contain a positive integer 'best_epoch', got {val_best_epoch!r}"
            )
        if "val_sample_count" in f_metrics and f_metrics["val_sample_count"] != len(oof_dfs[-1]):
            raise ValueError(
                f"Integrity Error: fold {fold} val_sample_count does not match "
                "val_predictions.csv"
            )
        best_epochs.append(val_best_epoch)

    pooled_oof_df = pd.concat(oof_dfs, ignore_index=True)
    validate_oof_predictions(
        pooled_oof_df,
        n_folds=n_folds,
        expected_manifest_df=expected_manifest_df,
    )
    oof_csv_path = os.path.join(run_dir, "oof_predictions.csv")
    pooled_oof_df.to_csv(oof_csv_path, index=False, float_format="%.17g")

    # Collapse diagnostics on slice-level OOF
    run_collapse_warnings(
        pooled_oof_df["y_true"].values,
        pooled_oof_df["stone_probability"].values,
        fold_ids=pooled_oof_df["fold"].values,
    )

    # Patient-level aggregation (primary evaluation unit)
    patient_oof_df = aggregate_patient_level(pooled_oof_df)
    patient_csv_path = os.path.join(run_dir, "oof_predictions_patient_level.csv")
    patient_oof_df.to_csv(patient_csv_path, index=False, float_format="%.17g")

    y_true_pat = patient_oof_df["y_true"].values
    y_prob_pat = patient_oof_df["stone_probability"].values
    y_true_slice = pooled_oof_df["y_true"].values
    y_prob_slice = pooled_oof_df["stone_probability"].values

    # Record patient-level fold stability separately from pooled OOF quality.
    fold_patient_metrics: List[Dict[str, Any]] = []
    for fold in range(n_folds):
        fold_patient_df = aggregate_patient_level(
            pooled_oof_df[pooled_oof_df["fold"] == fold]
        )
        fold_auc = float("nan")
        if fold_patient_df["y_true"].nunique() >= 2:
            fold_auc = float(
                roc_auc_score(
                    fold_patient_df["y_true"].values,
                    fold_patient_df["stone_probability"].values,
                )
            )
        fold_patient_metrics.append(
            {
                "fold": fold,
                "n_patients": int(len(fold_patient_df)),
                "roc_auc": fold_auc,
                "probability_std": float(fold_patient_df["stone_probability"].std(ddof=0)),
            }
        )

    # Determine Youden J threshold on patient-level OOF
    best_thresh, best_j = find_youden_j_threshold(y_true_pat, y_prob_pat)

    # Compute metrics at locked threshold and 0.5 reference
    locked_metrics = compute_metrics_at_threshold(y_true_pat, y_prob_pat, threshold=best_thresh)
    ref_0_5_metrics = compute_metrics_at_threshold(y_true_pat, y_prob_pat, threshold=0.5)
    slice_locked_metrics = compute_metrics_at_threshold(y_true_slice, y_prob_slice, threshold=best_thresh)
    slice_auc = float(roc_auc_score(y_true_slice, y_prob_slice))
    patient_auc = float(roc_auc_score(y_true_pat, y_prob_pat))

    acceptance_cfg = config.get("evaluation", {}).get("acceptance", {})
    min_patient_auc = float(acceptance_cfg.get("min_patient_roc_auc", 0.60))
    min_mean_fold_auc = float(acceptance_cfg.get("min_mean_fold_roc_auc", 0.55))
    min_worst_fold_auc = float(acceptance_cfg.get("min_worst_fold_roc_auc", 0.50))
    max_collapsed_folds = int(acceptance_cfg.get("max_collapsed_folds", 0))
    valid_fold_aucs = [m["roc_auc"] for m in fold_patient_metrics if np.isfinite(m["roc_auc"])]
    mean_fold_auc = float(np.mean(valid_fold_aucs)) if len(valid_fold_aucs) == n_folds else float("nan")
    worst_fold_auc = float(np.min(valid_fold_aucs)) if len(valid_fold_aucs) == n_folds else float("nan")
    collapsed_folds = [
        m["fold"] for m in fold_patient_metrics
        if m["probability_std"] < 0.03 or not np.isfinite(m["roc_auc"]) or m["roc_auc"] < 0.45
    ]
    acceptance_checks = {
        "patient_roc_auc": bool(patient_auc >= min_patient_auc),
        "mean_fold_patient_roc_auc": bool(np.isfinite(mean_fold_auc) and mean_fold_auc >= min_mean_fold_auc),
        "worst_fold_patient_roc_auc": bool(np.isfinite(worst_fold_auc) and worst_fold_auc >= min_worst_fold_auc),
        "collapsed_fold_count": bool(len(collapsed_folds) <= max_collapsed_folds),
    }
    acceptance = {
        "passed": bool(all(acceptance_checks.values())),
        "criteria": {
            "min_patient_roc_auc": min_patient_auc,
            "min_mean_fold_roc_auc": min_mean_fold_auc,
            "min_worst_fold_roc_auc": min_worst_fold_auc,
            "max_collapsed_folds": max_collapsed_folds,
        },
        "checks": acceptance_checks,
        "patient_roc_auc": patient_auc,
        "mean_fold_patient_roc_auc": mean_fold_auc,
        "worst_fold_patient_roc_auc": worst_fold_auc,
        "collapsed_folds": collapsed_folds,
        "fold_patient_metrics": fold_patient_metrics,
    }
    if not acceptance["passed"]:
        print(
            "WARNING: OOF acceptance gate FAILED. This run is not eligible for "
            f"finalization: {acceptance}"
        )

    if locked_metrics["specificity"] < 0.10 or locked_metrics["sensitivity_recall"] < 0.10:
        print(
            f"WARNING: Locked threshold {best_thresh:.4f} yields degenerate operating point "
            f"(sens={locked_metrics['sensitivity_recall']:.3f}, spec={locked_metrics['specificity']:.3f}). "
            "Inspect the patient-level ROC before trusting downstream results."
        )

    threshold_info = {
        "strategy": "youden_j_oof_patient_mean",
        "threshold": best_thresh,
        "youden_j": best_j,
        "reference_0_5_metrics": ref_0_5_metrics,
        "patient_level": True,
    }
    save_json(threshold_info, os.path.join(run_dir, "threshold.json"))

    oof_results = {
        "total_oof_samples": len(pooled_oof_df),
        "patient_level": True,
        "n_patients": int(len(patient_oof_df)),
        "patient_mean_roc_auc": patient_auc,
        "slice_level_roc_auc": slice_auc,
        "locked_threshold_metrics": locked_metrics,
        "slice_level_locked_metrics": slice_locked_metrics,
        "per_site_metrics": compute_per_site_metrics(patient_oof_df),
        "best_epochs_per_fold": best_epochs,
        "median_best_epoch": int(round(float(np.median(best_epochs)))),
        "acceptance": acceptance,
        "acceptance_passed": acceptance["passed"],
    }
    save_json(oof_results, os.path.join(run_dir, "oof_metrics.json"))

    # Generate Plots (patient-level primary)
    plots_dir = os.path.join(run_dir, "plots")
    class_names = config.get("evaluation", {}).get("class_names", ["Non-Stone", "Stone"])
    plot_roc_curve(y_true_pat, y_prob_pat, os.path.join(plots_dir, "roc_curve.png"))
    plot_pr_curve(y_true_pat, y_prob_pat, os.path.join(plots_dir, "pr_curve.png"))
    plot_confusion_matrix_heatmap(
        locked_metrics["confusion_matrix"],
        class_names=class_names,
        save_path=os.path.join(plots_dir, "confusion_matrix.png"),
        title=f"OOF Patient-Level Confusion Matrix (Threshold={best_thresh:.4f})",
    )
    plot_cv_learning_curves(run_dir, n_folds=n_folds, save_path=os.path.join(plots_dir, "learning_curves.png"))

    return oof_results


def evaluate_development_transfer(
    development_manifest_df: pd.DataFrame,
    data_dir: str,
    model_path: str,
    locked_threshold: float,
    run_dir: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Check final-model probability transfer using development data only.

    This pre-test guard catches a collapsed or miscalibrated final retrain before
    the sealed test set is opened. The comparison is distributional because the
    final model was trained on all development patients.
    """
    final_dir = os.path.join(run_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    dev_df = development_manifest_df[
        (development_manifest_df["split"] == "development")
        & (development_manifest_df["is_original"] == True)
    ].copy().reset_index(drop=True)
    if dev_df.empty:
        raise ValueError("Development transfer check requires original development images.")

    dev_ds = create_dataset(dev_df, data_dir, config, split_type="val")
    model = _load_inference_model(model_path)
    raw_preds = model.predict(dev_ds, verbose=0)
    if raw_preds.shape != (len(dev_df), 2):
        raise ValueError(f"Final model predictions must have shape (n, 2), got {raw_preds.shape}")

    dev_pred_df = pd.DataFrame({
        "path": dev_df["path"].tolist(),
        "patient_id": dev_df["patient_id"].tolist(),
        "source_image_id": dev_df["source_image_id"].tolist(),
        "y_true": dev_df["label"].astype(int).tolist(),
        "stone_probability": raw_preds[:, 1],
    })
    dev_patient_df = aggregate_patient_level(dev_pred_df)
    dev_patient_df.to_csv(
        os.path.join(final_dir, "development_predictions_patient_level.csv"),
        index=False,
        float_format="%.17g",
    )

    oof_patient_csv_path = os.path.join(run_dir, "oof_predictions_patient_level.csv")
    if not os.path.exists(oof_patient_csv_path):
        raise FileNotFoundError(
            "OOF patient predictions are required for the development transfer check."
        )
    oof_patient_df = pd.read_csv(oof_patient_csv_path)
    oof_probs = pd.to_numeric(oof_patient_df["stone_probability"], errors="coerce").to_numpy()
    if oof_probs.size == 0 or not np.isfinite(oof_probs).all():
        raise ValueError("OOF patient predictions contain no valid probabilities.")

    dev_probs = dev_patient_df["stone_probability"].to_numpy(dtype=float)
    if not np.isfinite(dev_probs).all():
        raise ValueError("Final development predictions contain non-finite probabilities.")

    dev_std = float(np.std(dev_probs))
    ks_stat = float(ks_2samp(dev_probs, oof_probs).statistic)
    oof_pos_rate = float(np.mean(oof_probs >= locked_threshold))
    dev_pos_rate = float(np.mean(dev_probs >= locked_threshold))
    transfer_delta = abs(dev_pos_rate - oof_pos_rate)
    gates = {
        "prob_std": {"value": dev_std, "passed": bool(dev_std >= 0.10)},
        "ks_vs_oof": {"value": ks_stat, "passed": bool(ks_stat < 0.15)},
        "threshold_transfer": {
            "oof_pos_rate": oof_pos_rate,
            "development_pos_rate": dev_pos_rate,
            "delta": transfer_delta,
            "passed": bool(transfer_delta <= 0.10),
        },
    }
    result = {
        "patient_level": True,
        "n_patients": int(len(dev_patient_df)),
        "probability_mean": float(np.mean(dev_probs)),
        "probability_std": dev_std,
        "ks_vs_oof": ks_stat,
        "locked_threshold": float(locked_threshold),
        "gates": gates,
        "gates_passed": bool(all(gate["passed"] for gate in gates.values())),
    }
    save_json(result, os.path.join(final_dir, "development_transfer.json"))

    if not result["gates_passed"]:
        print(
            "WARNING: Final development transfer gates failed; sealed test evaluation "
            "will be skipped."
        )
    return result


def evaluate_sealed_test_set(
    test_manifest_df: pd.DataFrame,
    data_dir: str,
    model_path: str,
    locked_threshold: float,
    run_dir: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate once on the sealed 40-patient test set per Section 8.4:
      - Uses original images only.
      - Predicts once with final trained model.
      - Saves test_predictions.csv before metrics calculation.
      - Computes patient-level (primary) and slice-level (secondary) metrics at the locked threshold.
      - Runs collapse/transfer gates; results recorded in 'gates_passed'.
    """
    final_dir = os.path.join(run_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    # Filter original test images
    test_df = test_manifest_df[
        (test_manifest_df["split"] == "test") & (test_manifest_df["is_original"] == True)
    ].copy().reset_index(drop=True)

    test_ds = create_dataset(test_df, data_dir, config, split_type="test")
    model = _load_inference_model(model_path)

    # 1. Predict and save raw probabilities
    raw_preds = model.predict(test_ds, verbose=0)
    stone_probs = raw_preds[:, 1]
    y_true = test_df["label"].astype(int).values

    test_pred_df = pd.DataFrame({
        "path": test_df["path"],
        "patient_id": test_df["patient_id"],
        "source_image_id": test_df["source_image_id"],
        "y_true": y_true,
        "stone_probability": stone_probs,
        "predicted_class": (stone_probs >= locked_threshold).astype(int),
    })
    test_pred_csv = os.path.join(final_dir, "test_predictions.csv")
    test_pred_df.to_csv(test_pred_csv, index=False, float_format="%.17g")

    # 2. Patient-level (primary) and slice-level (secondary) metrics
    patient_test_df = aggregate_patient_level(test_pred_df)
    patient_test_df.to_csv(
        os.path.join(final_dir, "test_predictions_patient_level.csv"),
        index=False,
        float_format="%.17g",
    )

    y_true_pat = patient_test_df["y_true"].values
    y_prob_pat = patient_test_df["stone_probability"].values

    test_metrics = compute_metrics_at_threshold(y_true_pat, y_prob_pat, threshold=locked_threshold)
    test_metrics["patient_level"] = True
    test_metrics["n_patients"] = int(len(patient_test_df))
    test_metrics["patient_mean_roc_auc"] = float(roc_auc_score(y_true_pat, y_prob_pat))
    test_metrics["slice_level_metrics"] = compute_metrics_at_threshold(y_true, stone_probs, threshold=locked_threshold)
    test_metrics["per_site_metrics"] = compute_per_site_metrics(patient_test_df)

    # 3. Collapse / transfer gates (compare patient-level distributions)
    gates: Dict[str, Any] = {}
    oof_patient_csv_path = os.path.join(run_dir, "oof_predictions_patient_level.csv")
    if os.path.exists(oof_patient_csv_path):
        oof_patient_df = pd.read_csv(oof_patient_csv_path)
        if "stone_probability" not in oof_patient_df.columns:
            raise ValueError("OOF patient predictions are missing stone_probability.")
        oof_probs = pd.to_numeric(oof_patient_df["stone_probability"], errors="coerce").to_numpy()
        if oof_probs.size == 0 or not np.isfinite(oof_probs).all():
            raise ValueError("OOF patient predictions contain no valid probabilities.")

        test_std = float(np.std(y_prob_pat))
        gates["prob_std"] = {"value": test_std, "passed": bool(test_std >= 0.10)}

        try:
            ks_stat = float(ks_2samp(y_prob_pat, oof_probs).statistic)
            gates["ks_vs_oof"] = {"value": ks_stat, "passed": bool(ks_stat < 0.15)}
        except Exception:
            gates["ks_vs_oof"] = {"value": None, "passed": False}

        oof_pos_rate_at_thr = float(np.mean(oof_probs >= locked_threshold))
        test_pos_rate_at_thr = float(np.mean(y_prob_pat >= locked_threshold))
        transfer_delta = abs(test_pos_rate_at_thr - oof_pos_rate_at_thr)
        gates["threshold_transfer"] = {
            "oof_pos_rate": oof_pos_rate_at_thr,
            "test_pos_rate": test_pos_rate_at_thr,
            "delta": transfer_delta,
            "passed": bool(transfer_delta <= 0.10),
        }
    else:
        gates["oof_reference"] = {
            "passed": False,
            "note": "oof_predictions_patient_level.csv not found; gates cannot be evaluated",
        }

    gates_passed = all(g.get("passed", True) for g in gates.values())
    test_metrics["gates"] = gates
    test_metrics["gates_passed"] = bool(gates_passed)
    save_json(test_metrics, os.path.join(final_dir, "test_metrics.json"))

    # 4. Generate test plots (patient-level primary)
    test_plots_dir = os.path.join(final_dir, "plots")
    class_names = config.get("evaluation", {}).get("class_names", ["Non-Stone", "Stone"])
    plot_roc_curve(y_true_pat, y_prob_pat, os.path.join(test_plots_dir, "test_roc_curve.png"))
    plot_pr_curve(y_true_pat, y_prob_pat, os.path.join(test_plots_dir, "test_pr_curve.png"))
    plot_confusion_matrix_heatmap(
        test_metrics["confusion_matrix"],
        class_names=class_names,
        save_path=os.path.join(test_plots_dir, "test_confusion_matrix.png"),
        title=f"Final Test Patient-Level Confusion Matrix (Threshold={locked_threshold:.4f})",
    )

    # 5. Report gate status loudly
    if gates_passed:
        print("Collapse/transfer gates: ALL PASSED.")
    else:
        print("\n" + "=" * 70)
        print("WARNING: COLLAPSE/TRANSFER GATES FAILED — DO NOT DEPLOY THIS BUNDLE")
        for name, g in gates.items():
            if not g.get("passed", True):
                print(f"  FAILED gate '{name}': {g}")
        print("The final model's calibration does not match the CV models' OOF regime")
        print("or its outputs have collapsed. Bundle export will be skipped.")
        print("=" * 70 + "\n")

    return test_metrics


def export_inference_bundle(
    run_dir: str,
    final_model_path: str,
    locked_threshold: float,
    config: Dict[str, Any],
    metadata: Dict[str, Any],
    gates_passed: bool = False,
) -> str:
    """
    Export inference bundle per Section 10.1:
      runs/<run_id>/final/bundle/
        ├── model.keras
        ├── inference_config.json
        └── provenance.json
    Export is refused when collapse/transfer gates failed.
    """
    if not gates_passed:
        raise RuntimeError(
            "Inference bundle export REFUSED: collapse/transfer gates failed. "
            "The sealed-test evaluation shows the final model's calibration is "
            "inconsistent with the OOF regime. Re-train or recalibrate before deploying."
        )

    bundle_dir = os.path.join(run_dir, "final", "bundle")
    os.makedirs(bundle_dir, exist_ok=True)

    # 1. Copy model.keras
    bundle_model_path = os.path.join(bundle_dir, "model.keras")
    shutil.copyfile(final_model_path, bundle_model_path)

    # 2. Write inference_config.json matching spec Section 10.1
    inference_config = {
        "image_size": config.get("data", {}).get("image_size", 384),
        "channels": config.get("data", {}).get("channels", 1),
        "resize": "resize_with_pad",
        "scale": "0_1",
        "class_names": config.get("evaluation", {}).get("class_names", ["Non-Stone", "Stone"]),
        "stone_class_index": 1,
        "threshold": locked_threshold,
        "patient_aggregation": "mean",
    }
    save_json(inference_config, os.path.join(bundle_dir, "inference_config.json"))

    # 3. Write provenance.json
    provenance = {
        "run_id": metadata.get("run_id"),
        "timestamp": metadata.get("timestamp"),
        "git_commit": metadata.get("git_commit"),
        "dataset_manifest_hash": metadata.get("dataset_manifest_hash"),
        "outer_split_hash": metadata.get("outer_split_hash"),
        "cv_fold_hash": metadata.get("cv_fold_hash"),
        "model_sha256": compute_sha256(bundle_model_path),
        "locked_threshold": locked_threshold,
        "patient_aggregation": "mean",
    }
    save_json(provenance, os.path.join(bundle_dir, "provenance.json"))

    print(f"Inference bundle successfully exported to: {bundle_dir}")
    return bundle_dir
