"""Build reproducible confusion-matrix images from preserved research artifacts.

This script only reads existing OOF/transfer artifacts and writes presentation
images plus a machine-readable manifest inside METRICS_PACK.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PACK = Path(__file__).resolve().parent
ISOLATED = PACK.parent
RUNS = ISOLATED / "runs"
OUT = PACK / "graphs" / "confusion_matrices"
OUT.mkdir(parents=True, exist_ok=True)


RUN_SPECS = [
    {
        "key": "historical_20260826",
        "short_label": "Historical 20260826",
        "label": "Historical pre-autoresearch run",
        "run": "20260826_231405_460200",
        "note": "Earlier development OOF result; its sealed-test transfer is shown separately and is invalidated.",
    },
    {
        "key": "e000_baseline",
        "short_label": "E000 baseline",
        "label": "E000 baseline",
        "run": "20260827_173706_573400",
        "note": "Experiment 0: the unchanged, shuffle-remediated baseline used as the comparison anchor.",
    },
    {
        "key": "e004_seed43",
        "short_label": "E004 seed 43",
        "label": "E004 seed-43 complete CV",
        "run": "20260909_230224_165100",
        "note": "Strong OOF candidate, later rejected because the locked final retrain did not transfer its score regime.",
    },
    {
        "key": "e014_seed44",
        "short_label": "E014 seed 44",
        "label": "E014 seed-44 recovered CV",
        "run": "20260910_e014_seed44_recovered",
        "note": "Independent seed confirmation; complete but not a meaningful improvement over E000.",
    },
    {
        "key": "e015_cross_seed",
        "short_label": "E015 cross-seed",
        "label": "E015 cross-seed OOF ensemble",
        "run": "20260910_e015_seed43_44_oof_ensemble",
        "note": "OOF near-miss: higher patient AUC, but the development transfer KS gate failed.",
    },
    {
        "key": "e019_seed44_selector",
        "short_label": "E019 seed 44 selector",
        "label": "E019 seed-44 patient-AUC selector CV",
        "run": "20260910_e019_seed44_patient_auc_cv",
        "note": "Selector-based seed-44 CV; its five-fold ensemble transfer is a separate diagnostic.",
    },
    {
        "key": "e023_current_best",
        "short_label": "E023 CURRENT BEST",
        "label": "E023 seed-43 patient-AUC selector CV",
        "run": "20260911_e023_seed43_patient_auc_cv",
        "note": "Current development-only champion. It passed OOF integrity, OOF gates, transfer gates, and final verification.",
    },
    {
        "key": "e024_cross_seed",
        "short_label": "E024 cross-seed blend",
        "label": "E024 cross-seed selector ensemble",
        "run": "20260911_e024_cross_seed_selector_ensemble",
        "note": "OOF score was slightly higher than E023, but transfer KS failed, so it was rejected.",
    },
]


TRANSFER_SPECS = [
    {
        "key": "e005_final_retrain",
        "short_label": "E005 final retrain",
        "label": "E005 final-development retrain",
        "run": "20260909_230224_165100",
        "csv": "final/development_predictions_patient_level.csv",
        "transfer": "final/development_transfer.json",
        "note": "Transfer diagnostic only; not a generalization result. All development patients were predicted positive.",
    },
    {
        "key": "e006_epoch1",
        "short_label": "E006 epoch 1",
        "label": "E006 one-epoch final retrain",
        "run": "20260910_e006_final_epoch1",
        "csv": "final/development_predictions_patient_level.csv",
        "transfer": "final/development_transfer.json",
        "note": "Transfer diagnostic only; not a generalization result. The score regime collapsed toward negative predictions.",
    },
    {
        "key": "e007_bn",
        "short_label": "E007 BN recalibration",
        "label": "E007 three-pass BN recalibration",
        "run": "20260910_e007_bn_recalibration",
        "csv": "final/development_predictions_patient_level.csv",
        "transfer": "final/development_transfer.json",
        "note": "Transfer diagnostic only; not a generalization result. Prevalence recovered, but KS remained too high.",
    },
    {
        "key": "e008_bn_onepass",
        "short_label": "E008 BN one-pass",
        "label": "E008 one-pass BN recalibration",
        "run": "20260910_e008_bn_onepass",
        "csv": "final/development_predictions_patient_level.csv",
        "transfer": "final/development_transfer.json",
        "note": "Transfer diagnostic only; not a generalization result. One pass did not materially change the BN failure.",
    },
    {
        "key": "e009_equal_mean",
        "short_label": "E009 equal mean",
        "label": "E009 equal-mean fold ensemble",
        "run": "20260910_e009_fold_ensemble",
        "csv": "development_predictions_patient_level.csv",
        "transfer": "development_transfer.json",
        "note": "Transfer diagnostic only; not a generalization result. A serious near-miss that failed the strict KS gate.",
    },
    {
        "key": "e010_median",
        "short_label": "E010 median",
        "label": "E010 median fold ensemble",
        "run": "20260910_e010_fold_ensemble_median",
        "csv": "development_predictions_patient_level.csv",
        "transfer": "development_transfer.json",
        "note": "Transfer diagnostic only; not a generalization result. Median aggregation was worse than equal mean.",
    },
    {
        "key": "e020_seed44_transfer",
        "short_label": "E020 seed 44 transfer",
        "label": "E020 seed-44 five-checkpoint transfer",
        "run": "20260910_e019_seed44_patient_auc_cv",
        "csv": "final_ensemble_transfer/development_predictions_patient_level.csv",
        "transfer": "final_ensemble_transfer/development_transfer.json",
        "note": "Transfer diagnostic only; not a generalization result. Spread and prevalence passed, but KS failed.",
    },
    {
        "key": "e021_final_retrain",
        "short_label": "E021 final retrain",
        "label": "E021 exact-26-epoch final retrain",
        "run": "20260911_e021_seed44_final_epoch26_transfer",
        "csv": "final/development_predictions_patient_level.csv",
        "transfer": "final/development_transfer.json",
        "note": "Transfer diagnostic only; not a generalization result. Single-model retraining did not preserve the CV score regime.",
    },
    {
        "key": "e023_transfer",
        "short_label": "E023 transfer",
        "label": "E023 five-checkpoint development transfer",
        "run": "20260911_e023_seed43_patient_auc_cv",
        "csv": "final_ensemble_transfer/development_predictions_patient_level.csv",
        "transfer": "final_ensemble_transfer/development_transfer.json",
        "note": "Transfer diagnostic only; not a generalization result. This is the current champion's development-only guard.",
    },
    {
        "key": "e024_transfer",
        "short_label": "E024 transfer",
        "label": "E024 cross-seed development transfer",
        "run": "20260911_e024_cross_seed_selector_ensemble",
        "csv": "development_predictions_patient_level.csv",
        "transfer": "development_transfer.json",
        "note": "Transfer diagnostic only; not a generalization result. Despite strong threshold counts, KS failed.",
    },
]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def confusion(metrics: dict) -> dict[str, int]:
    matrix = metrics.get("confusion_matrix", {})
    return {
        "tn": int(matrix.get("tn", 0)),
        "fp": int(matrix.get("fp", 0)),
        "fn": int(matrix.get("fn", 0)),
        "tp": int(matrix.get("tp", 0)),
    }


def derived_metrics(cm: dict[str, int]) -> dict[str, float]:
    tn, fp, fn, tp = (cm[name] for name in ("tn", "fp", "fn", "tp"))
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "accuracy": (tn + tp) / (tn + fp + fn + tp) if tn + fp + fn + tp else 0.0,
    }


def metric_line(metrics: dict, cm: dict[str, int]) -> str:
    values = {**derived_metrics(cm), **metrics}
    return (
        f"ROC-AUC {values.get('roc_auc', float('nan')):.3f} | "
        f"PR-AUC {values.get('pr_auc', float('nan')):.3f} | "
        f"Sens {values.get('sensitivity_recall', float('nan')):.3f} | "
        f"Spec {values.get('specificity', float('nan')):.3f} | "
        f"F1 {values.get('f1_score', float('nan')):.3f}"
    )


def count_line(cm: dict[str, int]) -> str:
    return f"TN {cm['tn']} | FP {cm['fp']} | FN {cm['fn']} | TP {cm['tp']}"


def cm_from_prediction_csv(path: Path, threshold: float) -> dict[str, int]:
    cm = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            actual_positive = int(float(row["y_true"])) == 1
            predicted_positive = float(row["stone_probability"]) >= threshold
            if actual_positive and predicted_positive:
                cm["tp"] += 1
            elif actual_positive:
                cm["fn"] += 1
            elif predicted_positive:
                cm["fp"] += 1
            else:
                cm["tn"] += 1
    return cm


def render_matrix(
    *,
    matrix: np.ndarray,
    title: str,
    subtitle: str,
    footer: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.1), dpi=220)
    maximum = max(int(matrix.max()), 1)
    ax.imshow(matrix, cmap="Greys", vmin=0, vmax=maximum, aspect="equal")
    ax.set_xticks([0, 1], ["Predicted Negative", "Predicted Positive"])
    ax.set_yticks([0, 1], ["Actual Negative", "Actual Positive"])
    ax.tick_params(axis="both", labelsize=10, length=0, pad=8)
    ax.set_xlabel("Model prediction", fontsize=11, labelpad=12)
    ax.set_ylabel("Ground truth", fontsize=11, labelpad=12)
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold", color="#111827", pad=18)
    fig.text(
        0.10,
        0.87,
        fill(subtitle, 92),
        ha="left",
        va="top",
        fontsize=9.5,
        color="#4b5563",
    )

    for row in range(2):
        for col in range(2):
            value = int(matrix[row, col])
            ink = "white" if value / maximum > 0.52 else "#111827"
            ax.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=23,
                fontweight="bold",
                color=ink,
            )

    for edge in ax.spines.values():
        edge.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.text(
        0.10,
        0.075,
        fill(footer, 104),
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#4b5563",
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.18, right=0.96, top=0.73, bottom=0.22)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_overview(
    entries: list[dict],
    *,
    title: str,
    subtitle: str,
    output_path: Path,
) -> None:
    columns = 5 if len(entries) >= 9 else 4
    rows = (len(entries) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(13.5, 3.8 * rows), dpi=180)
    axes = np.atleast_1d(axes).reshape(rows, columns)
    for axis in axes.flat:
        axis.axis("off")

    for axis, entry in zip(axes.flat, entries):
        matrix = np.array(entry["matrix"], dtype=int)
        maximum = max(int(matrix.max()), 1)
        axis.imshow(matrix, cmap="Greys", vmin=0, vmax=maximum, aspect="equal")
        axis.set_xticks([0, 1], ["Pred -", "Pred +"])
        axis.set_yticks([0, 1], ["Actual -", "Actual +"])
        axis.tick_params(axis="both", labelsize=7, length=0, pad=3)
        for row in range(2):
            for col in range(2):
                value = int(matrix[row, col])
                ink = "white" if value / maximum > 0.52 else "#111827"
                axis.text(
                    col,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    color=ink,
                )
        axis.set_title(
            f"{entry['short_label']}\n{entry['count_line']}",
            fontsize=8.5,
            fontweight="bold",
            color="#111827",
            pad=8,
        )
        for edge in axis.spines.values():
            edge.set_visible(False)

    fig.suptitle(title, x=0.055, y=0.985, ha="left", fontsize=17, fontweight="bold", color="#111827")
    fig.text(0.055, 0.955, fill(subtitle, 150), ha="left", va="top", fontsize=9, color="#4b5563")
    fig.subplots_adjust(left=0.055, right=0.98, top=0.90, bottom=0.06, wspace=0.34, hspace=0.55)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_oof_entries() -> tuple[list[dict], list[dict], list[dict]]:
    patient_entries: list[dict] = []
    slice_entries: list[dict] = []
    fixed_entries: list[dict] = []

    for spec in RUN_SPECS:
        run_dir = RUNS / spec["run"]
        oof_path = run_dir / "oof_metrics.json"
        threshold_path = run_dir / "threshold.json"
        if not oof_path.exists():
            print(f"SKIP missing OOF artifact: {oof_path}")
            continue

        oof = read_json(oof_path)
        locked = oof.get("locked_threshold_metrics", {})
        slice_locked = oof.get("slice_level_locked_metrics", {})
        if not locked.get("confusion_matrix") or not slice_locked.get("confusion_matrix"):
            print(f"SKIP incomplete OOF confusion metrics: {oof_path}")
            continue

        def build_entry(metrics: dict, kind: str, threshold_label: str, suffix: str) -> dict:
            cm = confusion(metrics)
            entry = {
                "key": f"{kind}_{spec['key']}_{suffix}",
                "short_label": spec["short_label"],
                "label": spec["label"],
                "kind": kind,
                "threshold": float(metrics.get("threshold", 0.5)),
                "matrix": [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                "cm": cm,
                "count_line": count_line(cm),
                "metric_line": metric_line(metrics, cm),
                "note": spec["note"],
                "source": str(oof_path),
            }
            filename = f"{kind}__{spec['key']}__{suffix}.png"
            entry["image"] = f"graphs/confusion_matrices/{filename}"
            render_matrix(
                matrix=np.array(entry["matrix"], dtype=int),
                title=f"{spec['short_label']} - {kind.replace('_', ' ').title()}",
                subtitle=f"{threshold_label}. {entry['count_line']}. {entry['metric_line']}.",
                footer=spec["note"],
                output_path=OUT / filename,
            )
            return entry

        patient_entries.append(
            build_entry(
                locked,
                "patient_oof_locked",
                f"Official patient-level OOF at locked threshold {float(locked['threshold']):.3f}",
                "locked",
            )
        )
        slice_entries.append(
            build_entry(
                slice_locked,
                "slice_oof_locked",
                f"Official slice-level OOF at locked threshold {float(slice_locked['threshold']):.3f}",
                "locked",
            )
        )

        if threshold_path.exists():
            threshold_data = read_json(threshold_path)
            reference = threshold_data.get("reference_0_5_metrics", {})
        else:
            reference = {}

        # A few later ensemble artifacts intentionally have no threshold.json.
        # Reconstruct the common 0.5 reference directly from the preserved
        # patient-level OOF predictions without changing any evaluation data.
        if not reference.get("confusion_matrix"):
            prediction_path = run_dir / "oof_predictions_patient_level.csv"
            if prediction_path.exists():
                reference_cm = cm_from_prediction_csv(prediction_path, 0.5)
                reference = {
                    "threshold": 0.5,
                    "roc_auc": oof.get("patient_mean_roc_auc", locked.get("roc_auc")),
                    "pr_auc": locked.get("pr_auc"),
                    "confusion_matrix": reference_cm,
                    **derived_metrics(reference_cm),
                }

        if reference.get("confusion_matrix"):
            fixed_entries.append(
                build_entry(
                    reference,
                    "patient_oof_fixed05",
                    "Reference patient-level OOF at fixed threshold 0.500",
                    "fixed05",
                )
            )

    return patient_entries, slice_entries, fixed_entries


def make_transfer_entries() -> list[dict]:
    entries: list[dict] = []
    for spec in TRANSFER_SPECS:
        run_dir = RUNS / spec["run"]
        csv_path = run_dir / spec["csv"]
        transfer_path = run_dir / spec["transfer"]
        if not csv_path.exists() or not transfer_path.exists():
            print(f"SKIP missing transfer artifact: {spec['key']}")
            continue

        transfer = read_json(transfer_path)
        threshold = float(transfer["locked_threshold"])
        rows = []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "y_true": int(float(row["y_true"])),
                        "probability": float(row["stone_probability"]),
                    }
                )

        cm = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
        for row in rows:
            actual_positive = row["y_true"] == 1
            predicted_positive = row["probability"] >= threshold
            if actual_positive and predicted_positive:
                cm["tp"] += 1
            elif actual_positive:
                cm["fn"] += 1
            elif predicted_positive:
                cm["fp"] += 1
            else:
                cm["tn"] += 1

        derived = derived_metrics(cm)
        gates = transfer.get("gates", {})
        ks = float(transfer.get("ks_vs_oof", float("nan")))
        delta = float(gates.get("threshold_transfer", {}).get("delta", float("nan")))
        passed = bool(transfer.get("gates_passed", False))
        entry = {
            "key": f"transfer_diagnostic_{spec['key']}",
            "short_label": spec["short_label"],
            "label": spec["label"],
            "kind": "transfer_diagnostic",
            "threshold": threshold,
            "matrix": [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
            "cm": cm,
            "count_line": count_line(cm),
            "metric_line": (
                f"Sens {derived['sensitivity_recall']:.3f} | "
                f"Spec {derived['specificity']:.3f} | "
                f"F1 {derived['f1_score']:.3f} | "
                f"KS {ks:.3f} | Threshold delta {delta:.3f} | "
                f"Guard {'PASS' if passed else 'FAIL'}"
            ),
            "note": spec["note"],
            "source": str(csv_path),
            "transfer_source": str(transfer_path),
            "ks_vs_oof": ks,
            "threshold_delta": delta,
            "gates_passed": passed,
        }
        filename = f"transfer_diagnostic__{spec['key']}.png"
        entry["image"] = f"graphs/confusion_matrices/{filename}"
        subtitle = (
            f"Full-development transfer diagnostic at locked threshold {threshold:.3f}. "
            f"{entry['count_line']}. {entry['metric_line']}."
        )
        render_matrix(
            matrix=np.array(entry["matrix"], dtype=int),
            title=f"{spec['short_label']} - Transfer Diagnostic",
            subtitle=subtitle,
            footer=spec["note"],
            output_path=OUT / filename,
        )
        entries.append(entry)
    return entries


def make_invalidated_sealed_entries() -> list[dict]:
    metrics_path = RUNS / "20260826_231405_460200" / "final" / "test_metrics.json"
    if not metrics_path.exists():
        return []
    data = read_json(metrics_path)
    entries: list[dict] = []
    patient = data
    slice_metrics = data.get("slice_level_metrics", {})
    for kind, metrics, suffix in (
        ("sealed_test_invalidated_patient", patient, "patient"),
        ("sealed_test_invalidated_slice", slice_metrics, "slice"),
    ):
        cm = confusion(metrics)
        threshold = float(metrics.get("threshold", data.get("threshold", 0.5)))
        entry = {
            "key": kind,
            "short_label": f"Invalidated sealed test ({suffix})",
            "label": "Historical sealed-test transfer - invalidated",
            "kind": "sealed_test_invalidated",
            "threshold": threshold,
            "matrix": [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
            "cm": cm,
            "count_line": count_line(cm),
            "metric_line": metric_line(metrics, cm),
            "note": (
                "Permanently invalidated sealed-test diagnostic. It must not be used for "
                "tuning, model selection, or claims of generalization."
            ),
            "source": str(metrics_path),
        }
        filename = f"sealed_test_invalidated__{suffix}.png"
        entry["image"] = f"graphs/confusion_matrices/{filename}"
        render_matrix(
            matrix=np.array(entry["matrix"], dtype=int),
            title=f"Historical Sealed Test - Invalidated ({suffix.title()})",
            subtitle=(
                f"Historical sealed-test diagnostic at threshold {threshold:.3f}. "
                f"{entry['count_line']}. {entry['metric_line']}."
            ),
            footer=entry["note"],
            output_path=OUT / filename,
        )
        entries.append(entry)
    return entries


def main() -> None:
    patient_entries, slice_entries, fixed_entries = make_oof_entries()
    transfer_entries = make_transfer_entries()
    sealed_entries = make_invalidated_sealed_entries()

    if patient_entries:
        render_overview(
            patient_entries,
            title="Patient-level OOF confusion matrices",
            subtitle="Primary development comparison. Each matrix uses that run's locked threshold; rows are actual labels and columns are model predictions.",
            output_path=OUT / "patient_oof_locked_overview.png",
        )
    if slice_entries:
        render_overview(
            slice_entries,
            title="Slice-level OOF confusion matrices",
            subtitle="Secondary image-level view. Patient-level results are the primary evidence because slices from one patient are correlated.",
            output_path=OUT / "slice_oof_locked_overview.png",
        )
    if fixed_entries:
        render_overview(
            fixed_entries,
            title="Patient-level OOF matrices at fixed threshold 0.5",
            subtitle="Reference-only comparison at one common threshold. These are not the locked-threshold operating points used by the main report.",
            output_path=OUT / "patient_oof_fixed05_overview.png",
        )
    if transfer_entries:
        render_overview(
            transfer_entries,
            title="Development transfer diagnostic matrices",
            subtitle="Diagnostic only: these predictions are on the full development set, not independent holdout data. Do not read them as generalization performance.",
            output_path=OUT / "transfer_diagnostic_overview.png",
        )

    all_entries = patient_entries + slice_entries + fixed_entries + transfer_entries + sealed_entries
    manifest = {
        "description": "Confusion-matrix images built from preserved kidney-stone research artifacts.",
        "generated_from": "build_confusion_matrix_images.py",
        "primary_evidence": "patient_oof_locked",
        "sealed_test_policy": "historical sealed-test images are retained for audit only and permanently invalidated for tuning.",
        "entries": all_entries,
        "overviews": [
            "graphs/confusion_matrices/patient_oof_locked_overview.png",
            "graphs/confusion_matrices/slice_oof_locked_overview.png",
            "graphs/confusion_matrices/patient_oof_fixed05_overview.png",
            "graphs/confusion_matrices/transfer_diagnostic_overview.png",
        ],
    }
    manifest_path = PACK / "confusion_matrix_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Generated {len(all_entries)} individual confusion matrices.")
    print(f"Patient OOF: {len(patient_entries)}; slice OOF: {len(slice_entries)}; fixed 0.5: {len(fixed_entries)}; transfer diagnostics: {len(transfer_entries)}; invalidated sealed test: {len(sealed_entries)}.")
    print(f"Output folder: {OUT}")


if __name__ == "__main__":
    main()
