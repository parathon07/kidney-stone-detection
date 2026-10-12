"""Development-only full-development retrain diagnostic for E019.

This intentionally calls the final-training primitive directly rather than the
CLI ``finalize`` command.  It trains on original development images for the
locked median E019 epoch budget, then runs only the development transfer guard.
The sealed test is never loaded or evaluated.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.evaluate import evaluate_development_transfer
from src.splits import load_and_validate_splits
from src.trainer import train_final_model
from src.utils import load_config, load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source-cv-dir", required=True)
    parser.add_argument("--final-epochs", type=int, required=True)
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / args.run_dir
    source_cv_dir = PROJECT_ROOT / args.source_cv_dir
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {run_dir}")
    if not source_cv_dir.exists():
        raise FileNotFoundError(f"Source CV directory not found: {source_cv_dir}")
    if args.final_epochs < 1:
        raise ValueError("--final-epochs must be positive")
    run_dir.mkdir(parents=True, exist_ok=True)

    provenance_files = (
        "config_resolved.yaml",
        "config_sha256.txt",
        "data_manifest_hash.txt",
        "split_hashes.json",
        "oof_predictions_patient_level.csv",
        "oof_metrics.json",
    )
    for name in provenance_files:
        source = source_cv_dir / name
        if not source.exists():
            raise FileNotFoundError(f"Required E019 provenance artifact missing: {source}")
        shutil.copy2(source, run_dir / name)

    config = load_config(str(run_dir / "config_resolved.yaml"))
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "manifest.csv")
    merged = load_and_validate_splits(
        manifest,
        splits_dir=str(PROJECT_ROOT / "splits"),
    )
    data_dir = PROJECT_ROOT.parent / "kidney stone detection" / "data" / "dataset" / "Kindey Stone Dataset"
    if not data_dir.exists():
        raise FileNotFoundError(f"Frozen source-data directory not found: {data_dir}")

    oof_metrics = load_json(str(run_dir / "oof_metrics.json"))
    locked_threshold = float(oof_metrics["locked_threshold_metrics"]["threshold"])
    print(
        f"TRAINING E021 full development model for exactly {args.final_epochs} epochs "
        f"using E019 locked threshold {locked_threshold:.10f}",
        flush=True,
    )
    model_path = train_final_model(
        merged,
        str(data_dir),
        config,
        args.final_epochs,
        str(run_dir),
    )
    transfer = evaluate_development_transfer(
        development_manifest_df=merged,
        data_dir=str(data_dir),
        model_path=model_path,
        locked_threshold=locked_threshold,
        run_dir=str(run_dir),
        config=config,
    )
    summary = {
        "experiment_id": "E021",
        "source_cv": str(source_cv_dir),
        "final_epochs": int(args.final_epochs),
        "locked_threshold": locked_threshold,
        "model_path": str(Path(model_path)),
        "development_transfer": transfer,
        "sealed_test_used": False,
    }
    save_json(summary, str(run_dir / "e021_summary.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
