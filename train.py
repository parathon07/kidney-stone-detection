import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf

from src.utils import (
    load_config,
    save_config,
    save_json,
    load_json,
    set_seed,
    compute_sha256,
    get_metadata,
    configure_mixed_precision,
    generate_run_id,
)
from src.data import scan_dataset_and_create_manifest, create_dataset, canonical_preprocess_image
from src.splits import (
    generate_patient_splits,
    load_and_validate_splits,
    get_split_hashes_dict,
)
from src.model import build_model
from src.trainer import train_single_fold, train_final_model
from src.evaluate import (
    evaluate_oof_and_select_threshold,
    evaluate_sealed_test_set,
    export_inference_bundle,
    compute_metrics_at_threshold,
)


def run_prepare(args: argparse.Namespace) -> None:
    """Execute dataset scanning, manifest generation, and frozen split creation."""
    print("=== [1/4] Running Pipeline Stage: PREPARE ===")
    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Provided dataset directory not found: {data_dir}")

    manifest_path = "data/manifest.csv"
    print(f"Scanning dataset at '{data_dir}'...")
    manifest_df = scan_dataset_and_create_manifest(data_dir, output_manifest_path=manifest_path)
    print(f"Manifest written to '{manifest_path}' with {len(manifest_df)} total indexed images.")

    # Generate or load frozen patient-wise splits
    print("Generating/verifying deterministic patient-wise outer split and 5 CV folds...")
    generate_patient_splits(manifest_df, splits_dir="splits", n_dev=161, n_test=40, n_folds=5, seed=config.get("seed", 42))
    
    # Audit integrity & assert zero leakage
    merged_df = load_and_validate_splits(manifest_df, splits_dir="splits")
    dev_count = (merged_df["split"] == "development").sum()
    test_count = (merged_df["split"] == "test").sum()
    orig_dev = ((merged_df["split"] == "development") & (merged_df["is_original"] == True)).sum()
    orig_test = ((merged_df["split"] == "test") & (merged_df["is_original"] == True)).sum()

    print(f"Integrity check PASSED successfully.")
    print(f"Development Set: {dev_count} images ({orig_dev} original) across 161 patients.")
    print(f"Final Test Set:  {test_count} images ({orig_test} original) across 40 patients.")
    print("Splits successfully frozen in 'splits/outer_split.csv' and 'splits/cv_folds.csv'.")


def run_smoke(args: argparse.Namespace) -> None:
    """Execute smoke test to quickly verify tensor shapes, model forward pass, checkpointing, and metrics."""
    print("=== [2/4] Running Pipeline Stage: SMOKE TEST ===")
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    configure_mixed_precision(config.get("training", {}).get("mixed_precision", "auto"))

    image_size = config.get("data", {}).get("image_size", 384)
    channels = config.get("data", {}).get("channels", 1)

    print(f"1. Testing Model Construction for {config['model']['name']}...")
    model = build_model(config)
    print(f"   Model built successfully. Total params: {model.count_params():,}")

    # Forward pass with random tensor
    print(f"2. Testing Model Forward Pass on dummy input (shape: (2, {image_size}, {image_size}, {channels}))...")
    dummy_input = tf.random.normal((2, image_size, image_size, channels))
    dummy_output = model(dummy_input, training=False)
    assert dummy_output.shape == (2, 2), f"Expected output shape (2, 2), got {dummy_output.shape}"
    row_sums = tf.reduce_sum(dummy_output, axis=-1).numpy()
    assert np.allclose(row_sums, [1.0, 1.0], atol=1e-5), f"Softmax output row sums must equal 1.0, got {row_sums}"
    print(f"   Forward pass output shape: {dummy_output.shape}, Softmax row sums: {row_sums} [PASS]")

    # Minimal training roundtrip
    print("3. Testing Minimal Training & Checkpoint Roundtrip...")
    smoke_run_dir = os.path.join("runs", "smoke_test")
    os.makedirs(smoke_run_dir, exist_ok=True)
    checkpoint_path = os.path.join(smoke_run_dir, "smoke_best.keras")

    dummy_x = np.random.rand(8, image_size, image_size, channels).astype(np.float32)
    dummy_y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
    smoke_ds = tf.data.Dataset.from_tensor_slices((dummy_x, dummy_y)).batch(4)

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    callbacks = [tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="loss", save_best_only=True)]
    model.fit(smoke_ds, epochs=1, callbacks=callbacks, verbose=0)

    assert os.path.exists(checkpoint_path), "Smoke checkpoint was not created!"
    reloaded_model = tf.keras.models.load_model(checkpoint_path)
    reloaded_preds = reloaded_model.predict(smoke_ds, verbose=0)
    assert reloaded_preds.shape == (8, 2), f"Expected shape (8, 2), got {reloaded_preds.shape}"
    print("   Checkpoint save and reload verified [PASS]")

    # Metric calculation test
    print("4. Testing Metric Formulas on Toy Labels & Probabilities...")
    toy_true = np.array([0, 0, 1, 1])
    toy_prob = np.array([0.1, 0.4, 0.8, 0.9])
    metrics = compute_metrics_at_threshold(toy_true, toy_prob, threshold=0.5)
    assert metrics["sensitivity_recall"] == 1.0
    assert metrics["specificity"] == 1.0
    print("   Metric computation verified [PASS]")

    print("\n=== SMOKE TEST PASSED: All contracts, shapes, and pipeline mechanics verified ===")


def run_cv(args: argparse.Namespace) -> None:
    """Execute 5-fold cross-validation or a single designated fold."""
    print("=== [3/4] Running Pipeline Stage: CROSS-VALIDATION ===")
    start_time = time.time()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    configure_mixed_precision(config.get("training", {}).get("mixed_precision", "auto"))

    if args.batch_size:
        config["training"]["batch_size"] = int(args.batch_size)

    data_dir = args.data_dir
    manifest_path = "data/manifest.csv"
    if not os.path.exists(manifest_path):
        print(f"Manifest not found. Running prepare step first on '{data_dir}'...")
        scan_dataset_and_create_manifest(data_dir, output_manifest_path=manifest_path)
        generate_patient_splits(pd.read_csv(manifest_path), splits_dir="splits")

    manifest_df = pd.read_csv(manifest_path)
    merged_manifest = load_and_validate_splits(manifest_df, splits_dir="splits")
    dev_manifest = merged_manifest[merged_manifest["split"] == "development"].copy()

    run_id = generate_run_id()
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Compute provenance hashes
    manifest_hash = compute_sha256(manifest_path)
    split_hashes = get_split_hashes_dict("splits")

    save_config(config, os.path.join(run_dir, "config_resolved.yaml"))
    with open(os.path.join(run_dir, "data_manifest_hash.txt"), "w", encoding="utf-8") as f:
        f.write(manifest_hash)
    save_json(split_hashes, os.path.join(run_dir, "split_hashes.json"))

    folds_to_run = [int(args.fold)] if args.fold is not None else list(range(5))

    for fold_idx in folds_to_run:
        train_single_fold(fold_idx, dev_manifest, data_dir, config, run_dir)

    # If all 5 folds completed, run OOF aggregation
    if len(folds_to_run) == 5:
        print("\n--- Aggregating Out-Of-Fold (OOF) Predictions & Optimizing Threshold ---")
        oof_results = evaluate_oof_and_select_threshold(run_dir, config, n_folds=5)
        print(f"OOF ROC-AUC: {oof_results['locked_threshold_metrics']['roc_auc']:.4f}")
        print(f"OOF Sensitivity: {oof_results['locked_threshold_metrics']['sensitivity_recall']:.4f}")
        print(f"OOF Specificity: {oof_results['locked_threshold_metrics']['specificity']:.4f}")
        print(f"Optimal Youden's J Threshold: {oof_results['locked_threshold_metrics']['threshold']:.4f}")
        print(f"Median CV Best Epoch: {oof_results['median_best_epoch']}")

    elapsed_time = time.time() - start_time
    metadata = get_metadata(
        command="cv",
        seed=config.get("seed", 42),
        run_id=run_id,
        dataset_manifest_hash=manifest_hash,
        outer_split_hash=split_hashes["outer_split_sha256"],
        cv_fold_hash=split_hashes["cv_folds_sha256"],
        elapsed_time_sec=elapsed_time,
    )
    save_json(metadata, os.path.join(run_dir, "metadata.json"))
    print(f"\nCV Run completed successfully. Run artifacts stored at: {run_dir}")


def run_finalize(args: argparse.Namespace) -> None:
    """Retrain on all 161 development patients, evaluate once on sealed test set, and export bundle."""
    print("=== [4/4] Running Pipeline Stage: FINALIZE ===")
    start_time = time.time()
    run_dir = args.run_dir
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Specified run directory not found: {run_dir}")

    config_path = os.path.join(run_dir, "config_resolved.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Resolved config not found in run directory: {config_path}")
    config = load_config(config_path)
    set_seed(config.get("seed", 42))
    configure_mixed_precision(config.get("training", {}).get("mixed_precision", "auto"))

    if args.batch_size:
        config["training"]["batch_size"] = int(args.batch_size)

    data_dir = args.data_dir
    manifest_path = "data/manifest.csv"
    manifest_df = pd.read_csv(manifest_path)
    merged_manifest = load_and_validate_splits(manifest_df, splits_dir="splits")

    # Load OOF results and locked threshold
    oof_metrics_path = os.path.join(run_dir, "oof_metrics.json")
    threshold_path = os.path.join(run_dir, "threshold.json")
    if not os.path.exists(oof_metrics_path) or not os.path.exists(threshold_path):
        raise FileNotFoundError("OOF metrics or threshold.json missing. Ensure 5-fold CV was completed.")

    threshold_data = load_json(threshold_path)
    locked_threshold = float(threshold_data["threshold"])

    oof_data = load_json(oof_metrics_path)
    final_epochs = max(1, int(oof_data["median_best_epoch"]))

    print(f"Resolved locked threshold: {locked_threshold:.4f}")
    print(f"Resolved final training epoch budget: {final_epochs}")

    # Write locked_config.yaml
    locked_config = dict(config)
    locked_config["final_locked_settings"] = {
        "locked_threshold": locked_threshold,
        "final_epochs": final_epochs,
        "evaluation_strategy": "youden_j_oof",
    }
    save_config(locked_config, os.path.join(run_dir, "locked_config.yaml"))

    # 1. Retrain from scratch on all 161 development patients
    final_model_path = train_final_model(merged_manifest, data_dir, config, final_epochs, run_dir)

    # 2. Evaluate once on the sealed 40-patient test set
    print("\n--- Evaluating on Sealed 40-Patient Final Test Set (Original Images Only) ---")
    test_metrics = evaluate_sealed_test_set(
        test_manifest_df=merged_manifest,
        data_dir=data_dir,
        model_path=final_model_path,
        locked_threshold=locked_threshold,
        run_dir=run_dir,
        config=config,
    )
    print(f"Final Test ROC-AUC:      {test_metrics['roc_auc']:.4f}")
    print(f"Final Test PR-AUC:       {test_metrics['pr_auc']:.4f}")
    print(f"Final Test Sensitivity:  {test_metrics['sensitivity_recall']:.4f}")
    print(f"Final Test Specificity:  {test_metrics['specificity']:.4f}")
    print(f"Final Test Precision:    {test_metrics['precision']:.4f}")
    print(f"Final Test F1-Score:     {test_metrics['f1_score']:.4f}")
    print(f"Final Test Balanced Acc: {test_metrics['balanced_accuracy']:.4f}")

    # 3. Export Inference Bundle
    metadata_path = os.path.join(run_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        metadata = load_json(metadata_path)

    bundle_dir = export_inference_bundle(
        run_dir=run_dir,
        final_model_path=final_model_path,
        locked_threshold=locked_threshold,
        config=config,
        metadata=metadata,
    )
    print(f"\nFinalization complete. Model bundle ready for deployment at: {bundle_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Kidney Stone Detection CNN: V1 Training, Cross-Validation, and Finalization Pipeline"
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True, help="Pipeline subcommand")

    # 1. prepare
    prep_parser = subparsers.add_parser("prepare", help="Scan dataset, create manifest, and freeze patient splits")
    prep_parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    prep_parser.add_argument("--data-dir", type=str, required=True, help="Path to raw dataset directory")

    # 2. smoke
    smoke_parser = subparsers.add_parser("smoke", help="Verify model shapes, checkpointing, and metrics")
    smoke_parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    smoke_parser.add_argument("--data-dir", type=str, default=None, help="Optional path to dataset directory")

    # 3. cv
    cv_parser = subparsers.add_parser("cv", help="Run 5-fold cross-validation and OOF evaluation")
    cv_parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    cv_parser.add_argument("--data-dir", type=str, required=True, help="Path to raw dataset directory")
    cv_parser.add_argument("--fold", type=int, default=None, help="Specific fold index (0-4) to train")
    cv_parser.add_argument("--batch-size", type=int, default=None, help="Hardware batch size override")

    # 4. finalize
    fin_parser = subparsers.add_parser("finalize", help="Retrain on all dev patients, evaluate sealed test, export bundle")
    fin_parser.add_argument("--run-dir", type=str, required=True, help="Path to completed CV run directory (e.g. runs/<run_id>)")
    fin_parser.add_argument("--data-dir", type=str, required=True, help="Path to raw dataset directory")
    fin_parser.add_argument("--batch-size", type=int, default=None, help="Hardware batch size override")

    args = parser.parse_args()

    if args.subcommand == "prepare":
        run_prepare(args)
    elif args.subcommand == "smoke":
        run_smoke(args)
    elif args.subcommand == "cv":
        run_cv(args)
    elif args.subcommand == "finalize":
        run_finalize(args)


if __name__ == "__main__":
    main()
