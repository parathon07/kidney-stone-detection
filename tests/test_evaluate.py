import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import tensorflow as tf
import pytest

from src.evaluate import (
    compute_metrics_at_threshold,
    find_youden_j_threshold,
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix_heatmap,
    plot_cv_learning_curves,
    evaluate_oof_and_select_threshold,
    export_inference_bundle,
)
from src.utils import save_json


@pytest.fixture
def temp_eval_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_compute_metrics_all_zeros():
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])
    metrics = compute_metrics_at_threshold(y_true, y_prob, threshold=0.5)
    assert metrics["specificity"] == 1.0
    assert metrics["confusion_matrix"]["tn"] == 4
    assert metrics["confusion_matrix"]["tp"] == 0


def test_compute_metrics_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics_at_threshold(y_true, y_prob, threshold=0.5)
    assert metrics["roc_auc"] == 1.0
    assert metrics["sensitivity_recall"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0


def test_plotting_functions(temp_eval_dir):
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    roc_path = os.path.join(temp_eval_dir, "roc.png")
    pr_path = os.path.join(temp_eval_dir, "pr.png")
    cm_path = os.path.join(temp_eval_dir, "cm.png")
    curves_path = os.path.join(temp_eval_dir, "learning_curves.png")

    plot_roc_curve(y_true, y_prob, roc_path)
    assert os.path.exists(roc_path)

    plot_pr_curve(y_true, y_prob, pr_path)
    assert os.path.exists(pr_path)

    cm_dict = {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    plot_confusion_matrix_heatmap(cm_dict, ["Non-Stone", "Stone"], cm_path)
    assert os.path.exists(cm_path)

    # Empty fold directories should safely generate learning curves without throwing
    plot_cv_learning_curves(temp_eval_dir, n_folds=2, save_path=curves_path)
    assert os.path.exists(curves_path)


def test_evaluate_oof_and_select_threshold(temp_eval_dir):
    # Setup mock fold directories
    for fold in range(5):
        f_dir = os.path.join(temp_eval_dir, f"fold_{fold}")
        os.makedirs(f_dir, exist_ok=True)
        val_df = pd.DataFrame({
            "path": [f"img_{fold}_0.jpg", f"img_{fold}_1.jpg"],
            "patient_id": [f"P{fold}_A", f"P{fold}_B"],
            "source_image_id": [f"src_{fold}_0", f"src_{fold}_1"],
            "y_true": [0, 1],
            "stone_probability": [0.1 * (fold + 1), 0.8],
            "predicted_class_at_0_5": [0, 1],
            "fold": [fold, fold],
        })
        val_df.to_csv(os.path.join(f_dir, "val_predictions.csv"), index=False)
        save_json({"fold": fold, "best_epoch": fold + 1, "best_val_loss": 0.3}, os.path.join(f_dir, "fold_metrics.json"))

    config = {"evaluation": {"class_names": ["Non-Stone", "Stone"]}}
    results = evaluate_oof_and_select_threshold(temp_eval_dir, config, n_folds=5)

    assert results["total_oof_samples"] == 10
    assert "locked_threshold_metrics" in results
    assert os.path.exists(os.path.join(temp_eval_dir, "oof_predictions.csv"))
    assert os.path.exists(os.path.join(temp_eval_dir, "threshold.json"))
    assert os.path.exists(os.path.join(temp_eval_dir, "oof_metrics.json"))


def test_evaluate_oof_rejects_missing_or_corrupt_metrics(temp_eval_dir):
    corrupt_dir = os.path.join(temp_eval_dir, "corrupt_run")
    f0_dir = os.path.join(corrupt_dir, "fold_0")
    os.makedirs(f0_dir, exist_ok=True)
    
    val_df = pd.DataFrame({
        "path": ["img_0.jpg"],
        "patient_id": ["P0"],
        "source_image_id": ["src_0"],
        "y_true": [1],
        "stone_probability": [0.9],
        "predicted_class_at_0_5": [1],
        "fold": [0],
    })
    val_df.to_csv(os.path.join(f0_dir, "val_predictions.csv"), index=False)
    config = {"evaluation": {"class_names": ["Non-Stone", "Stone"]}}

    # Case 1: Missing fold_metrics.json file entirely
    with pytest.raises(FileNotFoundError):
        evaluate_oof_and_select_threshold(corrupt_dir, config, n_folds=1)

    # Case 2: fold_metrics.json missing best_epoch
    metrics_path = os.path.join(f0_dir, "fold_metrics.json")
    save_json({"fold": 0, "best_val_loss": 0.25}, metrics_path)
    with pytest.raises(ValueError, match="positive integer 'best_epoch'"):
        evaluate_oof_and_select_threshold(corrupt_dir, config, n_folds=1)

    # Case 3: best_epoch = 0
    save_json({"fold": 0, "best_epoch": 0, "best_val_loss": 0.25}, metrics_path)
    with pytest.raises(ValueError, match="positive integer 'best_epoch'"):
        evaluate_oof_and_select_threshold(corrupt_dir, config, n_folds=1)

    # Case 4: best_epoch is boolean True
    save_json({"fold": 0, "best_epoch": True, "best_val_loss": 0.25}, metrics_path)
    with pytest.raises(ValueError, match="positive integer 'best_epoch'"):
        evaluate_oof_and_select_threshold(corrupt_dir, config, n_folds=1)

    # Case 5: best_epoch is float
    save_json({"fold": 0, "best_epoch": 3.14, "best_val_loss": 0.25}, metrics_path)
    with pytest.raises(ValueError, match="positive integer 'best_epoch'"):
        evaluate_oof_and_select_threshold(corrupt_dir, config, n_folds=1)


def test_export_inference_bundle(temp_eval_dir):
    # Create genuine minimal Keras model
    inputs = tf.keras.Input(shape=(384, 384, 1), name="input_image")
    outputs = tf.keras.layers.Dense(2, activation="softmax")(tf.keras.layers.GlobalAveragePooling2D()(inputs))
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="test_export_model")
    
    saved_model_path = os.path.join(temp_eval_dir, "model.keras")
    model.save(saved_model_path)

    config = {
        "data": {"image_size": 384, "channels": 1},
        "evaluation": {"class_names": ["Non-Stone", "Stone"]},
    }
    meta = {
        "run_id": "test_run_123",
        "timestamp": "2026-08-22 12:00:00 UTC",
        "git_commit": "abc1234",
        "dataset_manifest_hash": "hash1",
        "outer_split_hash": "hash2",
        "cv_fold_hash": "hash3",
    }

    bundle_dir = export_inference_bundle(
        run_dir=temp_eval_dir,
        final_model_path=saved_model_path,
        locked_threshold=0.5234,
        config=config,
        metadata=meta,
    )

    bundle_model_file = os.path.join(bundle_dir, "model.keras")
    assert os.path.exists(bundle_model_file)
    assert os.path.exists(os.path.join(bundle_dir, "inference_config.json"))
    assert os.path.exists(os.path.join(bundle_dir, "provenance.json"))

    # Reload bundle model and verify contracts
    reloaded_model = tf.keras.models.load_model(bundle_model_file)
    assert reloaded_model.input_shape == (None, 384, 384, 1)
    assert reloaded_model.output_shape == (None, 2)
