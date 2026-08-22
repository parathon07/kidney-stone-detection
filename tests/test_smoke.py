import os
import shutil
import tempfile
import numpy as np
import tensorflow as tf
import pytest

from src.utils import load_config
from src.model import build_model
from src.evaluate import compute_metrics_at_threshold, find_youden_j_threshold


@pytest.fixture
def default_config():
    return load_config("configs/default.yaml")


def test_model_architecture_and_contract(default_config):
    """
    Test model contract per Section 11.1:
      - Shape (2, 384, 384, 1) -> (2, 2)
      - Every row is finite, in [0, 1], and sums to ~1.0
      - Expected param count ~4,788,866 (~4,785,026 trainable)
    """
    model = build_model(default_config)
    
    # Forward pass on random batch of 2
    dummy_input = tf.random.normal((2, 384, 384, 1))
    preds = model(dummy_input, training=False).numpy()

    assert preds.shape == (2, 2), f"Expected shape (2, 2), got {preds.shape}"
    assert np.all(np.isfinite(preds)), "All prediction outputs must be finite"
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0), "Probabilities must be in [0, 1]"
    row_sums = np.sum(preds, axis=1)
    assert np.allclose(row_sums, [1.0, 1.0], atol=1e-5), f"Probabilities must sum to 1.0, got {row_sums}"

    # Parameter count verification
    total_params = model.count_params()
    # Number of trainable weights
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    
    assert total_params == 4788866, f"Expected 4,788,866 total params, got {total_params}"
    assert trainable_params == 4785026, f"Expected 4,785,026 trainable params, got {trainable_params}"


def test_metric_correctness():
    """
    Test metric formulas against known ground truth counts per Section 11.1.
    """
    # 2 true negatives (prob 0.1, 0.2), 1 false positive (prob 0.7), 
    # 1 false negative (prob 0.3), 2 true positives (prob 0.8, 0.9)
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.7, 0.3, 0.8, 0.9])
    
    metrics = compute_metrics_at_threshold(y_true, y_prob, threshold=0.5)
    
    cm = metrics["confusion_matrix"]
    assert cm["tn"] == 2
    assert cm["fp"] == 1
    assert cm["fn"] == 1
    assert cm["tp"] == 2

    # Sensitivity = TP / (TP + FN) = 2 / 3
    assert np.isclose(metrics["sensitivity_recall"], 2.0 / 3.0)
    # Specificity = TN / (TN + FP) = 2 / 3
    assert np.isclose(metrics["specificity"], 2.0 / 3.0)
    # Precision = TP / (TP + FP) = 2 / 3
    assert np.isclose(metrics["precision"], 2.0 / 3.0)
    # Balanced Accuracy = 2 / 3
    assert np.isclose(metrics["balanced_accuracy"], 2.0 / 3.0)


def test_youden_j_threshold_selection():
    """Test Youden's J threshold optimization."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    
    best_thresh, best_j = find_youden_j_threshold(y_true, y_prob)
    assert 0.4 <= best_thresh <= 0.6
    assert np.isclose(best_j, 1.0)


def test_checkpoint_roundtrip(default_config):
    """
    Test minimal training and checkpoint reload round-trip on synthetic tensors.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        model = build_model(default_config)
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        )

        dummy_x = np.random.rand(4, 384, 384, 1).astype(np.float32)
        dummy_y = np.array([0, 1, 0, 1], dtype=np.int32)
        ds = tf.data.Dataset.from_tensor_slices((dummy_x, dummy_y)).batch(2)

        checkpoint_path = os.path.join(temp_dir, "best.keras")
        cb = [tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=False)]
        model.fit(ds, epochs=1, callbacks=cb, verbose=0)

        assert os.path.exists(checkpoint_path)
        reloaded = tf.keras.models.load_model(checkpoint_path)
        preds = reloaded.predict(ds, verbose=0)
        assert preds.shape == (4, 2)
        assert np.allclose(np.sum(preds, axis=1), 1.0, atol=1e-5)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
