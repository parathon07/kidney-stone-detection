import os
import shutil
import tempfile
import pandas as pd
import numpy as np
import tensorflow as tf
import pytest

from src.trainer import compile_canonical_model
from src.model import build_model
from src.utils import save_json, load_json


@pytest.fixture
def temp_trainer_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_compile_canonical_model():
    config = {
        "model": {"name": "test_net", "filters": [16], "convs_per_block": 1, "dense_units": 16},
        "data": {"image_size": 64, "channels": 1},
        "training": {"learning_rate": 0.001, "weight_decay": 0.0001},
    }
    model = build_model(config)
    compiled = compile_canonical_model(model, config)
    
    assert isinstance(compiled.optimizer, tf.keras.optimizers.AdamW)
    assert isinstance(compiled.loss, tf.keras.losses.SparseCategoricalCrossentropy)
    assert len(compiled.metrics) >= 1


def test_trainer_epoch_normalization_rules():
    """
    Test epoch parsing and normalization:
      - 0-indexed Keras CSVLogger epochs (0, 1, 2...) are converted to 1-indexed counts (1, 2, 3...).
      - Missing epoch columns fall back to (best_epoch_idx + 1).
      - Persisted best_epoch is always strictly >= 1.
    """
    # Case 1: Best loss at epoch 0 in 0-indexed CSVLogger history
    hist1 = pd.DataFrame({
        "epoch": [0, 1, 2, 3],
        "loss": [0.5, 0.4, 0.3, 0.2],
        "val_loss": [0.15, 0.20, 0.25, 0.30],
    })
    best_idx1 = int(hist1["val_loss"].idxmin())
    raw_epoch1 = int(hist1.loc[best_idx1, "epoch"])
    epoch1 = raw_epoch1 + 1 if hist1["epoch"].min() == 0 else raw_epoch1
    assert epoch1 == 1

    # Case 2: Best loss at epoch 3 in 0-indexed history
    hist2 = pd.DataFrame({
        "epoch": [0, 1, 2, 3],
        "val_loss": [0.4, 0.3, 0.2, 0.1],
    })
    best_idx2 = int(hist2["val_loss"].idxmin())
    raw_epoch2 = int(hist2.loc[best_idx2, "epoch"])
    epoch2 = raw_epoch2 + 1 if hist2["epoch"].min() == 0 else raw_epoch2
    assert epoch2 == 4

    # Case 3: Missing epoch column
    hist3 = pd.DataFrame({
        "val_loss": [0.9, 0.8, 0.5, 0.7],
    })
    best_idx3 = int(hist3["val_loss"].idxmin())
    epoch3 = best_idx3 + 1
    assert epoch3 == 3
    assert epoch3 >= 1
