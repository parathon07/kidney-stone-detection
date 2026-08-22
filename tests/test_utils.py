import os
import shutil
import tempfile
import pandas as pd
import numpy as np
import tensorflow as tf
import pytest

from src.utils import (
    load_config,
    save_config,
    load_json,
    save_json,
    set_seed,
    compute_sha256,
    compute_df_sha256,
    get_git_info,
    get_gpu_info,
    generate_run_id,
    get_metadata,
    configure_mixed_precision,
)


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_config_roundtrip(temp_dir):
    cfg_path = os.path.join(temp_dir, "test_cfg.yaml")
    data = {"seed": 123, "model": {"name": "test_net"}}
    save_config(data, cfg_path)
    loaded = load_config(cfg_path)
    assert loaded == data


def test_config_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config_file_12345.yaml")


def test_json_roundtrip(temp_dir):
    json_path = os.path.join(temp_dir, "test.json")
    data = {"accuracy": 0.95, "threshold": 0.52}
    save_json(data, json_path)
    loaded = load_json(json_path)
    assert loaded == data


def test_json_not_found():
    with pytest.raises(FileNotFoundError):
        load_json("non_existent_json_file_12345.json")


def test_set_seed():
    set_seed(42)
    val1 = np.random.rand()
    set_seed(42)
    val2 = np.random.rand()
    assert np.isclose(val1, val2)


def test_compute_sha256(temp_dir):
    file_path = os.path.join(temp_dir, "sample.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("medical ai evaluation test")
    h1 = compute_sha256(file_path)
    h2 = compute_sha256(file_path)
    assert h1 == h2
    assert len(h1) == 64


def test_compute_df_sha256():
    df1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df2 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert compute_df_sha256(df1) == compute_df_sha256(df2)


def test_git_info():
    info = get_git_info()
    assert "commit" in info
    assert "dirty" in info


def test_gpu_info():
    info = get_gpu_info()
    assert isinstance(info, str)


def test_generate_run_id():
    run_id = generate_run_id()
    assert len(run_id) >= 15
    assert "_" in run_id


def test_get_metadata():
    meta = get_metadata(
        command="smoke",
        seed=42,
        run_id="20260822_120000",
        dataset_manifest_hash="hash123",
        outer_split_hash="hash456",
        cv_fold_hash="hash789",
        elapsed_time_sec=12.5,
    )
    assert meta["run_id"] == "20260822_120000"
    assert meta["command"] == "smoke"
    assert meta["seed"] == 42
    assert meta["dataset_manifest_hash"] == "hash123"
    assert meta["elapsed_time_seconds"] == 12.5


def test_configure_mixed_precision():
    # Should execute without throwing on CPU/GPU
    configure_mixed_precision("float32")
    configure_mixed_precision("auto")
