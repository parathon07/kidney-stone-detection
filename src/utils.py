import os
import sys
import json
import yaml
import time
import random
import hashlib
import subprocess
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save dictionary to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def save_json(data: Any, save_path: str, indent: int = 2) -> None:
    """Save serializable data to JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def load_json(path: str) -> Any:
    """Load JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compute_sha256(file_path: str, block_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            sha256.update(block)
    return sha256.hexdigest()


def compute_df_sha256(df) -> str:
    """Compute deterministic SHA-256 hash of a pandas DataFrame content."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def get_git_info() -> Dict[str, Any]:
    """Retrieve current Git commit hash and dirty status if available."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        is_dirty = len(status) > 0
        return {"commit": commit, "dirty": is_dirty}
    except Exception:
        return {"commit": "unknown", "dirty": False}


def get_gpu_info() -> str:
    """Get GPU device name or CPU description."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        details = []
        for gpu in gpus:
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                details.append(gpu_details.get("device_name", gpu.name))
            except Exception:
                details.append(gpu.name)
        return ", ".join(details)
    return "CPU"


def generate_run_id() -> str:
    """Generate timestamp-based unique run identifier."""
    return time.strftime("%Y%m%d_%H%M%S")


def get_metadata(
    command: str,
    seed: int,
    run_id: str,
    dataset_manifest_hash: Optional[str] = None,
    outer_split_hash: Optional[str] = None,
    cv_fold_hash: Optional[str] = None,
    elapsed_time_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate run metadata dictionary matching spec section 10."""
    git_info = get_git_info()
    return {
        "run_id": run_id,
        "command": command,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "python_version": sys.version.split()[0],
        "tensorflow_version": tf.__version__,
        "gpu_name": get_gpu_info(),
        "seed": seed,
        "dataset_manifest_hash": dataset_manifest_hash or "none",
        "outer_split_hash": outer_split_hash or "none",
        "cv_fold_hash": cv_fold_hash or "none",
        "elapsed_time_seconds": elapsed_time_sec,
    }


def configure_mixed_precision(policy: str = "auto") -> None:
    """Configure TensorFlow mixed precision if requested and available."""
    if policy == "auto":
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
            except Exception:
                pass
    elif policy == "mixed_float16":
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    elif policy in ["float32", "none", "off"]:
        tf.keras.mixed_precision.set_global_policy("float32")
