import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import pytest

from src.data import (
    parse_filename_metadata,
    compute_file_sha256,
    scan_dataset_and_create_manifest,
    canonical_preprocess_image,
    build_augmentation_layer,
    apply_mild_contrast,
    create_dataset,
)


@pytest.fixture
def temp_data_env():
    temp_dir = tempfile.mkdtemp()
    
    # Create fake dataset directory structure
    stone_dir = os.path.join(temp_dir, "Stone", "Patient_001")
    nonstone_dir = os.path.join(temp_dir, "Non_Stone", "Patient_002")
    os.makedirs(stone_dir, exist_ok=True)
    os.makedirs(nonstone_dir, exist_ok=True)

    # Save small valid synthetic images
    img1_path = os.path.join(stone_dir, "Patient_001_slice_01.jpg")
    img2_path = os.path.join(stone_dir, "Patient_001_slice_01_rot10.jpg")
    img3_path = os.path.join(nonstone_dir, "Patient_002_slice_01.jpg")

    arr = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    Image.fromarray(arr).save(img1_path)
    Image.fromarray(arr).save(img2_path)
    Image.fromarray(arr).save(img3_path)

    yield {
        "root": temp_dir,
        "img1": img1_path,
        "img2": img2_path,
        "img3": img3_path,
    }
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_parse_filename_metadata():
    # Stone original
    meta1 = parse_filename_metadata("dataset/Stone/Patient_005_slice_02.jpg")
    assert meta1["label"] == 1
    assert meta1["is_original"] is True
    assert meta1["patient_id"] == "PATIENT_005"
    assert "005" in meta1["source_image_id"]

    # Non-stone augmented
    meta2 = parse_filename_metadata("dataset/Non_Stone/Patient_010_slice_03_rot10.jpg")
    assert meta2["label"] == 0
    assert meta2["is_original"] is False
    assert meta2["patient_id"] == "PATIENT_010"

    # Case pattern fallback
    meta3 = parse_filename_metadata("Normal (14).jpg")
    assert meta3["label"] == 0
    assert meta3["patient_id"] == "PAT_NONSTONE_14"


def test_scan_dataset_and_create_manifest(temp_data_env):
    manifest_path = os.path.join(temp_data_env["root"], "manifest.csv")
    manifest_df = scan_dataset_and_create_manifest(
        temp_data_env["root"], output_manifest_path=manifest_path
    )

    assert os.path.exists(manifest_path)
    assert len(manifest_df) == 3
    assert set(manifest_df["label"].unique()) == {0, 1}
    assert (manifest_df["is_original"] == True).sum() == 2
    assert (manifest_df["is_original"] == False).sum() == 1


def test_canonical_preprocess_image(temp_data_env):
    img_path = temp_data_env["img1"]
    processed, label = canonical_preprocess_image(
        tf.constant(img_path), label=tf.constant(1), image_size=384
    )

    assert processed.shape == (384, 384, 1)
    assert processed.dtype == tf.float32
    assert label.numpy() == 1

    # Pixel range check [0, 1]
    p_min = tf.reduce_min(processed).numpy()
    p_max = tf.reduce_max(processed).numpy()
    assert 0.0 <= p_min <= 1.0
    assert 0.0 <= p_max <= 1.0


def test_build_augmentation_layer():
    config = {
        "data": {
            "augmentation": {
                "rotation_deg": 10,
                "translation_fraction": 0.03,
                "zoom_fraction": 0.05,
            }
        }
    }
    aug_layer = build_augmentation_layer(config)
    dummy = tf.random.normal((2, 384, 384, 1))
    out = aug_layer(dummy, training=True)
    assert out.shape == (2, 384, 384, 1)


def test_apply_mild_contrast():
    dummy = tf.random.uniform((384, 384, 1), 0.0, 1.0)
    adjusted = apply_mild_contrast(dummy, contrast_fraction=0.10)
    assert adjusted.shape == (384, 384, 1)
    assert tf.reduce_min(adjusted).numpy() >= 0.0
    assert tf.reduce_max(adjusted).numpy() <= 1.0


def test_create_dataset(temp_data_env):
    manifest_path = os.path.join(temp_data_env["root"], "manifest.csv")
    manifest_df = scan_dataset_and_create_manifest(
        temp_data_env["root"], output_manifest_path=manifest_path
    )
    config = {
        "data": {
            "image_size": 128,
            "use_supplied_derivatives": False,
            "augmentation": {
                "rotation_deg": 10,
                "translation_fraction": 0.03,
                "zoom_fraction": 0.05,
                "contrast_fraction": 0.10,
            },
        },
        "training": {"batch_size": 2},
        "seed": 42,
    }

    train_ds = create_dataset(manifest_df, temp_data_env["root"], config, split_type="train")
    for batch_x, batch_y in train_ds.take(1):
        assert batch_x.shape[1:] == (128, 128, 1)
        assert len(batch_y) <= 2
