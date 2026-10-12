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
    apply_mild_augmentation,
    build_patient_balanced_manifest,
    create_dataset,
    resolve_shuffle_buffer,
    validate_manifest_files,
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

    # Unclassified path should raise ValueError
    with pytest.raises(ValueError):
        parse_filename_metadata("unclassified_folder/sample_scan.jpg")


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

    # Verify Patient_001 original and derivative share patient_id and source_image_id
    p1_rows = manifest_df[manifest_df["patient_id"] == "PATIENT_001"]
    assert len(p1_rows) == 2
    assert len(p1_rows["source_image_id"].unique()) == 1
    audit = validate_manifest_files(manifest_df, temp_data_env["root"])
    assert audit["file_count"] == 3


def test_manifest_validation_rejects_invalid_labels_and_stale_dimensions(temp_data_env):
    manifest_df = scan_dataset_and_create_manifest(
        temp_data_env["root"],
        output_manifest_path=os.path.join(temp_data_env["root"], "manifest.csv"),
    )

    invalid_label = manifest_df.copy()
    invalid_label["label"] = invalid_label["label"].astype(float)
    invalid_label.loc[0, "label"] = 0.5
    with pytest.raises(ValueError, match="labels"):
        validate_manifest_files(invalid_label, temp_data_env["root"])

    stale_dimensions = manifest_df.copy()
    stale_dimensions.loc[0, "height"] += 1
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        validate_manifest_files(stale_dimensions, temp_data_env["root"])


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


def test_mild_augmentation_is_reproducible_with_stateless_seed():
    config = {
        "data": {
            "augmentation": {
                "rotation_deg": 10,
                "translation_fraction": 0.03,
                "zoom_fraction": 0.05,
                "contrast_fraction": 0.10,
                "brightness_fraction": 0.05,
            }
        }
    }
    image = tf.ones((32, 32, 1), dtype=tf.float32) * 0.5
    seed = tf.constant([42, 7], dtype=tf.int32)
    first = apply_mild_augmentation(image, config, image_size=32, seed=seed)
    second = apply_mild_augmentation(image, config, image_size=32, seed=seed)
    assert np.array_equal(first.numpy(), second.numpy())


def test_shuffle_buffer_defaults_to_the_full_dataset():
    assert resolve_shuffle_buffer(2796) == 2796
    assert resolve_shuffle_buffer(2796, requested=1024) == 1024
    assert resolve_shuffle_buffer(2796, requested=5000) == 2796

    with pytest.raises(ValueError, match="positive"):
        resolve_shuffle_buffer(10, requested=0)


def test_patient_balanced_manifest_equalizes_patient_frequency():
    manifest_df = pd.DataFrame({
        "patient_id": ["P1"] * 5 + ["P2"] * 2 + ["P3"],
        "path": [f"image_{idx}.jpg" for idx in range(8)],
        "label": [0] * 5 + [1] * 3,
    })
    balanced = build_patient_balanced_manifest(manifest_df, seed=17)
    counts = balanced.groupby("patient_id").size()

    assert len(balanced) == len(manifest_df)
    assert counts.max() - counts.min() <= 1
    assert set(counts.index) == {"P1", "P2", "P3"}
    pd.testing.assert_frame_equal(
        balanced,
        build_patient_balanced_manifest(manifest_df, seed=17),
    )


def test_create_dataset(temp_data_env):
    manifest_path = os.path.join(temp_data_env["root"], "manifest.csv")
    manifest_df = scan_dataset_and_create_manifest(
        temp_data_env["root"], output_manifest_path=manifest_path
    )
    config = {
        "data": {
            "image_size": 128,
            "use_supplied_derivatives": False,
            "cache_in_ram": False,
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


def test_cached_dataset_matches_canonical_preprocessing(temp_data_env):
    manifest_path = os.path.join(temp_data_env["root"], "manifest.csv")
    manifest_df = scan_dataset_and_create_manifest(
        temp_data_env["root"], output_manifest_path=manifest_path
    )
    config = {
        "data": {
            "image_size": 128,
            "use_supplied_derivatives": False,
            "cache_in_ram": True,
        },
        "training": {"batch_size": 8},
        "seed": 42,
    }
    val_df = manifest_df[manifest_df["is_original"] == True].reset_index(drop=True)
    cached_x, _ = next(iter(create_dataset(
        val_df, temp_data_env["root"], config, split_type="val"
    )))
    canonical_x, _ = canonical_preprocess_image(
        tf.constant(os.path.join(temp_data_env["root"], val_df.iloc[0]["path"])),
        label=tf.constant(int(val_df.iloc[0]["label"])),
        image_size=128,
    )
    assert np.allclose(cached_x[0].numpy(), canonical_x.numpy(), atol=1.0 / 255.0)
