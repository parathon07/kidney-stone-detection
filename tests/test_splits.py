import os
import shutil
import tempfile
import pandas as pd
import pytest
from src.splits import generate_patient_splits, load_and_validate_splits


@pytest.fixture
def temp_splits_env():
    """Create a temporary directory for synthetic split testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def create_synthetic_manifest(n_patients=201, images_per_patient=5) -> pd.DataFrame:
    """Generate synthetic manifest mimicking Mendeley V2 CT dataset structure."""
    records = []
    for i in range(n_patients):
        patient_id = f"PATIENT_{i:03d}"
        label = 1 if i % 2 == 0 else 0  # Balanced classes
        for j in range(images_per_patient):
            source_image_id = f"img_{i:03d}_{j:02d}"
            # Original image
            records.append({
                "path": f"images/{patient_id}/{source_image_id}.jpg",
                "sha256": f"hash_orig_{i:03d}_{j:02d}",
                "patient_id": patient_id,
                "source_image_id": source_image_id,
                "label": label,
                "is_original": True,
                "augmentation_type": "none",
                "hospital": "hospital_a",
                "height": 512,
                "width": 512,
                "channels": 1,
            })
            # Add one synthetic supplied derivative
            records.append({
                "path": f"images/{patient_id}/{source_image_id}_rot10.jpg",
                "sha256": f"hash_aug_{i:03d}_{j:02d}",
                "patient_id": patient_id,
                "source_image_id": source_image_id,
                "label": label,
                "is_original": False,
                "augmentation_type": "rot10",
                "hospital": "hospital_a",
                "height": 512,
                "width": 512,
                "channels": 1,
            })
    return pd.DataFrame(records)


def test_split_generation_counts(temp_splits_env):
    """Test that split generation creates exactly 161 dev and 40 test patients with 5 folds."""
    manifest = create_synthetic_manifest(n_patients=201)
    outer_df, cv_df = generate_patient_splits(
        manifest, splits_dir=temp_splits_env, n_dev=161, n_test=40, n_folds=5, seed=42
    )

    assert len(outer_df) == 201
    assert (outer_df["split"] == "development").sum() == 161
    assert (outer_df["split"] == "test").sum() == 40
    assert len(cv_df) == 161
    assert set(cv_df["fold"].unique()) == {0, 1, 2, 3, 4}


def test_zero_leakage_assertion(temp_splits_env):
    """Test that validation succeeds on clean splits and rejects any artificial leakage."""
    manifest = create_synthetic_manifest(n_patients=201)
    generate_patient_splits(manifest, splits_dir=temp_splits_env, n_dev=161, n_test=40, n_folds=5, seed=42)

    # 1. Clean validation should pass without error
    merged = load_and_validate_splits(manifest, splits_dir=temp_splits_env, expected_dev=161, expected_test=40, expected_folds=5)
    assert len(merged) == len(manifest)

    # 2. Artificial patient leakage test (cross dev/test boundary)
    corrupt_outer = pd.read_csv(os.path.join(temp_splits_env, "outer_split.csv"))
    corrupt_outer.loc[0, "split"] = "test"  # Move a dev patient to test
    # Force duplicate patient across both
    corrupt_outer = pd.concat([corrupt_outer, pd.DataFrame([{"patient_id": corrupt_outer.loc[1, "patient_id"], "split": "test"}])], ignore_index=True)
    corrupt_outer.to_csv(os.path.join(temp_splits_env, "outer_split.csv"), index=False)

    with pytest.raises(Exception):
        load_and_validate_splits(manifest, splits_dir=temp_splits_env)


def test_no_silent_regeneration(temp_splits_env):
    """Test that existing frozen split files are strictly loaded rather than regenerated."""
    manifest = create_synthetic_manifest(n_patients=201)
    outer_1, cv_1 = generate_patient_splits(manifest, splits_dir=temp_splits_env, seed=42)

    # Modify one patient assignment intentionally in file
    outer_path = os.path.join(temp_splits_env, "outer_split.csv")
    df = pd.read_csv(outer_path)
    test_pid = df[df["split"] == "test"].iloc[0]["patient_id"]

    # Re-call generate_patient_splits with a different seed - should NOT change file
    outer_2, cv_2 = generate_patient_splits(manifest, splits_dir=temp_splits_env, seed=999)
    assert (outer_2["patient_id"] == df["patient_id"]).all()
    assert (outer_2["split"] == df["split"]).all()
