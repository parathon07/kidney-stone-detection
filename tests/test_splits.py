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


def test_seed_controls_new_split_generation(temp_splits_env):
    manifest = create_synthetic_manifest(n_patients=201)
    first_dir = os.path.join(temp_splits_env, "seed_42")
    second_dir = os.path.join(temp_splits_env, "seed_999")

    outer_1, _ = generate_patient_splits(manifest, splits_dir=first_dir, seed=42)
    outer_2, _ = generate_patient_splits(manifest, splits_dir=second_dir, seed=999)

    assignments_1 = outer_1.set_index("patient_id")["split"]
    assignments_2 = outer_2.set_index("patient_id")["split"]
    assert not assignments_1.equals(assignments_2)


def test_patient_label_conflicts_are_rejected(temp_splits_env):
    manifest = create_synthetic_manifest(n_patients=201)
    patient_rows = manifest[manifest["patient_id"] == "PATIENT_000"].index
    manifest.loc[patient_rows[0], "label"] = 0

    with pytest.raises(ValueError, match="within patient records"):
        generate_patient_splits(manifest, splits_dir=temp_splits_env)


def test_connected_component_label_conflicts_include_patient_mapping(temp_splits_env):
    manifest = create_synthetic_manifest(n_patients=201)
    first = manifest[manifest["patient_id"] == "PATIENT_000"].index[0]
    second = manifest[manifest["patient_id"] == "PATIENT_001"].index[0]
    shared_hash = "shared-conflicting-hash"
    manifest.loc[first, "sha256"] = shared_hash
    manifest.loc[second, "sha256"] = shared_hash

    with pytest.raises(ValueError, match="PATIENT_000.*PATIENT_001"):
        generate_patient_splits(manifest, splits_dir=temp_splits_env)


def test_partial_frozen_split_is_not_regenerated(temp_splits_env):
    manifest = create_synthetic_manifest(n_patients=201)
    pd.DataFrame({
        "patient_id": sorted(manifest["patient_id"].unique()),
        "split": ["development"] * 161 + ["test"] * 40,
    }).to_csv(os.path.join(temp_splits_env, "outer_split.csv"), index=False)

    with pytest.raises(FileExistsError, match="incomplete"):
        generate_patient_splits(manifest, splits_dir=temp_splits_env)


def test_missing_development_fold_is_rejected(temp_splits_env):
    manifest = create_synthetic_manifest(n_patients=201)
    generate_patient_splits(manifest, splits_dir=temp_splits_env)
    cv_path = os.path.join(temp_splits_env, "cv_folds.csv")
    cv_df = pd.read_csv(cv_path)
    cv_df = cv_df.iloc[1:].reset_index(drop=True)
    cv_df.to_csv(cv_path, index=False)

    with pytest.raises(ValueError, match="exactly the development patients"):
        load_and_validate_splits(manifest, splits_dir=temp_splits_env)
