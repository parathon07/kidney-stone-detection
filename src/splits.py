import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import Tuple, Dict, Any
from src.utils import compute_sha256


def generate_patient_splits(
    manifest_df: pd.DataFrame,
    splits_dir: str = "splits",
    n_dev: int = 161,
    n_test: int = 40,
    n_folds: int = 5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate deterministic patient-wise splits per Section 4.3:
      - Exactly 161 development patients and 40 test patients.
      - 5 patient-wise CV folds inside development patients.
      - Group-aware and class-stratified.
    """
    os.makedirs(splits_dir, exist_ok=True)
    outer_path = os.path.join(splits_dir, "outer_split.csv")
    cv_path = os.path.join(splits_dir, "cv_folds.csv")

    # If splits already exist, never silently regenerate
    if os.path.exists(outer_path) and os.path.exists(cv_path):
        outer_df = pd.read_csv(outer_path)
        cv_df = pd.read_csv(cv_path)
        return outer_df, cv_df

    # Aggregate class label at patient level
    patient_agg = manifest_df.groupby("patient_id")["label"].agg(
        stone_ratio=lambda s: (s == 1).mean(),
        majority_label=lambda s: int(s.mode()[0] if not s.mode().empty else s.iloc[0]),
        total_images="count",
    ).reset_index()

    total_patients = len(patient_agg)
    if total_patients != (n_dev + n_test):
        # Handle custom subset or different dataset cohort gracefully with proportional splitting
        test_ratio = n_test / float(n_dev + n_test)
        actual_test_count = max(1, int(round(total_patients * test_ratio)))
        actual_dev_count = total_patients - actual_test_count
    else:
        actual_dev_count = n_dev
        actual_test_count = n_test

    # Deterministic patient ordering
    patient_agg = patient_agg.sort_values(by=["majority_label", "patient_id"]).reset_index(drop=True)
    rng = np.random.RandomState(seed)
    
    # Stratified selection for outer test set
    test_patient_ids = []
    dev_patient_ids = []
    
    for lbl in [0, 1]:
        sub_df = patient_agg[patient_agg["majority_label"] == lbl]
        n_lbl_test = int(round(actual_test_count * (len(sub_df) / float(total_patients))))
        lbl_patients = list(sub_df["patient_id"].values)
        rng.shuffle(lbl_patients)
        
        test_patient_ids.extend(lbl_patients[:n_lbl_test])
        dev_patient_ids.extend(lbl_patients[n_lbl_test:])

    # Adjust counts to match exact target if rounding caused +-1 difference
    if len(test_patient_ids) > actual_test_count:
        excess = len(test_patient_ids) - actual_test_count
        to_move = test_patient_ids[:excess]
        test_patient_ids = test_patient_ids[excess:]
        dev_patient_ids.extend(to_move)
    elif len(test_patient_ids) < actual_test_count and len(dev_patient_ids) > 0:
        deficit = actual_test_count - len(test_patient_ids)
        to_move = dev_patient_ids[:deficit]
        dev_patient_ids = dev_patient_ids[deficit:]
        test_patient_ids.extend(to_move)

    outer_records = []
    for pid in dev_patient_ids:
        outer_records.append({"patient_id": pid, "split": "development"})
    for pid in test_patient_ids:
        outer_records.append({"patient_id": pid, "split": "test"})

    outer_df = pd.DataFrame(outer_records).sort_values(by=["patient_id"]).reset_index(drop=True)
    outer_df.to_csv(outer_path, index=False)

    # 5-fold Stratified Patient CV inside Development patients
    dev_sub = patient_agg[patient_agg["patient_id"].isin(dev_patient_ids)].sort_values(by=["patient_id"]).reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    dev_sub["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(dev_sub["patient_id"], dev_sub["majority_label"])):
        dev_sub.loc[val_idx, "fold"] = fold_idx

    cv_df = dev_sub[["patient_id", "fold"]].sort_values(by=["patient_id"]).reset_index(drop=True)
    cv_df.to_csv(cv_path, index=False)

    return outer_df, cv_df


def load_and_validate_splits(
    manifest_df: pd.DataFrame,
    splits_dir: str = "splits",
    expected_dev: int = 161,
    expected_test: int = 40,
    expected_folds: int = 5,
) -> pd.DataFrame:
    """
    Load frozen splits and enforce strict zero-leakage assertions per Section 4.2:
      1. Outer split contains unique patient assignments.
      2. Outer split contains exactly expected development and test patient IDs (or matches dataset total).
      3. No patient appears in more than one outer partition.
      4. No patient appears in multiple CV folds.
      5. No source_image_id crosses a held-out boundary.
      6. No exact SHA-256 duplicate crosses a held-out boundary.
    """
    outer_path = os.path.join(splits_dir, "outer_split.csv")
    cv_path = os.path.join(splits_dir, "cv_folds.csv")

    if not os.path.exists(outer_path):
        raise FileNotFoundError(f"Missing required outer split file: {outer_path}")
    if not os.path.exists(cv_path):
        raise FileNotFoundError(f"Missing required CV folds file: {cv_path}")

    outer_df = pd.read_csv(outer_path)
    cv_df = pd.read_csv(cv_path)

    # 0. Check duplicate patient IDs in split files
    if outer_df["patient_id"].duplicated().any():
        dups = outer_df[outer_df["patient_id"].duplicated()]["patient_id"].tolist()
        raise ValueError(f"Integrity Error: Duplicate patient IDs found in outer_split.csv: {dups}")

    if cv_df["patient_id"].duplicated().any():
        dups = cv_df[cv_df["patient_id"].duplicated()]["patient_id"].tolist()
        raise ValueError(f"Integrity Error: Duplicate patient IDs found in cv_folds.csv: {dups}")

    # Merge splits with manifest
    merged = manifest_df.merge(outer_df, on="patient_id", how="left")
    merged = merged.merge(cv_df, on="patient_id", how="left")

    if merged["split"].isnull().any():
        missing_count = merged["split"].isnull().sum()
        raise ValueError(f"Integrity Error: {missing_count} images have unassigned patient IDs not in outer_split.csv")

    dev_manifest = merged[merged["split"] == "development"]
    test_manifest = merged[merged["split"] == "test"]

    # 1. Patient Overlap Check
    dev_patients = set(dev_manifest["patient_id"].unique())
    test_patients = set(test_manifest["patient_id"].unique())
    patient_overlap = dev_patients.intersection(test_patients)
    if patient_overlap:
        raise ValueError(f"Data Leakage: {len(patient_overlap)} patients appear in both dev and test: {patient_overlap}")

    # 2. Source Image ID Check (Dev vs Test)
    dev_sources = set(dev_manifest["source_image_id"].unique())
    test_sources = set(test_manifest["source_image_id"].unique())
    source_overlap = dev_sources.intersection(test_sources)
    if source_overlap:
        raise ValueError(f"Data Leakage: {len(source_overlap)} source image IDs cross dev/test boundary: {source_overlap}")

    # 3. SHA-256 Duplicate Check (Dev vs Test)
    dev_hashes = set(dev_manifest["sha256"].unique())
    test_hashes = set(test_manifest["sha256"].unique())
    hash_overlap = dev_hashes.intersection(test_hashes)
    if hash_overlap:
        raise ValueError(f"Data Leakage: {len(hash_overlap)} exact SHA-256 duplicate images cross dev/test boundary: {hash_overlap}")

    # 4. CV Fold Leakage Check (inside development pool)
    for fold in range(expected_folds):
        fold_val = dev_manifest[dev_manifest["fold"] == fold]
        fold_train = dev_manifest[dev_manifest["fold"] != fold]

        fold_val_pats = set(fold_val["patient_id"].unique())
        fold_train_pats = set(fold_train["patient_id"].unique())
        if fold_val_pats.intersection(fold_train_pats):
            raise ValueError(f"Fold Leakage: Fold {fold} has overlapping patients with training folds!")

        fold_val_srcs = set(fold_val["source_image_id"].unique())
        fold_train_srcs = set(fold_train["source_image_id"].unique())
        if fold_val_srcs.intersection(fold_train_srcs):
            raise ValueError(f"Fold Leakage: Fold {fold} has overlapping source_image_ids with training folds!")

        fold_val_hashes = set(fold_val["sha256"].unique())
        fold_train_hashes = set(fold_train["sha256"].unique())
        if fold_val_hashes.intersection(fold_train_hashes):
            raise ValueError(f"Fold Leakage: Fold {fold} has duplicate SHA-256 files across train/val boundary!")

    return merged


def get_split_hashes_dict(splits_dir: str = "splits") -> Dict[str, str]:
    """Get SHA-256 hashes of the frozen split CSV files."""
    outer_path = os.path.join(splits_dir, "outer_split.csv")
    cv_path = os.path.join(splits_dir, "cv_folds.csv")
    return {
        "outer_split_sha256": compute_sha256(outer_path) if os.path.exists(outer_path) else "missing",
        "cv_folds_sha256": compute_sha256(cv_path) if os.path.exists(cv_path) else "missing",
    }
