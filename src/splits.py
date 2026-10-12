import os
import pandas as pd
import numpy as np
import networkx as nx
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
    Generate deterministic patient-wise splits per Section 4.3 using group-aware connected component clustering:
      - Exactly 161 development patients and 40 test patients (or matching dataset proportion).
      - 5 patient-wise CV folds inside development patients.
      - Patients sharing exact SHA-256 duplicate slices or source IDs are kept in the same partition.
      - Group-aware and class-stratified.
    """
    os.makedirs(splits_dir, exist_ok=True)
    outer_path = os.path.join(splits_dir, "outer_split.csv")
    cv_path = os.path.join(splits_dir, "cv_folds.csv")

    # If either file exists, the frozen pair must be completed and validated
    # explicitly rather than partially regenerated.
    if os.path.exists(outer_path) or os.path.exists(cv_path):
        if not (os.path.exists(outer_path) and os.path.exists(cv_path)):
            raise FileExistsError(
                "A frozen split is incomplete: both outer_split.csv and cv_folds.csv "
                "must exist before generation can be reused."
            )
        outer_df = pd.read_csv(outer_path)
        cv_df = pd.read_csv(cv_path)
        return outer_df, cv_df

    # 1. Build Patient Graph to link any patients sharing duplicate SHA256 hashes or source IDs
    G = nx.Graph()
    for p in manifest_df["patient_id"].unique():
        G.add_node(p)
        
    for _, pats in manifest_df.groupby("sha256")["patient_id"].unique().items():
        if len(pats) > 1:
            for i in range(len(pats) - 1):
                G.add_edge(pats[i], pats[i + 1])

    for _, pats in manifest_df.groupby("source_image_id")["patient_id"].unique().items():
        if len(pats) > 1:
            for i in range(len(pats) - 1):
                G.add_edge(pats[i], pats[i + 1])

    components = [sorted(list(c)) for c in nx.connected_components(G)]
    
    # A patient must have one label. Do not silently resolve inconsistent
    # manifest rows by majority vote.
    patient_label_values = manifest_df.groupby("patient_id")["label"].agg(
        lambda s: sorted(set(int(value) for value in s))
    )
    conflicting_patients = patient_label_values[patient_label_values.map(len) > 1]
    if not conflicting_patients.empty:
        details = {
            str(patient_id): values
            for patient_id, values in conflicting_patients.items()
        }
        raise ValueError(f"Conflicting class labels within patient records: {details}")
    pat_labels = patient_label_values.map(lambda values: values[0]).to_dict()

    rng = np.random.default_rng(seed)
    comp_records = []
    for c in components:
        component_labels = {pat_labels[p] for p in c}
        if len(component_labels) > 1:
            label_mapping = {p: pat_labels[p] for p in c}
            raise ValueError(
                "Conflicting class labels within connected patient component "
                f"{c}: {label_mapping}"
            )
        comp_records.append({
            "comp": c,
            "size": len(c),
            "label": list(component_labels)[0],
            "id": c[0],
            # Randomized tie-breaking makes the requested seed meaningful
            # while preserving class-first and component-size ordering.
            "random_order": float(rng.random()),
        })

    # Sort components deterministically
    comp_records = sorted(
        comp_records,
        key=lambda x: (x["label"], -x["size"], x["random_order"], x["id"]),
    )

    total_patients = len(manifest_df["patient_id"].unique())
    if total_patients != (n_dev + n_test):
        test_ratio = n_test / float(n_dev + n_test)
        actual_test_count = max(1, int(round(total_patients * test_ratio)))
    else:
        actual_test_count = n_test

    target_test_per_label = {
        0: actual_test_count // 2,
        1: actual_test_count - (actual_test_count // 2),
    }

    test_pats = []
    dev_comps = []
    curr_test = {0: 0, 1: 0}

    for c in comp_records:
        lbl = c["label"]
        if curr_test[lbl] + c["size"] <= target_test_per_label[lbl]:
            test_pats.extend(c["comp"])
            curr_test[lbl] += c["size"]
        else:
            dev_comps.append(c)

    # If any shortfall in test due to discrete component sizes, fill with remaining singletons
    if len(test_pats) < actual_test_count and dev_comps:
        rem_needed = actual_test_count - len(test_pats)
        remaining_dev = []
        for c in dev_comps:
            if c["size"] <= rem_needed and rem_needed > 0:
                test_pats.extend(c["comp"])
                rem_needed -= c["size"]
            else:
                remaining_dev.append(c)
        dev_comps = remaining_dev

    if len(test_pats) != actual_test_count:
        raise ValueError(
            f"Unable to construct the requested test partition: expected {actual_test_count} "
            f"patients, selected {len(test_pats)} without splitting a connected component."
        )

    # 5-fold Stratified Patient CV inside Development components
    fold_pats = {i: [] for i in range(n_folds)}
    for lbl in [0, 1]:
        lbl_comps = [c for c in dev_comps if c["label"] == lbl]
        for c in lbl_comps:
            # Assign component to fold with fewest patients of this label
            best_f = min(
                range(n_folds),
                key=lambda f: (sum(1 for p in fold_pats[f] if pat_labels[p] == lbl), len(fold_pats[f])),
            )
            fold_pats[best_f].extend(c["comp"])

    if any(not fold_pats[f] for f in range(n_folds)):
        raise ValueError("Unable to construct non-empty CV folds from the patient components.")

    outer_rows = []
    cv_rows = []
    for f in range(n_folds):
        for p in fold_pats[f]:
            outer_rows.append({"patient_id": p, "split": "development"})
            cv_rows.append({"patient_id": p, "fold": f})

    for p in test_pats:
        outer_rows.append({"patient_id": p, "split": "test"})

    outer_df = pd.DataFrame(outer_rows).sort_values(by=["patient_id"]).reset_index(drop=True)
    cv_df = pd.DataFrame(cv_rows).sort_values(by=["patient_id"]).reset_index(drop=True)

    outer_df.to_csv(outer_path, index=False)
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

    required_outer = {"patient_id", "split"}
    required_cv = {"patient_id", "fold"}
    if not required_outer.issubset(outer_df.columns):
        raise ValueError(
            f"Integrity Error: outer_split.csv must contain {sorted(required_outer)}"
        )
    if not required_cv.issubset(cv_df.columns):
        raise ValueError(
            f"Integrity Error: cv_folds.csv must contain {sorted(required_cv)}"
        )
    if manifest_df["patient_id"].isna().any():
        raise ValueError("Integrity Error: manifest contains null patient IDs")
    if manifest_df["label"].isna().any():
        raise ValueError("Integrity Error: manifest contains null labels")
    labels = pd.to_numeric(manifest_df["label"], errors="coerce")
    if labels.isna().any() or not set(labels.astype(int).unique()).issubset({0, 1}):
        raise ValueError("Integrity Error: manifest labels must be exactly 0 or 1")
    if manifest_df["patient_id"].astype(str).str.lower().eq("unknown").any():
        raise ValueError("Integrity Error: manifest contains unknown patient IDs")

    if outer_df["patient_id"].isna().any() or cv_df["patient_id"].isna().any():
        raise ValueError("Integrity Error: split files contain null patient IDs")
    if not outer_df["split"].isin(["development", "test"]).all():
        bad = sorted(set(outer_df.loc[~outer_df["split"].isin(["development", "test"]), "split"]))
        raise ValueError(f"Integrity Error: invalid outer split values: {bad}")

    # 0. Check duplicate patient IDs in split files
    if outer_df["patient_id"].duplicated().any():
        dups = outer_df[outer_df["patient_id"].duplicated()]["patient_id"].tolist()
        raise ValueError(f"Integrity Error: Duplicate patient IDs found in outer_split.csv: {dups}")

    if cv_df["patient_id"].duplicated().any():
        dups = cv_df[cv_df["patient_id"].duplicated()]["patient_id"].tolist()
        raise ValueError(f"Integrity Error: Duplicate patient IDs found in cv_folds.csv: {dups}")

    manifest_patients = set(manifest_df["patient_id"].unique())
    split_patients = set(outer_df["patient_id"].unique())
    missing_outer = manifest_patients - split_patients
    extra_outer = split_patients - manifest_patients
    if missing_outer or extra_outer:
        raise ValueError(
            "Integrity Error: outer_split.csv patient IDs do not exactly match the manifest "
            f"(missing={sorted(missing_outer)[:5]}, extra={sorted(extra_outer)[:5]})"
        )

    # Merge splits with manifest
    merged = manifest_df.merge(outer_df, on="patient_id", how="left")
    merged = merged.merge(cv_df, on="patient_id", how="left")

    if merged["split"].isnull().any():
        missing_count = merged["split"].isnull().sum()
        raise ValueError(f"Integrity Error: {missing_count} images have unassigned patient IDs not in outer_split.csv")

    dev_manifest = merged[merged["split"] == "development"]
    test_manifest = merged[merged["split"] == "test"]

    # 1. Patient Overlap & Partition Count Checks
    dev_patients = set(dev_manifest["patient_id"].unique())
    test_patients = set(test_manifest["patient_id"].unique())
    patient_overlap = dev_patients.intersection(test_patients)
    if patient_overlap:
        raise ValueError(f"Data Leakage: {len(patient_overlap)} patients appear in both dev and test: {patient_overlap}")

    if len(dev_patients) != expected_dev:
        raise ValueError(f"Integrity Error: Expected {expected_dev} development patients, found {len(dev_patients)}")
    if len(test_patients) != expected_test:
        raise ValueError(f"Integrity Error: Expected {expected_test} test patients, found {len(test_patients)}")

    # Every development patient must have exactly one valid fold, and test
    # patients must not have a CV assignment at all.
    cv_patients = set(cv_df["patient_id"].unique())
    missing_cv = dev_patients - cv_patients
    extra_cv = cv_patients - dev_patients
    if missing_cv or extra_cv:
        raise ValueError(
            "Integrity Error: cv_folds.csv must contain exactly the development patients "
            f"(missing={sorted(missing_cv)[:5]}, extra={sorted(extra_cv)[:5]})"
        )
    numeric_folds = pd.to_numeric(cv_df["fold"], errors="coerce")
    if numeric_folds.isna().any() or (numeric_folds != numeric_folds.astype(int)).any():
        raise ValueError("Integrity Error: CV fold values must be integers")
    if not numeric_folds.astype(int).isin(range(expected_folds)).all():
        raise ValueError(
            f"Integrity Error: CV fold values must be in 0..{expected_folds - 1}"
        )

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

        if fold_val.empty:
            raise ValueError(f"Integrity Error: CV fold {fold} is empty")
        if fold_val["label"].nunique() < 2:
            raise ValueError(f"Integrity Error: CV fold {fold} does not contain both classes")

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
