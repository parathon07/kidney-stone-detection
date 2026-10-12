import os
import re
import math
import hashlib
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Tuple, Optional, List


def parse_filename_metadata(rel_path: str) -> Dict[str, Any]:
    """
    Parse relative file path from Mendeley V2 / CT dataset to extract metadata.
    
    Extracts:
      - label (0: Non-Stone, 1: Stone)
      - is_original (bool)
      - augmentation_type (str)
      - patient_id (str)
      - source_image_id (str)
      - hospital / metadata (str)
    """
    normalized_path = rel_path.replace("\\", "/").lower()
    filename = os.path.basename(normalized_path)
    stem, _ = os.path.splitext(filename)

    # 1. Determine Label
    if any(k in normalized_path for k in ["non_stone", "non-stone", "normal", "nonstone", "no_stone", "negative"]):
        label = 0
    elif any(k in normalized_path for k in ["stone", "kidney_stone", "kidneystone", "positive", "calculus"]):
        label = 1
    else:
        # Fallback to binary detection in path or filename
        if "0" in normalized_path.split("/") or "class_0" in normalized_path:
            label = 0
        elif "1" in normalized_path.split("/") or "class_1" in normalized_path:
            label = 1
        else:
            raise ValueError(f"Unable to parse class label for image: {rel_path}")

    # 2. Check for Augmented vs Original
    is_augmented = False
    aug_type = "none"
    
    # Mendeley V2 specific augmentation suffixes
    mendeley_aug_suffixes = ["do", "el", "fh", "fv", "gb", "gn", "mul", "pa", "pe", "cr", "rot", "lc"]
    aug_keywords = ["augmented", "aug_", "_aug", "_rot", "_flip", "_trans", "_scale", "_bright", "_contrast", "_noise", "_zoom"]
    
    # Priority 1: Check filename suffix code directly
    for code in mendeley_aug_suffixes:
        if stem.endswith(f"_{code}"):
            is_augmented = True
            aug_type = code.upper()
            break

    # Priority 2: Check explicit augmentation keywords in path/filename if not matched above
    if not is_augmented:
        for kw in aug_keywords:
            if kw in normalized_path:
                is_augmented = True
                aug_type = kw.strip("_/")
                break

    is_original = not is_augmented

    # 3. Extract Patient ID & Source Image ID
    # Pattern e.g., P001_FA_M_NS_I01 or P135_RA_F_S_I13_PE
    patient_id = "unknown"
    
    pat_match = re.search(r"(p\d+|patient[_\-\s]?\d+|case[_\-\s]?\d+)", stem, re.IGNORECASE)
    if pat_match:
        patient_id = pat_match.group(1).upper().replace(" ", "_").replace("-", "_")
    else:
        parts = normalized_path.split("/")
        for part in parts[:-1]:
            folder_pat = re.search(r"(p\d+|patient[_\-\s]?\d+|case[_\-\s]?\d+)", part, re.IGNORECASE)
            if folder_pat:
                patient_id = folder_pat.group(1).upper().replace(" ", "_").replace("-", "_")
                break
        
        if patient_id == "unknown":
            num_match = re.search(r"\((\d+)\)", stem)
            if num_match:
                patient_id = f"PAT_{'STONE' if label == 1 else 'NONSTONE'}_{num_match.group(1)}"
            else:
                patient_id = f"PAT_{'STONE' if label == 1 else 'NONSTONE'}_{stem}"

    # Strip augmentation suffixes to establish canonical source_image_id
    clean_source = stem
    for code in mendeley_aug_suffixes:
        if clean_source.endswith(f"_{code}"):
            clean_source = clean_source[:-len(f"_{code}")]
            break
    for kw in aug_keywords:
        clean_source = re.sub(rf"{kw}[a-zA-Z0-9_\-]*", "", clean_source, flags=re.IGNORECASE)
    clean_source = clean_source.strip("_-")
    source_image_id = f"{'stone' if label == 1 else 'nonstone'}_{clean_source}"

    # Hospital/metadata site tag if present (e.g. FA, ME, RA in Mendeley dataset)
    hospital = "default_site"
    mendeley_parts = stem.split("_")
    if len(mendeley_parts) >= 5 and mendeley_parts[0].upper().startswith("P"):
        hospital = mendeley_parts[1].upper()
    else:
        for site in ["hospital_a", "hospital_b", "site1", "site2", "center1", "center2", "fa", "me", "ra"]:
            if site in normalized_path:
                hospital = site.upper()
                break

    return {
        "label": label,
        "is_original": is_original,
        "augmentation_type": aug_type,
        "patient_id": patient_id,
        "source_image_id": source_image_id,
        "hospital": hospital,
    }


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _process_single_image(args: Tuple[str, str]) -> Dict[str, Any]:
    full_path, rel_path = args
    try:
        with Image.open(full_path) as img:
            img.load()
            width, height = img.size
            channels = len(img.getbands())
    except Exception as e:
        raise RuntimeError(f"Corrupt or unreadable image file: {full_path} - {str(e)}")

    sha256_hash = compute_file_sha256(full_path)
    meta = parse_filename_metadata(rel_path)

    return {
        "path": rel_path,
        "sha256": sha256_hash,
        "patient_id": meta["patient_id"],
        "source_image_id": meta["source_image_id"],
        "label": meta["label"],
        "is_original": meta["is_original"],
        "augmentation_type": meta["augmentation_type"],
        "hospital": meta["hospital"],
        "height": height,
        "width": width,
        "channels": channels,
    }


def scan_dataset_and_create_manifest(
    data_dir: str,
    output_manifest_path: str = "data/manifest.csv",
    max_workers: int = 16,
) -> pd.DataFrame:
    """
    Scan dataset directory concurrently, decode every image for dimensions, hash contents,
    parse metadata, and write canonical data/manifest.csv.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    supported_exts = {".jpg", ".jpeg", ".png"}
    tasks = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_exts:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, data_dir).replace("\\", "/")
                tasks.append((full_path, rel_path))

    if not tasks:
        raise ValueError(f"No valid images found in dataset directory: {data_dir}")

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        records = list(executor.map(_process_single_image, tasks))

    manifest_df = pd.DataFrame(records)
    
    # Sort deterministically
    manifest_df = manifest_df.sort_values(by=["path"]).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_manifest_path) or ".", exist_ok=True)
    manifest_df.to_csv(output_manifest_path, index=False)
    
    return manifest_df


def validate_manifest_files(manifest_df: pd.DataFrame, data_dir: str) -> Dict[str, Any]:
    """Verify that manifest paths, bytes, and image decoding still match disk.

    The manifest hash protects metadata, not the directory passed to a later
    command. This audit closes that gap before a fit or final evaluation.
    """
    required_columns = {
        "path", "sha256", "patient_id", "source_image_id", "label",
        "is_original", "height", "width", "channels",
    }
    missing = sorted(required_columns.difference(manifest_df.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    if manifest_df.empty:
        raise ValueError("Manifest is empty.")
    if manifest_df["path"].isna().any() or manifest_df["path"].duplicated().any():
        raise ValueError("Manifest paths must be non-null and unique.")
    if manifest_df[["sha256", "patient_id", "source_image_id", "label"]].isna().any().any():
        raise ValueError("Manifest contains null provenance or label values.")

    labels = pd.to_numeric(manifest_df["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Manifest labels must be exactly 0 or 1.")
    if manifest_df["patient_id"].astype(str).str.lower().eq("unknown").any():
        raise ValueError("Manifest contains unknown patient IDs.")

    original_values = manifest_df["is_original"]
    if pd.api.types.is_bool_dtype(original_values):
        original_values = original_values.astype(bool)
    else:
        normalized_originals = original_values.astype(str).str.strip().str.lower()
        if not normalized_originals.isin(["true", "false"]).all():
            raise ValueError("Manifest is_original values must be boolean.")
        original_values = normalized_originals.eq("true")

    dimensions = manifest_df[["height", "width", "channels"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if (
        dimensions.isna().any().any()
        or (dimensions <= 0).any().any()
        or (dimensions != dimensions.round()).any().any()
    ):
        raise ValueError("Manifest image dimensions must be positive integers.")

    root = os.path.normcase(os.path.abspath(data_dir))
    rows = list(
        manifest_df[["path", "sha256", "height", "width", "channels"]]
        .itertuples(index=False, name=None)
    )

    def verify_one(row: Tuple[str, str, int, int, int]) -> Optional[str]:
        rel_path, expected_hash, expected_height, expected_width, expected_channels = row
        rel_path = str(rel_path).replace("\\", "/")
        full_path = os.path.abspath(os.path.join(data_dir, rel_path))
        full_norm = os.path.normcase(full_path)
        try:
            if os.path.commonpath([root, full_norm]) != root:
                return f"path escapes data_dir: {rel_path}"
        except ValueError:
            return f"path is on a different volume: {rel_path}"
        if not os.path.isfile(full_path):
            return f"missing file: {rel_path}"
        try:
            actual_hash = compute_file_sha256(full_path)
            if actual_hash.lower() != str(expected_hash).lower():
                return f"SHA-256 mismatch: {rel_path}"
            with Image.open(full_path) as image:
                image.load()
                actual_width, actual_height = image.size
                actual_channels = len(image.getbands())
            if (
                actual_height != int(expected_height)
                or actual_width != int(expected_width)
                or actual_channels != int(expected_channels)
            ):
                return f"image metadata mismatch: {rel_path}"
        except Exception as exc:
            return f"unreadable file {rel_path}: {exc}"
        return None

    max_workers = min(16, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        errors = [error for error in executor.map(verify_one, rows) if error]
    if errors:
        preview = "; ".join(errors[:5])
        suffix = "" if len(errors) <= 5 else f"; and {len(errors) - 5} more"
        raise RuntimeError(f"Manifest file audit failed: {preview}{suffix}")

    return {
        "file_count": int(len(rows)),
        "original_count": int(original_values.sum()),
        "derivative_count": int((~original_values).sum()),
    }


def canonical_preprocess_image(
    file_path: tf.Tensor,
    label: Optional[tf.Tensor] = None,
    image_size: int = 384,
) -> Tuple[tf.Tensor, Any]:
    """
    Canonical preprocessing per Section 5.1:
      1. Read JPG bytes with TensorFlow and decode deterministically as 1 grayscale channel.
      2. Convert to float32.
      3. Resize preserving aspect ratio and pad to 384x384 using tf.image.resize_with_pad(antialias=True).
      4. Scale pixel values to [0, 1].
      5. Return tensor shape (384, 384, 1) and label (if provided).
    """
    img_bytes = tf.io.read_file(file_path)
    # decode_image with 1 channel handles jpeg, png deterministically
    image = tf.io.decode_image(img_bytes, channels=1, expand_animations=False)
    image = tf.cast(image, tf.float32)
    
    # Aspect-ratio preserving resize with pad
    image = tf.image.resize_with_pad(image, target_height=image_size, target_width=image_size, antialias=True)
    
    # Scale to [0, 1]
    image = image / 255.0
    
    # Ensure static shape for graph execution
    image.set_shape([image_size, image_size, 1])

    if label is not None:
        label = tf.cast(label, tf.int32)
        return image, label
    return image


def build_augmentation_layer(config: Dict[str, Any]) -> tf.keras.Sequential:
    """
    Mild online augmentation pipeline per Section 5.2:
      - Rotation: +-10 deg
      - Translation: +-3% height/width
      - Zoom: +-5%
      - Contrast: +-10%
      - Flips / Elastic: DISABLED
    """
    aug_cfg = config.get("data", {}).get("augmentation", {})
    rotation_deg = aug_cfg.get("rotation_deg", 10)
    translation_fraction = aug_cfg.get("translation_fraction", 0.03)
    zoom_fraction = aug_cfg.get("zoom_fraction", 0.05)

    rot_factor = rotation_deg / 360.0

    aug_layers = [
        tf.keras.layers.RandomRotation(factor=(-rot_factor, rot_factor), fill_mode="constant", fill_value=0.0),
        tf.keras.layers.RandomTranslation(
            height_factor=(-translation_fraction, translation_fraction),
            width_factor=(-translation_fraction, translation_fraction),
            fill_mode="constant",
            fill_value=0.0,
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-zoom_fraction, zoom_fraction),
            width_factor=(-zoom_fraction, zoom_fraction),
            fill_mode="constant",
            fill_value=0.0,
        ),
    ]
    return tf.keras.Sequential(aug_layers, name="online_augmentation")


def _fold_random_seed(seed: tf.Tensor, salt: int) -> tf.Tensor:
    """Derive independent stateless random streams for one augmentation."""
    return tf.random.experimental.stateless_fold_in(tf.cast(seed, tf.int32), salt)


def _random_uniform(
    shape: Any,
    minval: float,
    maxval: float,
    seed: Optional[tf.Tensor],
    salt: int,
) -> tf.Tensor:
    if seed is None:
        return tf.random.uniform(shape, minval=minval, maxval=maxval)
    return tf.random.stateless_uniform(
        shape,
        seed=_fold_random_seed(seed, salt),
        minval=minval,
        maxval=maxval,
    )


def apply_mild_contrast(
    image: tf.Tensor,
    contrast_fraction: float = 0.10,
    seed: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Apply mild random contrast +-10% and clip to [0, 1]."""
    lower = 1.0 - contrast_fraction
    upper = 1.0 + contrast_fraction
    if seed is None:
        image = tf.image.random_contrast(image, lower=lower, upper=upper)
    else:
        image = tf.image.stateless_random_contrast(
            image, lower=lower, upper=upper, seed=_fold_random_seed(seed, 0)
        )
    return tf.clip_by_value(image, 0.0, 1.0)


def apply_mild_augmentation(
    image: tf.Tensor,
    config: Dict[str, Any],
    image_size: int = 384,
    seed: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """
    Mild online augmentation pipeline per Section 5.2:
      - Rotation: +-10 deg
      - Translation: +-3% height/width
      - Zoom: +-5%
      - Contrast: +-10%
      - Flips / Elastic: DISABLED
    Implemented via pure TensorFlow operations for native graph / tf.data execution.
    """
    aug_cfg = config.get("data", {}).get("augmentation", {})
    rotation_deg = float(aug_cfg.get("rotation_deg", 10.0))
    translation_fraction = float(aug_cfg.get("translation_fraction", 0.03))
    zoom_fraction = float(aug_cfg.get("zoom_fraction", 0.05))
    contrast_fraction = float(aug_cfg.get("contrast_fraction", 0.10))
    brightness_fraction = float(aug_cfg.get("brightness_fraction", 0.0))

    # 1. Random rotation angle in radians
    max_angle_rad = rotation_deg * math.pi / 180.0
    angle_rad = _random_uniform([], -max_angle_rad, max_angle_rad, seed, 0)

    # 2. Random translation in pixels
    max_dx = translation_fraction * float(image_size)
    max_dy = translation_fraction * float(image_size)
    dx = _random_uniform([], -max_dx, max_dx, seed, 1)
    dy = _random_uniform([], -max_dy, max_dy, seed, 2)

    # 3. Random zoom scale
    scale = _random_uniform([], 1.0 - zoom_fraction, 1.0 + zoom_fraction, seed, 3)

    # ImageProjectiveTransformV3 samples input coordinates for each output
    # coordinate, so use the inverse of the requested forward transform.
    cos_a = tf.cos(angle_rad) / scale
    sin_a = tf.sin(angle_rad) / scale
    cx = float(image_size) / 2.0
    cy = float(image_size) / 2.0

    a0 = cos_a
    a1 = sin_a
    a2 = cx - cos_a * (cx + dx) - sin_a * (cy + dy)
    b0 = -sin_a
    b1 = cos_a
    b2 = cy + sin_a * (cx + dx) - cos_a * (cy + dy)

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])
    img_4d = tf.expand_dims(image, 0)
    transformed = tf.raw_ops.ImageProjectiveTransformV3(
        images=img_4d,
        transforms=tf.expand_dims(transform, 0),
        output_shape=tf.constant([image_size, image_size], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
        fill_value=0.0,
    )
    img_aug = tf.squeeze(transformed, 0)

    # 4. Mild brightness + contrast (contrast step clips back to [0, 1])
    if seed is None:
        img_aug = tf.image.random_brightness(img_aug, max_delta=brightness_fraction)
        img_aug = apply_mild_contrast(img_aug, contrast_fraction)
    else:
        img_aug = tf.image.stateless_random_brightness(
            img_aug, max_delta=brightness_fraction, seed=_fold_random_seed(seed, 4)
        )
        img_aug = apply_mild_contrast(
            img_aug, contrast_fraction, seed=_fold_random_seed(seed, 5)
        )
    img_aug.set_shape([image_size, image_size, 1])
    return img_aug


def _load_single_image_for_dataset(path: str, image_size: int = 384) -> np.ndarray:
    """Load a single image and canonically preprocess it to a uint8 array.

    Aspect-ratio preserving: pad to square (centered, zero-padded), then resize
    to image_size x image_size — matches canonical_preprocess_image's
    resize_with_pad semantics. Returns uint8 (4x smaller RAM footprint than
    float32); scaling to [0,1] happens inside the tf.data map.
    """
    processed = canonical_preprocess_image(tf.convert_to_tensor(path), image_size=image_size)
    # Keep the RAM cache compact while retaining the same decode/pad/resize
    # semantics as the non-cached path. Quantization is bounded to 1/255.
    arr = np.asarray(processed.numpy(), dtype=np.float32)
    return np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def resolve_shuffle_buffer(dataset_size: int, requested: Optional[int] = None) -> int:
    """Return a safe shuffle buffer, using the full dataset by default."""
    if dataset_size < 0:
        raise ValueError("dataset_size must be non-negative")
    if dataset_size == 0:
        return 0
    if requested is None:
        return dataset_size
    if requested < 1:
        raise ValueError("requested shuffle buffer must be positive")
    return min(dataset_size, requested)


def build_patient_balanced_manifest(manifest_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Resample training rows so each patient contributes equally per epoch.

    The returned manifest has the same row count as the input. Patients with
    fewer than their equal-share quota are sampled with replacement; patients
    with more rows are subsampled. This is deliberately a training-only
    transform and does not alter validation rows, labels, or frozen splits.
    """
    if manifest_df.empty:
        return manifest_df.copy().reset_index(drop=True)
    if "patient_id" not in manifest_df.columns:
        raise ValueError("Patient-balanced sampling requires a 'patient_id' column.")

    patients = sorted(manifest_df["patient_id"].dropna().unique().tolist(), key=str)
    if not patients:
        raise ValueError("Patient-balanced sampling requires at least one patient.")

    total_rows = len(manifest_df)
    base_quota, remainder = divmod(total_rows, len(patients))
    rng = np.random.default_rng(int(seed))
    extra_patients = set(rng.choice(patients, size=remainder, replace=False).tolist())

    parts = []
    for patient_id in patients:
        patient_rows = manifest_df[manifest_df["patient_id"] == patient_id]
        quota = base_quota + (1 if patient_id in extra_patients else 0)
        if quota == 0:
            continue
        replace = quota > len(patient_rows)
        sampled_indices = rng.choice(patient_rows.index.to_numpy(), size=quota, replace=replace)
        parts.append(patient_rows.loc[sampled_indices])

    balanced = pd.concat(parts, axis=0, ignore_index=True)
    shuffled_order = rng.permutation(len(balanced))
    return balanced.iloc[shuffled_order].reset_index(drop=True)


def create_dataset(
    manifest_df: pd.DataFrame,
    data_dir: str,
    config: Dict[str, Any],
    split_type: str = "train",
    shuffle_buffer: Optional[int] = None,
    cache_in_ram: bool = True,
) -> tf.data.Dataset:
    """
    Build high-performance tf.data pipeline per Section 5.3:
      TRAIN: manifest filter -> RAM cache -> full shuffle(seed) -> online augment -> batch -> prefetch(AUTOTUNE)
      VAL:   manifest filter -> RAM cache -> batch -> prefetch(AUTOTUNE)
      TEST:  manifest filter -> RAM cache -> batch -> prefetch(AUTOTUNE)
    """
    image_size = config.get("data", {}).get("image_size", 384)
    batch_size = config.get("training", {}).get("batch_size", 16)
    seed = config.get("seed", 42)
    if "cache_in_ram" in config.get("data", {}):
        cache_in_ram = bool(config["data"]["cache_in_ram"])

    # Filter original images if V1 default
    use_derivatives = config.get("data", {}).get("use_supplied_derivatives", False)
    if not use_derivatives or split_type in ["val", "test"]:
        if "is_original" in manifest_df.columns:
            manifest_df = manifest_df[manifest_df["is_original"] == True].copy().reset_index(drop=True)

    if split_type == "train" and config.get("data", {}).get("patient_balanced_sampling", False):
        manifest_df = build_patient_balanced_manifest(manifest_df, seed=seed)

    file_paths = [os.path.join(data_dir, p).replace("\\", "/") for p in manifest_df["path"].tolist()]
    labels = manifest_df["label"].astype(int).values
    resolved_shuffle_buffer = resolve_shuffle_buffer(len(file_paths), shuffle_buffer)

    if cache_in_ram and len(file_paths) > 0:
        max_workers = min(16, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            imgs_list = list(executor.map(lambda p: _load_single_image_for_dataset(p, image_size=image_size), file_paths))
        imgs_array = np.stack(imgs_list, axis=0)
        del imgs_list

        def _to_float(img, lbl):
            return (tf.cast(img, tf.float32) / 255.0, lbl)

        with tf.device("/CPU:0"):
            ds = tf.data.Dataset.from_tensor_slices((imgs_array, labels))
            del imgs_array
            if split_type == "train":
                ds = ds.shuffle(buffer_size=resolved_shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
                ds = ds.map(_to_float, num_parallel_calls=tf.data.AUTOTUNE)
                ds = ds.enumerate()

                def _augment(index, data):
                    img, lbl = data
                    aug_seed = tf.stack([tf.cast(seed, tf.int32), tf.cast(index, tf.int32)])
                    return apply_mild_augmentation(img, config, image_size=image_size, seed=aug_seed), lbl

                ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
            else:
                ds = ds.map(_to_float, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.batch(batch_size, drop_remainder=False)
            ds = ds.prefetch(tf.data.AUTOTUNE)
            return ds
    else:
        ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        if split_type == "train":
            ds = ds.shuffle(buffer_size=resolved_shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
            ds = ds.map(
                lambda path, lbl: canonical_preprocess_image(path, lbl, image_size=image_size),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            ds = ds.enumerate()

            def _augment(index, data):
                img, lbl = data
                aug_seed = tf.stack([tf.cast(seed, tf.int32), tf.cast(index, tf.int32)])
                return apply_mild_augmentation(img, config, image_size=image_size, seed=aug_seed), lbl

            ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(
                lambda path, lbl: canonical_preprocess_image(path, lbl, image_size=image_size),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
