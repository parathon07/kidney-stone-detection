import os
import re
import hashlib
import pandas as pd
import tensorflow as tf
from PIL import Image
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
    
    # Check directory and filename suffixes
    mendeley_aug_suffixes = ["do", "el", "fh", "fv", "gb", "gn", "mul", "pa", "pe"]
    aug_keywords = ["augmented", "aug_", "_aug", "_rot", "_flip", "_trans", "_scale", "_bright", "_contrast", "_noise", "_zoom"]
    
    if "augmented" in normalized_path:
        is_augmented = True
        aug_type = "augmented"
        # Try to find specific transform code at the end of stem
        for code in mendeley_aug_suffixes:
            if stem.endswith(f"_{code}"):
                aug_type = code.upper()
                break
    else:
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
    
    os.makedirs(os.path.dirname(output_manifest_path), exist_ok=True)
    manifest_df.to_csv(output_manifest_path, index=False)
    
    return manifest_df


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

    # In Keras RandomRotation: factor is fraction of 2*pi (e.g. 10/360 = 0.02777)
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


def apply_mild_contrast(image: tf.Tensor, contrast_fraction: float = 0.10) -> tf.Tensor:
    """Apply mild random contrast +-10% and clip to [0, 1]."""
    lower = 1.0 - contrast_fraction
    upper = 1.0 + contrast_fraction
    image = tf.image.random_contrast(image, lower=lower, upper=upper)
    return tf.clip_by_value(image, 0.0, 1.0)


def create_dataset(
    manifest_df: pd.DataFrame,
    data_dir: str,
    config: Dict[str, Any],
    split_type: str = "train",
    shuffle_buffer: int = 1024,
) -> tf.data.Dataset:
    """
    Build tf.data pipeline per Section 5.3:
      TRAIN: manifest filter -> shuffle(seed) -> decode/preprocess -> augment -> batch -> prefetch(AUTOTUNE)
      VAL:   manifest filter -> decode/preprocess -> batch -> prefetch(AUTOTUNE)
      TEST:  manifest filter -> decode/preprocess -> batch -> prefetch(AUTOTUNE)
    """
    image_size = config.get("data", {}).get("image_size", 384)
    batch_size = config.get("training", {}).get("batch_size", 16)
    seed = config.get("seed", 42)
    contrast_fraction = config.get("data", {}).get("augmentation", {}).get("contrast_fraction", 0.10)

    # Filter original images if V1 default
    use_derivatives = config.get("data", {}).get("use_supplied_derivatives", False)
    if not use_derivatives or split_type in ["val", "test"]:
        manifest_df = manifest_df[manifest_df["is_original"] == True].copy()

    file_paths = [os.path.join(data_dir, p).replace("\\", "/") for p in manifest_df["path"].tolist()]
    labels = manifest_df["label"].astype(int).tolist()

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if split_type == "train":
        ds = ds.shuffle(buffer_size=min(len(file_paths), shuffle_buffer), seed=seed, reshuffle_each_iteration=True)
        
        # 1. Decode & Preprocess
        ds = ds.map(
            lambda path, lbl: canonical_preprocess_image(path, lbl, image_size=image_size),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        
        # 2. Online Augmentation
        aug_layer = build_augmentation_layer(config)
        ds = ds.map(
            lambda img, lbl: (apply_mild_contrast(aug_layer(img, training=True), contrast_fraction), lbl),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        # VAL / TEST: Canonical preprocessing only (NO augmentation)
        ds = ds.map(
            lambda path, lbl: canonical_preprocess_image(path, lbl, image_size=image_size),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
