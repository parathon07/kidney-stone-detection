import os
import pandas as pd
import numpy as np
import keras
from keras.saving import register_keras_serializable
from typing import Dict, Any, Tuple
from src.model import build_model
from src.data import create_dataset
from src.utils import save_json, set_seed


@register_keras_serializable(package="kidney_cnn")
class SmoothedSparseCCE(keras.losses.Loss):
    """Sparse categorical cross-entropy with uniform label smoothing.

    Keras 3.11's SparseCategoricalCrossentropy does not expose label_smoothing.
    Smoothing caps prediction confidence, which keeps val_loss meaningful as an
    early-stopping signal when training on small weakly-labeled datasets.
    Mathematically identical to tf.keras' label_smoothing semantics.
    """

    def __init__(self, label_smoothing=0.0, name="smoothed_sparse_cce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true, y_pred):
        num_classes = y_pred.shape[-1]
        if num_classes is None:
            raise ValueError("SmoothedSparseCCE requires a statically known class dimension.")
        num_classes = int(num_classes)
        onehot = keras.ops.one_hot(keras.ops.cast(y_true, "int32"), num_classes)
        target = onehot * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)
        y_pred = keras.ops.clip(y_pred, 1e-7, 1.0)
        return -keras.ops.sum(target * keras.ops.log(y_pred), axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"label_smoothing": self.label_smoothing})
        return config


def compile_canonical_model(model: keras.Model, config: Dict[str, Any]) -> keras.Model:
    """
    Compile canonical model with AdamW, SparseCategoricalCrossentropy, and minimal metrics.
    BatchNorm gamma/beta and head biases are excluded from weight decay.
    """
    train_cfg = config.get("training", {})
    lr = train_cfg.get("learning_rate", 3e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))

    optimizer = keras.optimizers.AdamW(
        learning_rate=lr,
        weight_decay=weight_decay,
    )
    optimizer.exclude_from_weight_decay(var_names=["gamma", "beta", "bias"])

    if label_smoothing > 0.0:
        loss: keras.losses.Loss = SmoothedSparseCCE(label_smoothing=label_smoothing)
    else:
        loss = keras.losses.SparseCategoricalCrossentropy()

    metrics = [keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def compute_best_epoch(history_df: pd.DataFrame) -> Tuple[int, float]:
    """
    Extract best epoch number (1-indexed count >= 1) and best validation loss
    from training history DataFrame.
    """
    if "val_loss" not in history_df.columns or history_df.empty:
        raise ValueError("history DataFrame must contain non-empty 'val_loss' column.")
    best_epoch_idx = int(history_df["val_loss"].idxmin())
    if "epoch" in history_df.columns:
        raw_epoch = int(history_df.loc[best_epoch_idx, "epoch"])
        # Keras CSVLogger writes 0-indexed epochs (0, 1, 2...). If it starts from 0, convert to 1-indexed count.
        best_epoch_number = raw_epoch + 1 if history_df["epoch"].min() == 0 else raw_epoch
    else:
        best_epoch_number = best_epoch_idx + 1
    best_epoch_number = max(1, best_epoch_number)
    best_val_loss = float(history_df.loc[best_epoch_idx, "val_loss"])
    return best_epoch_number, best_val_loss


def train_single_fold(
    fold_idx: int,
    dev_manifest_df: pd.DataFrame,
    data_dir: str,
    config: Dict[str, Any],
    run_dir: str,
) -> Dict[str, Any]:
    """
    Train a single fold per Section 7.1:
      1. Split into train & val manifests based on fold.
      2. Fresh weights initialization.
      3. Fit with ModelCheckpoint, EarlyStopping, ReduceLROnPlateau.
      4. Reload best.keras.
      5. Predict original validation set and save val_predictions.csv.
      6. Save history.csv and fold_metrics.json.
    """
    fold_dir = os.path.join(run_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    # Deterministic per-fold seeding (seeds torch/keras too)
    seed = config.get("seed", 42)
    set_seed(seed + fold_idx)

    # 1. Manifests (Training uses originals + online aug if V1 default; Val uses originals only)
    use_derivatives = config.get("data", {}).get("use_supplied_derivatives", False)
    if not use_derivatives:
        train_df = dev_manifest_df[
            (dev_manifest_df["fold"] != fold_idx) & (dev_manifest_df["is_original"] == True)
        ].copy().reset_index(drop=True)
    else:
        train_df = dev_manifest_df[dev_manifest_df["fold"] != fold_idx].copy().reset_index(drop=True)

    val_df = dev_manifest_df[
        (dev_manifest_df["fold"] == fold_idx) & (dev_manifest_df["is_original"] == True)
    ].copy().reset_index(drop=True)

    train_ds = create_dataset(train_df, data_dir, config, split_type="train")
    val_ds = create_dataset(val_df, data_dir, config, split_type="val")

    # 2. Fresh Model
    model = build_model(config)
    model = compile_canonical_model(model, config)

    # 3. Callbacks
    best_checkpoint_path = os.path.join(fold_dir, "best.keras")
    history_csv_path = os.path.join(fold_dir, "history.csv")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=best_checkpoint_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.get("training", {}).get("early_stopping_patience", 15),
            min_delta=config.get("training", {}).get("early_stopping_min_delta", 0.01),
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            patience=config.get("training", {}).get("reduce_lr_patience", 5),
            factor=config.get("training", {}).get("reduce_lr_factor", 0.5),
            min_lr=config.get("training", {}).get("min_lr", 1e-6),
            verbose=1,
        ),
        keras.callbacks.CSVLogger(history_csv_path),
    ]

    epochs = config.get("training", {}).get("epochs", 60)
    print(f"\n--- Training Fold {fold_idx} (Train samples: {len(train_df)}, Val samples: {len(val_df)}) ---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # 4. Explicitly reload best.keras checkpoint
    if not os.path.exists(best_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint was not saved: {best_checkpoint_path}")
    
    best_model = keras.models.load_model(best_checkpoint_path, compile=False)

    # 5. Predict all original validation images
    raw_preds = best_model.predict(val_ds, verbose=0)
    stone_probs = raw_preds[:, 1]
    preds_at_0_5 = (stone_probs >= 0.5).astype(int)

    val_predictions_df = pd.DataFrame({
        "path": val_df["path"],
        "patient_id": val_df["patient_id"],
        "source_image_id": val_df["source_image_id"],
        "y_true": val_df["label"].astype(int),
        "stone_probability": stone_probs,
        "predicted_class_at_0_5": preds_at_0_5,
        "fold": fold_idx,
    })
    val_pred_path = os.path.join(fold_dir, "val_predictions.csv")
    val_predictions_df.to_csv(val_pred_path, index=False, float_format="%.17g")

    # 6. Parse history to find best epoch
    history_df = pd.read_csv(history_csv_path)
    best_epoch_number, best_val_loss = compute_best_epoch(history_df)

    fold_metrics = {
        "fold": fold_idx,
        "best_epoch": best_epoch_number,
        "best_val_loss": best_val_loss,
        "total_epochs_trained": len(history_df),
        "val_sample_count": len(val_df),
    }
    save_json(fold_metrics, os.path.join(fold_dir, "fold_metrics.json"))

    return fold_metrics


def train_final_model(
    dev_manifest_df: pd.DataFrame,
    data_dir: str,
    config: Dict[str, Any],
    final_epochs: int,
    run_dir: str,
) -> str:
    """
    Train final model from scratch on all 161 development patients per Section 8.3:
      - Uses all original development images.
      - Uses identical mild online augmentation.
      - Trains for exactly final_epochs (no early stopping, no final-test data).
      - Saves final model to runs/<run_id>/final/model.keras.
    """
    final_dir = os.path.join(run_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    # Deterministic seeding for the final model (seeds torch/keras too)
    set_seed(config.get("seed", 42) + 9999)

    # Honor use_supplied_derivatives exactly like train_single_fold.
    # The split=="development" filter is mandatory: dev_manifest_df here is the
    # full merged manifest, and dropping it would admit test-patient data (leak).
    use_derivatives = config.get("data", {}).get("use_supplied_derivatives", False)
    if use_derivatives:
        dev_orig_df = dev_manifest_df[
            (dev_manifest_df["split"] == "development")
        ].copy()
    else:
        dev_orig_df = dev_manifest_df[
            (dev_manifest_df["split"] == "development") & (dev_manifest_df["is_original"] == True)
        ].copy()
    print(f"Final training set: {len(dev_orig_df)} images "
          f"({'originals + supplied derivatives' if use_derivatives else 'originals only'})")

    train_ds = create_dataset(dev_orig_df, data_dir, config, split_type="train")

    # Fresh model initialization
    model = build_model(config)
    model = compile_canonical_model(model, config)

    # Monitor training loss (no validation data is used; sealed test is untouched)
    history_csv_path = os.path.join(final_dir, "retrain_history.csv")
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            mode="min",
            patience=config.get("training", {}).get("reduce_lr_patience", 5),
            factor=config.get("training", {}).get("reduce_lr_factor", 0.5),
            min_lr=config.get("training", {}).get("min_lr", 1e-6),
            verbose=1,
        ),
        keras.callbacks.CSVLogger(history_csv_path),
    ]

    print(f"\n--- Retraining Final Model on all 161 Dev Patients for {final_epochs} epochs ---")
    model.fit(
        train_ds,
        epochs=final_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # The locked budget is the selection protocol. Do not replace the exact
    # final epoch with a training-loss checkpoint, which is not an independent
    # validation signal and can change the selected epoch.
    final_model_path = os.path.join(final_dir, "model.keras")
    model.save(final_model_path)
    print(f"Final model (exact locked epoch budget) saved to: {final_model_path}")

    return final_model_path
