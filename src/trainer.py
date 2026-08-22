import os
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple
from src.model import build_model
from src.data import create_dataset
from src.utils import save_json


def compile_canonical_model(model: tf.keras.Model, config: Dict[str, Any]) -> tf.keras.Model:
    """
    Compile canonical model with AdamW, SparseCategoricalCrossentropy, and minimal metrics.
    """
    train_cfg = config.get("training", {})
    lr = train_cfg.get("learning_rate", 3e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-4)

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr,
        weight_decay=weight_decay,
    )

    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


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

    # 1. Manifests (Training uses originals + online aug; Val uses originals only)
    train_df = dev_manifest_df[dev_manifest_df["fold"] != fold_idx].copy()
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
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_checkpoint_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.get("training", {}).get("early_stopping_patience", 8),
            restore_best_weights=False,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            patience=config.get("training", {}).get("reduce_lr_patience", 3),
            factor=config.get("training", {}).get("reduce_lr_factor", 0.5),
            min_lr=config.get("training", {}).get("min_lr", 1e-6),
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(history_csv_path),
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
    
    best_model = tf.keras.models.load_model(best_checkpoint_path)

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
    val_predictions_df.to_csv(val_pred_path, index=False)

    # 6. Parse history to find best epoch
    history_df = pd.read_csv(history_csv_path)
    best_epoch_idx = int(history_df["val_loss"].idxmin())
    best_epoch_number = int(history_df.loc[best_epoch_idx, "epoch"]) if "epoch" in history_df.columns else best_epoch_idx + 1
    best_val_loss = float(history_df.loc[best_epoch_idx, "val_loss"])

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

    # Filter all original development images
    dev_orig_df = dev_manifest_df[
        (dev_manifest_df["split"] == "development") & (dev_manifest_df["is_original"] == True)
    ].copy()

    train_ds = create_dataset(dev_orig_df, data_dir, config, split_type="train")

    # Fresh model initialization
    model = build_model(config)
    model = compile_canonical_model(model, config)

    history_csv_path = os.path.join(final_dir, "retrain_history.csv")
    callbacks = [tf.keras.callbacks.CSVLogger(history_csv_path)]

    print(f"\n--- Retraining Final Model on all 161 Dev Patients for {final_epochs} epochs ---")
    model.fit(
        train_ds,
        epochs=final_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    final_model_path = os.path.join(final_dir, "model.keras")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    return final_model_path
