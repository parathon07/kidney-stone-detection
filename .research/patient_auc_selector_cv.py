"""Research-only fold trainer with patient-level validation-AUC selection.

This intentionally leaves the canonical trainer unchanged. It is used for the
E019 ablation that changes only checkpoint/early-stop selection from slice
validation loss to the primary patient-level validation ROC-AUC.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data import create_dataset
from src.model import Fp16BatchNorm, build_model
from src.splits import load_and_validate_splits
from src.trainer import compile_canonical_model
from src.utils import load_config, set_seed


class PatientAUCCheckpoint(keras.callbacks.Callback):
    def __init__(self, val_ds, val_df, checkpoint_path, patience, min_delta):
        super().__init__()
        self.val_ds = val_ds
        self.val_df = val_df.reset_index(drop=True)
        self.checkpoint_path = str(checkpoint_path)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_auc = -np.inf
        self.best_epoch = None
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        raw = self.model.predict(self.val_ds, verbose=0)
        frame = pd.DataFrame(
            {
                "patient_id": self.val_df["patient_id"].tolist(),
                "y_true": self.val_df["label"].astype(int).tolist(),
                "stone_probability": raw[:, 1].astype(float),
            }
        )
        patient = frame.groupby("patient_id", as_index=False).agg(
            y_true=("y_true", "first"),
            stone_probability=("stone_probability", "mean"),
        )
        auc = float(roc_auc_score(patient["y_true"], patient["stone_probability"]))
        logs["val_patient_auc"] = auc
        if auc > self.best_auc + self.min_delta:
            self.best_auc = auc
            self.best_epoch = int(epoch + 1)
            self.wait = 0
            self.model.save(self.checkpoint_path)
        else:
            self.wait += 1
        print(
            f"PATIENT_AUC epoch={epoch + 1} value={auc:.6f} "
            f"best={self.best_auc:.6f} wait={self.wait}/{self.patience}",
            flush=True,
        )
        if self.wait >= self.patience:
            self.model.stop_training = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    if args.fold not in range(5):
        raise ValueError("fold must be in 0..4")

    run_dir = Path(args.run_dir)
    fold_dir = run_dir / f"fold_{args.fold}"
    if fold_dir.exists() and any(fold_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty fold directory: {fold_dir}")
    fold_dir.mkdir(parents=True, exist_ok=True)

    config = load_config("configs/default.yaml")
    config["seed"] = int(args.seed)
    manifest = pd.read_csv("data/manifest.csv")
    merged = load_and_validate_splits(manifest, splits_dir="splits")
    dev = merged[merged["split"] == "development"].copy()
    train_df = dev[
        (dev["fold"] != args.fold) & (dev["is_original"] == True)
    ].copy().reset_index(drop=True)
    val_df = dev[
        (dev["fold"] == args.fold) & (dev["is_original"] == True)
    ].copy().reset_index(drop=True)
    data_dir = "D:/Documents/Desktop/kidney stone detection/data/dataset/Kindey Stone Dataset"

    set_seed(args.seed + args.fold)
    train_ds = create_dataset(train_df, data_dir, config, split_type="train")
    val_ds = create_dataset(val_df, data_dir, config, split_type="val")
    model = compile_canonical_model(build_model(config), config)
    checkpoint_path = fold_dir / "best.keras"
    selector = PatientAUCCheckpoint(
        val_ds,
        val_df,
        checkpoint_path,
        config["training"].get("early_stopping_patience", 15),
        config["training"].get("early_stopping_min_delta", 0.002),
    )
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            patience=config["training"].get("reduce_lr_patience", 5),
            factor=config["training"].get("reduce_lr_factor", 0.5),
            min_lr=config["training"].get("min_lr", 1e-6),
            verbose=1,
        ),
        selector,
        keras.callbacks.CSVLogger(str(fold_dir / "history.csv")),
    ]
    print(
        f"TRAINING PATIENT-AUC FOLD {args.fold} "
        f"(Train samples: {len(train_df)}, Val samples: {len(val_df)})",
        flush=True,
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["training"].get("epochs", 60),
        callbacks=callbacks,
        verbose=1,
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Patient-AUC checkpoint was not saved: {checkpoint_path}")

    best_model = keras.models.load_model(
        checkpoint_path,
        compile=False,
        custom_objects={"Fp16BatchNorm": Fp16BatchNorm},
    )
    raw = best_model.predict(val_ds, verbose=0)
    val_predictions = pd.DataFrame(
        {
            "path": val_df["path"].tolist(),
            "patient_id": val_df["patient_id"].tolist(),
            "source_image_id": val_df["source_image_id"].tolist(),
            "y_true": val_df["label"].astype(int).tolist(),
            "stone_probability": raw[:, 1].astype(float),
            "predicted_class_at_0_5": (raw[:, 1] >= 0.5).astype(int),
            "fold": args.fold,
        }
    )
    val_predictions.to_csv(
        fold_dir / "val_predictions.csv", index=False, float_format="%.17g"
    )
    patient = val_predictions.groupby("patient_id", as_index=False).agg(
        y_true=("y_true", "first"),
        stone_probability=("stone_probability", "mean"),
    )
    history = pd.read_csv(fold_dir / "history.csv")
    metrics = {
        "fold": args.fold,
        "selection_metric": "patient_level_val_roc_auc",
        "best_epoch": int(selector.best_epoch),
        "best_patient_val_auc": float(selector.best_auc),
        "final_patient_val_auc_from_checkpoint": float(
            roc_auc_score(patient["y_true"], patient["stone_probability"])
        ),
        "best_val_loss_in_history": float(history["val_loss"].min()),
        "total_epochs_trained": int(len(history)),
        "val_sample_count": int(len(val_df)),
        "sealed_test_used": False,
    }
    with open(fold_dir / "fold_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print("PATIENT_AUC_FOLD_RESULT", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
