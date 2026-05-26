import json
import logging
import os
import sys

import joblib
import numpy as np
from sklearn.metrics import accuracy_score

# Fix import path when running this script directly
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.model_selection import select_best_model
from src.preprocess import FEATURE_COLS, load_and_prepare

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
N_ESTIMATORS = 2000
MAX_DEPTH = 7
LEARNING_RATE = 0.012
SUBSAMPLE = 0.92
COLSAMPLE_BYTREE = 0.92
MIN_CHILD_WEIGHT = 1
REG_ALPHA = 0.02
REG_LAMBDA = 0.2
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 120

def train_one_asset(asset_name):
    try:
        logger.info(f"Training {asset_name.upper()}")

        data_path = os.path.join(BASE_DIR, "data", f"{asset_name}.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = load_and_prepare(data_path)
        logger.info(f"Data loaded: {len(df)} rows × {len(FEATURE_COLS)} features")

        X = df[FEATURE_COLS].copy()
        y = df["Target"].astype(int)

        logger.info("Splitting data (time-based)...")

        train_split = int(len(df) * TRAIN_RATIO)
        val_split = int(len(df) * (TRAIN_RATIO + VAL_RATIO))

        X_train = X.iloc[:train_split]
        y_train = y.iloc[:train_split]

        X_val = X.iloc[train_split:val_split]
        y_val = y.iloc[train_split:val_split]

        X_test = X.iloc[val_split:]
        y_test = y.iloc[val_split:]

        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("Insufficient data after split. Check your data source.")

        logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

        logger.info("Computing class weights...")

        pos = y_train.sum()
        neg = len(y_train) - pos

        if pos == 0 or neg == 0:
            raise ValueError("Target has no positive or negative samples")

        scale_pos_weight = neg / (pos + 1e-6)
        logger.info(f"Class weights - Positive: 1.0, Negative: {scale_pos_weight:.2f}")

        logger.info("Selecting best model candidate...")
        best_candidate, _ = select_best_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            scale_pos_weight=scale_pos_weight,
        )

        model = best_candidate.model
        threshold = best_candidate.threshold
        logger.info(f"Selected model: {best_candidate.model_name} ({best_candidate.model_label})")

        logger.info("Evaluating on test set...")
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
            preds = (proba >= threshold).astype(int)
        else:
            preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        majority_class = int(y_test.mean() >= 0.5)
        baseline_preds = np.full(len(y_test), majority_class)
        baseline_accuracy = accuracy_score(y_test, baseline_preds)

        logger.info(f"Test Accuracy: {acc*100:.2f}%")
        logger.info(f"Baseline Accuracy: {baseline_accuracy*100:.2f}%")

        logger.info("Saving model and metadata...")

        models_dir = os.path.join(BASE_DIR, "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"{asset_name}_model.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Model saved: {model_path}")

        def _sanitize_json(value):
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {k: _sanitize_json(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_sanitize_json(v) for v in value]
            return value

        info = {
            "accuracy": float(acc),
            "baseline_accuracy": float(baseline_accuracy),
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "n_features": int(len(FEATURE_COLS)),
            "features": FEATURE_COLS,
            "scale_pos_weight": float(scale_pos_weight),
            "best_iteration": int(best_candidate.best_iteration) if best_candidate.best_iteration is not None else None,
            "model_name": best_candidate.model_name,
            "model_label": best_candidate.model_label,
            "classification_threshold": float(threshold),
            "model_params": _sanitize_json(model.get_params()) if hasattr(model, "get_params") else {},
        }

        info_path = os.path.join(models_dir, f"{asset_name}_info.json")
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        logger.info(f"Metadata saved: {info_path}")

        return acc

    except FileNotFoundError as e:
        logger.error(f"File error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Training failed for {asset_name.upper()}: {str(e)}")
        raise

def main():
    logger.info("MODEL TRAINING PIPELINE")

    try:
        accuracies = {}

        gold_acc = train_one_asset("gold")
        accuracies["gold"] = gold_acc

        silver_acc = train_one_asset("silver")
        accuracies["silver"] = silver_acc

        logger.info("TRAINING SUMMARY")
        for asset, acc in accuracies.items():
            logger.info(f"{asset.upper()} accuracy: {acc*100:.2f}%")
        logger.info("Training complete!")

        return 0

    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())