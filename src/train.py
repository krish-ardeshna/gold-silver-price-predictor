import json
import logging
import os
import sys
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# Fix import path when running this script directly
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

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

THRESHOLD_GRID = np.arange(0.35, 0.66, 0.01)
DEFAULT_TRAIN_RATIO = TRAIN_RATIO
DEFAULT_VAL_RATIO = VAL_RATIO


@dataclass(frozen=True)
class TimeSplit:
    train_end: int
    val_end: int
    total_rows: int
    train_ratio: float
    val_ratio: float

    @property
    def test_start(self) -> int:
        return self.val_end

    @property
    def test_rows(self) -> int:
        return self.total_rows - self.val_end

    @property
    def train_rows(self) -> int:
        return self.train_end

    @property
    def val_rows(self) -> int:
        return self.val_end - self.train_end


def compute_time_split(
    total_rows: int,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
) -> TimeSplit:
    if total_rows < 3:
        raise ValueError("Need at least 3 rows for train/validation/test splits.")

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must leave room for a test split.")

    train_end = int(total_rows * train_ratio)
    val_end = int(total_rows * (train_ratio + val_ratio))

    if train_end <= 0:
        raise ValueError("Training split is empty.")

    if val_end <= train_end:
        raise ValueError("Validation split is empty.")

    if val_end >= total_rows:
        raise ValueError("Test split is empty.")

    return TimeSplit(
        train_end=train_end,
        val_end=val_end,
        total_rows=total_rows,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )


@dataclass
class CandidateResult:
    model_name: str
    model_label: str
    model: object
    threshold: float
    validation_accuracy: float
    validation_balanced_accuracy: float
    validation_f1: float
    best_iteration: int | None


def _best_threshold(y_true, probabilities):
    best = None
    for threshold in THRESHOLD_GRID:
        predictions = (probabilities >= threshold).astype(int)
        accuracy = accuracy_score(y_true, predictions)
        balanced_accuracy = balanced_accuracy_score(y_true, predictions)
        f1 = f1_score(y_true, predictions, zero_division=0)

        candidate = (
            balanced_accuracy,
            accuracy,
            f1,
            -abs(threshold - 0.5),
            float(threshold),
        )
        if best is None or candidate > best[0]:
            best = (
                candidate,
                {
                    "threshold": float(threshold),
                    "accuracy": float(accuracy),
                    "balanced_accuracy": float(balanced_accuracy),
                    "f1": float(f1),
                },
            )
    return best[1]


def build_candidate_models(scale_pos_weight: float):
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression

    return {
        "logistic_baseline": {
            "label": "Logistic Regression",
            "model": LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=1.0,
            ),
        },
        "xgboost_medium_depth": {
            "label": "XGBoost Medium Depth",
            "model": XGBClassifier(
                n_estimators=800,
                learning_rate=0.03,
                max_depth=4,
                min_child_weight=3,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                early_stopping_rounds=80,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
                n_jobs=-1,
            ),
        },
        "xgboost_current": {
            "label": "XGBoost Current",
            "model": XGBClassifier(
                n_estimators=2000,
                learning_rate=0.012,
                max_depth=7,
                min_child_weight=1,
                subsample=0.92,
                colsample_bytree=0.92,
                reg_alpha=0.02,
                reg_lambda=0.2,
                scale_pos_weight=scale_pos_weight,
                early_stopping_rounds=120,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
                n_jobs=-1,
            ),
        },
    }


def select_best_model(X_train, y_train, X_val, y_val, scale_pos_weight: float):
    results = []
    candidates = build_candidate_models(scale_pos_weight)

    for model_name, candidate in candidates.items():
        model = candidate["model"]
        fit_kwargs = {}
        if hasattr(model, "predict_proba"):
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False

        model.fit(X_train, y_train, **fit_kwargs)
        probabilities = model.predict_proba(X_val)[:, 1]
        threshold_metrics = _best_threshold(y_val, probabilities)
        best_iteration = getattr(model, "best_iteration", None)

        results.append(
            CandidateResult(
                model_name=model_name,
                model_label=candidate["label"],
                model=model,
                threshold=threshold_metrics["threshold"],
                validation_accuracy=threshold_metrics["accuracy"],
                validation_balanced_accuracy=threshold_metrics["balanced_accuracy"],
                validation_f1=threshold_metrics["f1"],
                best_iteration=int(best_iteration) if best_iteration is not None else None,
            )
        )

    results.sort(
        key=lambda result: (
            result.validation_balanced_accuracy,
            result.validation_accuracy,
            result.validation_f1,
        ),
        reverse=True,
    )
    return results[0], results

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

        split = compute_time_split(len(df))
        X_train = X.iloc[: split.train_end]
        y_train = y.iloc[: split.train_end]

        X_val = X.iloc[split.train_end:split.val_end]
        y_val = y.iloc[split.train_end:split.val_end]

        X_test = X.iloc[split.val_end:]
        y_test = y.iloc[split.val_end:]

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