from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from xgboost import XGBClassifier


THRESHOLD_GRID = np.arange(0.35, 0.66, 0.01)


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
        if isinstance(model, XGBClassifier):
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
