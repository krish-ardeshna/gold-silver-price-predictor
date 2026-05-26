import logging
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Fix import path when running this script directly
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.preprocess import FEATURE_COLS, load_and_prepare
from src.train import compute_time_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRANSACTION_COST = 0.0005
MAX_POSITION = 1.0
SHARPE_TARGET = 2.0
ANNUAL_TRADING_DAYS = 252


def _format_date_range(df, start_idx, end_idx):
    if "Date" in df.columns:
        start = df["Date"].iloc[start_idx]
        end = df["Date"].iloc[end_idx]
        if isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
            return f"{start.date()} to {end.date()}"
        return f"{start} to {end}"
    return f"rows {start_idx} to {end_idx}"


def evaluate_one_asset(asset_name):
    try:
        logger.info(f"EVALUATING: {asset_name.upper()}")

        model_path = os.path.join(BASE_DIR, "models", f"{asset_name}_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")

        model = joblib.load(model_path)
        logger.info(f"Model loaded: {model_path}")

        logger.info("Loading and preparing data...")
        file_path = os.path.join(BASE_DIR, "data", f"{asset_name}.csv")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = load_and_prepare(file_path)

        split = compute_time_split(len(df))
        df_test = df.iloc[split.test_start:].copy()

        if len(df_test) < 2:
            raise ValueError("Test dataset needs at least 2 rows to evaluate strategy returns.")

        logger.info(f"Test set size: {len(df_test)} rows")

        X_test = df_test[FEATURE_COLS].copy()
        y_test = df_test["Target"].astype(int).values
        y_train = df.iloc[: split.train_end]["Target"].astype(int).values
        baseline_class = int(y_train.mean() >= 0.5)

        classification_threshold = 0.5
        model_label = "Unknown"
        info_path = os.path.join(BASE_DIR, "models", f"{asset_name}_info.json")
        if os.path.exists(info_path):
            try:
                with open(info_path) as f:
                    info = json.load(f)
                classification_threshold = float(info.get("classification_threshold", 0.5))
                model_label = info.get("model_label", model_label)
            except Exception:
                logger.warning("Could not read saved model info. Using default threshold.")

        logger.info("Generating predictions...")
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= classification_threshold).astype(int)

        logger.info(f"Predictions: {len(y_pred)} samples")

        logger.info("Computing classification metrics...")

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        majority_class = int(y_test.mean() >= 0.5)
        baseline_preds = np.full(y_test.shape, majority_class)
        baseline_accuracy = accuracy_score(y_test, baseline_preds)

        logger.info(f"Accuracy: {accuracy*100:.2f}%")
        logger.info(f"Baseline Accuracy: {baseline_accuracy*100:.2f}%")

        logger.info("Simulating trading strategy...")

        actual_prices = df_test["Close"].values
        asset_returns = pd.Series(actual_prices).pct_change().dropna().values
        n_ret = min(len(asset_returns), len(y_proba))

        positions = (y_proba[:n_ret] - 0.5) * 2
        positions = np.clip(positions, -MAX_POSITION, MAX_POSITION)

        strategy_returns = asset_returns[:n_ret] * positions
        turnover = np.abs(np.diff(positions, prepend=0))
        strategy_returns -= turnover * TRANSACTION_COST

        bh_returns = asset_returns[:n_ret]

        total_strategy = float((1 + strategy_returns).prod() - 1)
        total_bh = float((1 + bh_returns).prod() - 1)
        alpha = total_strategy - total_bh

        logger.info(f"Strategy Return: {total_strategy*100:+.2f}%")
        logger.info(f"Buy & Hold: {total_bh*100:+.2f}%")
        logger.info(f"Alpha: {alpha*100:+.2f}%")

        mean_ret = strategy_returns.mean() if len(strategy_returns) > 0 else 0.0
        std_ret = (strategy_returns.std() if len(strategy_returns) > 0 else 0.0) + 1e-6
        sharpe = (mean_ret / std_ret) * np.sqrt(ANNUAL_TRADING_DAYS)
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")

        active_days = np.sum(np.abs(positions[:n_ret]) > 0.1)
        active_positions = positions[:n_ret][np.abs(positions[:n_ret]) > 0.1]
        active_returns = strategy_returns[np.abs(positions[:n_ret]) > 0.1]
        winning_days = np.sum(active_returns > 0)
        win_rate = winning_days / (len(active_returns) + 1e-6) if len(active_returns) > 0 else 0.0

        cumulative = (1 + strategy_returns).cumprod()
        rolling_max = pd.Series(cumulative).cummax()
        drawdown = (cumulative - rolling_max) / (rolling_max + 1e-6)
        max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0

        logger.info(f"Win Rate: {win_rate*100:.2f}%")
        logger.info(f"Max Drawdown: {max_drawdown*100:.2f}%")

        results = {
            "accuracy": float(accuracy),
            "baseline_accuracy": float(baseline_accuracy),
            "baseline_class": int(baseline_class),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "strategy_return": float(total_strategy),
            "buy_hold_return": float(total_bh),
            "alpha": float(alpha),
            "sharpe": float(sharpe),
            "win_rate": float(win_rate),
            "max_drawdown": float(max_drawdown),
            "trades_taken": int(active_days),
            "active_days": int(active_days),
            "test_rows": int(len(df_test)),
            "classification_threshold": float(classification_threshold),
            "model_label": model_label,
            "positioning_mode": "directional_long_short",
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "train_date_range": _format_date_range(df, 0, split.train_end - 1),
            "test_date_range": _format_date_range(df, split.test_start, len(df) - 1),
        }

        reports_dir = os.path.join(BASE_DIR, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        out_path = os.path.join(reports_dir, f"{asset_name}_evaluation.json")

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved: {out_path}")

        return results

    except FileNotFoundError as e:
        logger.error(f"File error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Evaluation failed for {asset_name.upper()}: {str(e)}")
        raise

def main():
    logger.info("MODEL EVALUATION PIPELINE")

    try:
        results = {}

        results["gold"] = evaluate_one_asset("gold")
        results["silver"] = evaluate_one_asset("silver")

        logger.info("EVALUATION SUMMARY")

        for asset, metrics in results.items():
            logger.info(f"\n{asset.upper()}:")
            logger.info(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
            logger.info(f"  Sharpe:   {metrics['sharpe']:.2f}")
            logger.info(f"  Alpha:    {metrics['alpha']*100:+.2f}%")

        logger.info("Evaluation complete!")

        return 0

    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())