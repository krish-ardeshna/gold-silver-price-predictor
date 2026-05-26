import logging
import pandas as pd
import numpy as np
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MA_SHORT = 5
MA_LONG = 25
RSI_PERIOD = 14
VOLATILITY_SHORT = 7
VOLATILITY_LONG = 14
MOMENTUM_SHORT = 3
MOMENTUM_LONG = 7
ZSCORE_PERIOD = 7
BREAKOUT_PERIOD = 7
TARGET_THRESHOLD = 0.0008

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(period).mean()
    avg_loss = down.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def load_and_prepare(file_path):
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows")

        if len(df) < 50:
            raise ValueError(f"Need at least 50 rows, got {len(df)}")

        df.columns = df.columns.str.strip()
        if "Close" not in df.columns:
            raise ValueError("CSV must contain 'Close' column")

        df["Close"] = df["Close"].astype(str).str.replace(",", "", regex=False)
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

        if df["Close"].isna().all():
            raise ValueError("All Close values are invalid")

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)

        logger.info("Creating features...")
        df["Return"] = df["Close"].pct_change(fill_method=None)

        df["MA7"] = df["Close"].rolling(MA_SHORT).mean().shift(1)
        df["MA30"] = df["Close"].rolling(MA_LONG).mean().shift(1)
        df["MA_Ratio"] = df["MA7"] / (df["MA30"] + 1e-6)

        df["Lag1_Return"] = df["Return"].shift(1)
        df["Lag2_Return"] = df["Return"].shift(2)
        df["Lag3_Return"] = df["Return"].shift(3)

        df["Lag1_PriceRatio"] = df["Close"].shift(1) / (df["Close"].shift(2) + 1e-6)
        df["Lag2_PriceRatio"] = df["Close"].shift(1) / (df["Close"].shift(3) + 1e-6)
        df["Lag3_PriceRatio"] = df["Close"].shift(1) / (df["Close"].shift(4) + 1e-6)

        df["Volatility7"] = df["Return"].rolling(VOLATILITY_SHORT).std().shift(1)
        df["Volatility14"] = df["Return"].rolling(VOLATILITY_LONG).std().shift(1)

        df["Momentum_3"] = df["Close"].shift(1) / (df["Close"].shift(4) + 1e-6)
        df["Momentum_7"] = df["Close"].shift(1) / (df["Close"].shift(8) + 1e-6)
        df["Trend_Strength"] = df["MA7"] - df["MA30"]
        df["Vol_Ratio"] = (df["Volatility7"] + 1e-6) / (df["Volatility14"] + 1e-6)
        df["RSI"] = compute_rsi(df["Close"], period=RSI_PERIOD).shift(1)

        ma = df["Close"].rolling(ZSCORE_PERIOD).mean()
        std = df["Close"].rolling(ZSCORE_PERIOD).std()
        df["Zscore_7"] = ((df["Close"] - ma) / (std + 1e-6)).shift(1)

        df["Breakout_7"] = (df["Close"] > df["Close"].rolling(BREAKOUT_PERIOD).max()).astype(int).shift(1)

        macd_line, signal_line, macd_hist = compute_macd(df["Close"])
        df["MACD"] = macd_line.shift(1)
        df["MACD_Signal"] = signal_line.shift(1)
        df["MACD_Hist"] = macd_hist.shift(1)

        bb_sma = df["Close"].rolling(20).mean().shift(1)
        bb_std = df["Close"].rolling(20).std().shift(1)
        df["BB_Upper"] = bb_sma + 2 * bb_std
        df["BB_Lower"] = bb_sma - 2 * bb_std
        # Use prior-day bands, then shift one day so BB_Position does not leak future values.
        df["BB_Position"] = ((df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-6)).shift(1)

        df["ADX"] = compute_rsi(df["Close"].diff().abs(), period=14).shift(1)  # proxy indicator, not true ADX

        future_return = df["Close"].pct_change(fill_method=None).shift(-1)
        df["Target"] = (future_return > TARGET_THRESHOLD).astype(int)

        logger.info("Removing NaN values...")
        initial_rows = len(df)
        df = df.dropna().reset_index(drop=True)
        removed_rows = initial_rows - len(df)
        logger.info(f"Removed {removed_rows} rows, kept {len(df)} rows")

        if len(df) == 0:
            raise ValueError("No valid data after removing NaN values")

        ups = df["Target"].sum()
        downs = len(df) - ups
        logger.info(f"Target: {ups} ups, {downs} downs")

        logger.info(f"Data ready: {len(df)} rows × {len(df.columns)} columns")
        return df

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except ValueError as e:
        logger.error(f"Data error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

FEATURE_COLS = [
    "MA7", "MA30", "MA_Ratio",
    "Lag1_Return", "Lag2_Return", "Lag3_Return",
    "Lag1_PriceRatio", "Lag2_PriceRatio", "Lag3_PriceRatio",
    "Volatility7", "Volatility14",
    "Momentum_3", "Momentum_7",
    "Trend_Strength", "Vol_Ratio",
    "RSI", "Zscore_7", "Breakout_7",
    "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Position", "ADX",
]