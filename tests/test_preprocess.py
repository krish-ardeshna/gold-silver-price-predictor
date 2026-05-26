import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    BREAKOUT_PERIOD,
    FEATURE_COLS,
    MA_SHORT,
    TARGET_THRESHOLD,
    compute_macd,
    compute_rsi,
    load_and_prepare,
)


class TestPreprocess:
    def test_compute_rsi_returns_bounded_values(self):
        series = pd.Series([100, 102, 101, 104, 105, 103, 108, 109, 111, 110, 113, 114, 116, 118, 117, 119])

        rsi = compute_rsi(series, period=5)

        assert len(rsi) == len(series)
        assert ((rsi.dropna() >= 0) & (rsi.dropna() <= 100)).all()

    def test_compute_macd_histogram_matches_difference(self):
        series = pd.Series(np.linspace(100, 120, 40))

        macd_line, signal_line, macd_hist = compute_macd(series)

        np.testing.assert_allclose(macd_hist, macd_line - signal_line)

    def test_load_and_prepare_builds_expected_features(self, workspace_tmp_path, make_price_frame):
        raw = make_price_frame(rows=120)
        file_path = workspace_tmp_path / "gold.csv"
        raw.to_csv(file_path, index=False)

        prepared = load_and_prepare(file_path)

        assert not prepared.empty
        assert "Target" in prepared.columns
        assert set(FEATURE_COLS).issubset(prepared.columns)
        assert not prepared[FEATURE_COLS + ["Target"]].isna().any().any()

    def test_load_and_prepare_sorts_dates(self, workspace_tmp_path, make_price_frame):
        raw = make_price_frame(rows=120).iloc[::-1]
        file_path = workspace_tmp_path / "gold.csv"
        raw.to_csv(file_path, index=False)

        prepared = load_and_prepare(file_path)

        assert prepared["Date"].is_monotonic_increasing

    def test_load_and_prepare_matches_shifted_ma_feature(self, workspace_tmp_path, make_price_frame):
        raw = make_price_frame(rows=120)
        raw["Expected_MA7"] = raw["Close"].rolling(MA_SHORT).mean().shift(1)
        file_path = workspace_tmp_path / "gold.csv"
        raw.drop(columns=["Expected_MA7"]).to_csv(file_path, index=False)

        prepared = load_and_prepare(file_path)
        merged = prepared.merge(raw[["Date", "Expected_MA7"]], on="Date", how="left")

        np.testing.assert_allclose(merged["MA7"], merged["Expected_MA7"])

    def test_load_and_prepare_matches_target_definition(self, workspace_tmp_path, make_price_frame):
        raw = make_price_frame(rows=120)
        raw["Expected_Target"] = (raw["Close"].pct_change(fill_method=None).shift(-1) > TARGET_THRESHOLD).astype(int)
        file_path = workspace_tmp_path / "gold.csv"
        raw.drop(columns=["Expected_Target"]).to_csv(file_path, index=False)

        prepared = load_and_prepare(file_path)
        merged = prepared.merge(raw[["Date", "Expected_Target"]], on="Date", how="left")

        assert merged["Target"].tolist() == merged["Expected_Target"].tolist()

    def test_load_and_prepare_handles_missing_close_column(self, workspace_tmp_path, make_price_frame):
        raw = make_price_frame(rows=120).drop(columns=["Close"])
        file_path = workspace_tmp_path / "gold.csv"
        raw.to_csv(file_path, index=False)

        with pytest.raises(ValueError, match="Close"):
            load_and_prepare(file_path)

    def test_load_and_prepare_handles_invalid_close_values(self, workspace_tmp_path, make_price_frame):
        raw = make_price_frame(rows=120)
        raw["Close"] = "bad"
        file_path = workspace_tmp_path / "gold.csv"
        raw.to_csv(file_path, index=False)

        with pytest.raises(ValueError, match="invalid"):
            load_and_prepare(file_path)

    def test_load_and_prepare_requires_enough_rows(self, workspace_tmp_path, make_price_frame):
        raw = make_price_frame(rows=20)
        file_path = workspace_tmp_path / "gold.csv"
        raw.to_csv(file_path, index=False)

        with pytest.raises(ValueError, match="Need at least 50 rows"):
            load_and_prepare(file_path)

    def test_feature_columns_list_stays_in_sync(self):
        expected_features = {
            "MA7",
            "MA30",
            "MA_Ratio",
            "Lag1_Return",
            "Lag2_Return",
            "Lag3_Return",
            "Lag1_PriceRatio",
            "Lag2_PriceRatio",
            "Lag3_PriceRatio",
            "Volatility7",
            "Volatility14",
            "Momentum_3",
            "Momentum_7",
            "Trend_Strength",
            "Vol_Ratio",
            "RSI",
            "Zscore_7",
            "Breakout_7",
            "MACD",
            "MACD_Signal",
            "MACD_Hist",
            "BB_Position",
            "ADX",
        }

        assert set(FEATURE_COLS) == expected_features
