import os
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocess import FEATURE_COLS


TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp" / "pytest-temp"
TMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = str(TMP_ROOT)
os.environ["TEMP"] = str(TMP_ROOT)
os.environ["TMPDIR"] = str(TMP_ROOT)
tempfile.tempdir = str(TMP_ROOT)


@pytest.fixture
def workspace_tmp_path():
    path = TMP_ROOT / f"case-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def make_price_frame():
    def _make(rows=120, seed=42, start="2020-01-01"):
        rng = np.random.default_rng(seed)
        dates = pd.date_range(start, periods=rows, freq="D")
        close = 1500 + np.cumsum(rng.normal(0, 8, size=rows))
        return pd.DataFrame(
            {
                "Date": dates,
                "Close": close,
                "Open": close - rng.uniform(0, 5, size=rows),
                "High": close + rng.uniform(0, 5, size=rows),
                "Low": close - rng.uniform(0, 5, size=rows),
                "Volume": rng.integers(1000, 5000, size=rows),
            }
        )

    return _make


@pytest.fixture
def make_feature_frame():
    def _make(rows=120, seed=42, target_pattern=None):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2020-01-01", periods=rows, freq="D")
        close = 1500 + np.cumsum(rng.normal(0, 2, size=rows))

        data = {
            "Date": dates,
            "Close": close,
        }
        for index, feature in enumerate(FEATURE_COLS, start=1):
            data[feature] = rng.normal(loc=index * 0.01, scale=1.0, size=rows)

        if target_pattern is None:
            target_pattern = np.array([0, 1] * (rows // 2) + [0] * (rows % 2))
        else:
            target_pattern = np.array(target_pattern)

        data["Target"] = target_pattern[:rows]
        return pd.DataFrame(data)

    return _make
