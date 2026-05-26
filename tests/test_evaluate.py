import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.evaluate as evaluate_module
from src.train import compute_time_split


class TestEvaluate:
    def _prepare_workspace(self, workspace_tmp_path, monkeypatch, asset_name="gold", info=None):
        data_dir = workspace_tmp_path / "data"
        models_dir = workspace_tmp_path / "models"
        data_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / f"{asset_name}.csv").write_text("placeholder")
        (models_dir / f"{asset_name}_model.pkl").write_bytes(b"model")
        if info is not None:
            (models_dir / f"{asset_name}_info.json").write_text(json.dumps(info))
        monkeypatch.setattr(evaluate_module, "BASE_DIR", str(workspace_tmp_path))

    @patch("src.evaluate.joblib.load")
    @patch("src.evaluate.load_and_prepare")
    def test_evaluate_one_asset_success(
        self,
        mock_load_and_prepare,
        mock_joblib_load,
        workspace_tmp_path,
        monkeypatch,
        make_feature_frame,
    ):
        self._prepare_workspace(workspace_tmp_path, monkeypatch)
        frame = make_feature_frame(rows=120)
        mock_load_and_prepare.return_value = frame

        split = compute_time_split(len(frame))
        test_rows = split.test_rows

        model = MagicMock()
        model.predict_proba.return_value = np.tile(np.array([[0.45, 0.55]]), (test_rows, 1))
        mock_joblib_load.return_value = model

        results = evaluate_module.evaluate_one_asset("gold")

        assert results["test_rows"] == test_rows
        assert "baseline_accuracy" in results
        assert "confusion_matrix" in results
        assert "train_date_range" in results
        assert "test_date_range" in results
        assert results["classification_threshold"] == 0.5
        saved = json.loads((workspace_tmp_path / "reports" / "gold_evaluation.json").read_text())
        assert saved["positioning_mode"] == "directional_long_short"

    @patch("src.evaluate.load_and_prepare")
    def test_evaluate_one_asset_requires_existing_model(self, mock_load_and_prepare, workspace_tmp_path, monkeypatch):
        monkeypatch.setattr(evaluate_module, "BASE_DIR", str(workspace_tmp_path))
        (workspace_tmp_path / "data").mkdir(parents=True, exist_ok=True)
        (workspace_tmp_path / "data" / "gold.csv").write_text("placeholder")

        with pytest.raises(FileNotFoundError):
            evaluate_module.evaluate_one_asset("gold")

        mock_load_and_prepare.assert_not_called()

    @patch("src.evaluate.joblib.load")
    @patch("src.evaluate.load_and_prepare")
    def test_evaluate_one_asset_requires_at_least_two_test_rows(
        self,
        mock_load_and_prepare,
        mock_joblib_load,
        workspace_tmp_path,
        monkeypatch,
        make_feature_frame,
    ):
        self._prepare_workspace(workspace_tmp_path, monkeypatch)
        frame = make_feature_frame(rows=10)
        mock_load_and_prepare.return_value = frame
        mock_joblib_load.return_value = MagicMock()

        with pytest.raises(ValueError, match="at least 2 rows"):
            evaluate_module.evaluate_one_asset("gold")

    @patch("src.evaluate.joblib.load")
    @patch("src.evaluate.load_and_prepare")
    def test_evaluate_uses_train_split_for_majority_baseline(
        self,
        mock_load_and_prepare,
        mock_joblib_load,
        workspace_tmp_path,
        monkeypatch,
        make_feature_frame,
    ):
        self._prepare_workspace(workspace_tmp_path, monkeypatch)
        target_pattern = [0] * 90 + [1] * 30
        frame = make_feature_frame(rows=120, target_pattern=target_pattern)
        mock_load_and_prepare.return_value = frame

        split = compute_time_split(len(frame))
        test_rows = split.test_rows

        model = MagicMock()
        model.predict_proba.return_value = np.tile(np.array([[0.6, 0.4]]), (test_rows, 1))
        mock_joblib_load.return_value = model

        results = evaluate_module.evaluate_one_asset("gold")

        assert results["baseline_class"] == 0

    @patch("src.evaluate.joblib.load")
    @patch("src.evaluate.load_and_prepare")
    def test_strategy_metrics_are_internally_consistent(
        self,
        mock_load_and_prepare,
        mock_joblib_load,
        workspace_tmp_path,
        monkeypatch,
        make_feature_frame,
    ):
        self._prepare_workspace(workspace_tmp_path, monkeypatch)
        frame = make_feature_frame(rows=120)
        mock_load_and_prepare.return_value = frame

        split = compute_time_split(len(frame))
        test_rows = split.test_rows

        model = MagicMock()
        model.predict_proba.return_value = np.tile(np.array([[0.35, 0.65]]), (test_rows, 1))
        mock_joblib_load.return_value = model

        results = evaluate_module.evaluate_one_asset("gold")

        assert results["active_days"] == test_rows - 1
        assert results["trades_taken"] > 0
        assert 0.0 <= results["win_rate"] <= 1.0

    @patch("src.evaluate.joblib.load")
    @patch("src.evaluate.load_and_prepare")
    def test_evaluate_uses_saved_classification_threshold(
        self,
        mock_load_and_prepare,
        mock_joblib_load,
        workspace_tmp_path,
        monkeypatch,
        make_feature_frame,
    ):
        self._prepare_workspace(
            workspace_tmp_path,
            monkeypatch,
            info={"classification_threshold": 0.6, "model_label": "Logistic Regression"},
        )
        frame = make_feature_frame(rows=120)
        mock_load_and_prepare.return_value = frame

        split = compute_time_split(len(frame))
        test_rows = split.test_rows

        model = MagicMock()
        probabilities = np.tile(np.array([[0.45, 0.55]]), (test_rows, 1))
        model.predict_proba.return_value = probabilities
        mock_joblib_load.return_value = model

        results = evaluate_module.evaluate_one_asset("gold")

        assert results["classification_threshold"] == 0.6
        assert results["model_label"] == "Logistic Regression"

    @patch("src.evaluate.evaluate_one_asset", side_effect=[{"accuracy": 0.5, "sharpe": 0.1, "alpha": 0.0}, {"accuracy": 0.51, "sharpe": 0.2, "alpha": 0.01}])
    def test_main_returns_zero_on_success(self, mock_evaluate):
        assert evaluate_module.main() == 0
        assert mock_evaluate.call_count == 2

    @patch("src.evaluate.evaluate_one_asset", side_effect=RuntimeError("boom"))
    def test_main_returns_one_on_failure(self, mock_evaluate):
        assert evaluate_module.main() == 1
        assert mock_evaluate.call_count == 1
