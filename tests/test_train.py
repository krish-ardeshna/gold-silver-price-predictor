import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.train as train_module
from src.preprocess import FEATURE_COLS
from src.splitting import compute_time_split


class TestTrain:
    def _prepare_workspace(self, workspace_tmp_path, monkeypatch, asset_name="gold"):
        data_dir = workspace_tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / f"{asset_name}.csv").write_text("placeholder")
        monkeypatch.setattr(train_module, "BASE_DIR", str(workspace_tmp_path))
        return data_dir

    @patch("src.train.joblib.dump")
    @patch("src.train.select_best_model")
    @patch("src.train.load_and_prepare")
    def test_train_one_asset_success(
        self,
        mock_load_and_prepare,
        mock_select_best_model,
        mock_joblib_dump,
        workspace_tmp_path,
        monkeypatch,
        make_feature_frame,
    ):
        self._prepare_workspace(workspace_tmp_path, monkeypatch)
        frame = make_feature_frame(rows=120)
        mock_load_and_prepare.return_value = frame

        model = MagicMock()
        model.best_iteration = 12
        model.predict_proba.return_value = np.tile(
            np.array([[0.45, 0.55]]),
            (compute_time_split(len(frame)).test_rows, 1),
        )
        model.get_params.return_value = {"model_type": "mock"}
        selected_result = MagicMock(
            model_name="logistic_baseline",
            model_label="Logistic Regression",
            model=model,
            threshold=0.55,
            validation_accuracy=0.6,
            validation_balanced_accuracy=0.61,
            validation_f1=0.58,
            best_iteration=12,
        )
        mock_select_best_model.return_value = (selected_result, [selected_result])

        accuracy = train_module.train_one_asset("gold")

        assert 0.0 <= accuracy <= 1.0
        mock_joblib_dump.assert_called_once()

        info = json.loads((workspace_tmp_path / "models" / "gold_info.json").read_text())
        assert info["train_rows"] == compute_time_split(len(frame)).train_rows
        assert info["val_rows"] == compute_time_split(len(frame)).val_rows
        assert info["test_rows"] == compute_time_split(len(frame)).test_rows
        assert "baseline_accuracy" in info
        assert "model_params" in info
        assert info["model_name"] == "logistic_baseline"
        assert info["classification_threshold"] == 0.55

    @patch("src.train.load_and_prepare")
    def test_train_one_asset_requires_existing_data_file(
        self,
        mock_load_and_prepare,
        workspace_tmp_path,
        monkeypatch,
    ):
        monkeypatch.setattr(train_module, "BASE_DIR", str(workspace_tmp_path))

        with pytest.raises(FileNotFoundError):
            train_module.train_one_asset("gold")

        mock_load_and_prepare.assert_not_called()

    @patch("src.train.load_and_prepare")
    def test_train_one_asset_rejects_single_class_target(
        self,
        mock_load_and_prepare,
        workspace_tmp_path,
        monkeypatch,
        make_feature_frame,
    ):
        self._prepare_workspace(workspace_tmp_path, monkeypatch)
        frame = make_feature_frame(rows=120, target_pattern=[1] * 120)
        mock_load_and_prepare.return_value = frame

        with pytest.raises(ValueError, match="Target has no positive or negative samples"):
            train_module.train_one_asset("gold")

    @patch("src.train.joblib.dump")
    @patch("src.train.select_best_model")
    @patch("src.train.load_and_prepare")
    def test_train_one_asset_uses_expected_feature_columns(
        self,
        mock_load_and_prepare,
        mock_select_best_model,
        mock_joblib_dump,
        workspace_tmp_path,
        monkeypatch,
        make_feature_frame,
    ):
        self._prepare_workspace(workspace_tmp_path, monkeypatch)
        frame = make_feature_frame(rows=120)
        mock_load_and_prepare.return_value = frame

        model = MagicMock()
        model.best_iteration = 5
        model.predict_proba.return_value = np.tile(
            np.array([[0.4, 0.6]]),
            (compute_time_split(len(frame)).test_rows, 1),
        )
        model.get_params.return_value = {"model_type": "mock"}
        selected_result = MagicMock(
            model_name="xgboost_regularized",
            model_label="XGBoost Regularized",
            model=model,
            threshold=0.6,
            validation_accuracy=0.55,
            validation_balanced_accuracy=0.56,
            validation_f1=0.57,
            best_iteration=5,
        )
        mock_select_best_model.return_value = (selected_result, [selected_result])

        train_module.train_one_asset("gold")

        call_kwargs = mock_select_best_model.call_args.kwargs
        X_train = call_kwargs["X_train"]
        assert list(X_train.columns) == FEATURE_COLS
        assert call_kwargs["scale_pos_weight"] > 0

    @patch("src.train.train_one_asset", side_effect=[0.55, 0.52])
    def test_main_returns_zero_on_success(self, mock_train):
        assert train_module.main() == 0
        assert mock_train.call_count == 2

    @patch("src.train.train_one_asset", side_effect=RuntimeError("boom"))
    def test_main_returns_one_on_failure(self, mock_train):
        assert train_module.main() == 1
        assert mock_train.call_count == 1
