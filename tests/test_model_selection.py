from unittest.mock import MagicMock, patch

import numpy as np

from src.model_selection import select_best_model


class TestModelSelection:
    def test_select_best_model_prefers_better_validation_candidate(self):
        X_train = np.arange(20).reshape(10, 2)
        y_train = np.array([0, 1] * 5)
        X_val = np.arange(12).reshape(6, 2)
        y_val = np.array([0, 1, 0, 1, 0, 1])

        weaker = MagicMock()
        weaker.predict_proba.return_value = np.array(
            [[0.45, 0.55], [0.48, 0.52], [0.40, 0.60], [0.35, 0.65], [0.55, 0.45], [0.30, 0.70]]
        )
        weaker.best_iteration = None

        stronger = MagicMock()
        stronger.predict_proba.return_value = np.array(
            [[0.80, 0.20], [0.20, 0.80], [0.75, 0.25], [0.25, 0.75], [0.70, 0.30], [0.30, 0.70]]
        )
        stronger.best_iteration = 7

        with patch(
            "src.model_selection.build_candidate_models",
            return_value={
                "weaker_model": {"label": "Weaker", "model": weaker},
                "stronger_model": {"label": "Stronger", "model": stronger},
            },
        ):
            selected, results = select_best_model(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                scale_pos_weight=1.0,
            )

        assert selected.model_name == "stronger_model"
        assert results[0].validation_balanced_accuracy >= results[1].validation_balanced_accuracy
        assert 0.35 <= selected.threshold <= 0.65

    def test_select_best_model_returns_ranked_candidates(self):
        X_train = np.arange(20).reshape(10, 2)
        y_train = np.array([0, 1] * 5)
        X_val = np.arange(12).reshape(6, 2)
        y_val = np.array([0, 1, 0, 1, 0, 1])

        candidate_a = MagicMock()
        candidate_a.predict_proba.return_value = np.array(
            [[0.45, 0.55], [0.48, 0.52], [0.40, 0.60], [0.35, 0.65], [0.55, 0.45], [0.30, 0.70]]
        )
        candidate_a.best_iteration = None

        candidate_b = MagicMock()
        candidate_b.predict_proba.return_value = np.array(
            [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]]
        )
        candidate_b.best_iteration = None

        with patch(
            "src.model_selection.build_candidate_models",
            return_value={
                "candidate_a": {"label": "Candidate A", "model": candidate_a},
                "candidate_b": {"label": "Candidate B", "model": candidate_b},
            },
        ):
            _, results = select_best_model(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                scale_pos_weight=1.0,
            )

        assert [result.model_name for result in results] == ["candidate_b", "candidate_a"]
