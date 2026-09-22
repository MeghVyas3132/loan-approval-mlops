"""Tests for candidate construction, metrics and the trained artefact."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import load_config
from src.training.train import build_candidates, compute_metrics
from src.transformation.features import MODEL_FEATURES, select_model_features


class TestBuildCandidates:
    def test_builds_every_configured_estimator(self):
        candidates = build_candidates(
            {"logistic_regression": {"C": 0.5}, "random_forest": {"n_estimators": 10},
             "gradient_boosting": {"n_estimators": 10}},
            random_state=42,
        )
        assert isinstance(candidates["logistic_regression"], LogisticRegression)
        assert isinstance(candidates["random_forest"], RandomForestClassifier)
        assert isinstance(candidates["gradient_boosting"], GradientBoostingClassifier)

    def test_hyperparameters_are_applied(self):
        candidates = build_candidates({"logistic_regression": {"C": 0.25}}, random_state=1)
        assert candidates["logistic_regression"].C == 0.25

    def test_random_state_is_propagated(self):
        candidates = build_candidates({"random_forest": {"n_estimators": 5}}, random_state=99)
        assert candidates["random_forest"].random_state == 99

    def test_unknown_model_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported candidate"):
            build_candidates({"deep_neural_net": {}}, random_state=1)

    def test_params_yaml_candidates_are_buildable(self):
        config = load_config()
        candidates = build_candidates(config.section("training")["candidates"], random_state=42)
        assert len(candidates) == 3


class TestComputeMetrics:
    def test_perfect_prediction(self):
        y = np.array([0, 1, 0, 1])
        metrics = compute_metrics(y, y, y.astype(float))
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["roc_auc"] == 1.0

    def test_all_metrics_present(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        metrics = compute_metrics(y_true, y_pred, np.array([0.1, 0.9, 0.4, 0.2, 0.8]))
        assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
        assert all(0.0 <= v <= 1.0 for v in metrics.values())

    def test_roc_auc_omitted_for_single_class_truth(self):
        y = np.array([1, 1, 1])
        assert "roc_auc" not in compute_metrics(y, y, np.array([0.9, 0.8, 0.7]))

    def test_zero_division_is_handled(self):
        metrics = compute_metrics(np.array([0, 0]), np.array([0, 0]))
        assert metrics["precision"] == 0.0


@pytest.mark.integration
class TestTrainedPipeline:
    def test_pipeline_learns_better_than_chance(self, trained_pipeline, encoded_frame):
        X = select_model_features(encoded_frame)
        accuracy = (trained_pipeline.predict(X) == encoded_frame["Loan_Status"]).mean()
        assert accuracy > 0.65

    def test_credit_history_drives_the_decision(self, trained_pipeline, sample_application):
        import pandas as pd

        from src.transformation.features import add_engineered_features

        good = add_engineered_features(pd.DataFrame([sample_application]))[MODEL_FEATURES]
        bad = good.copy()
        bad["Credit_History"] = 0.0
        assert trained_pipeline.predict_proba(good)[0, 1] > trained_pipeline.predict_proba(bad)[0, 1]

    def test_probabilities_are_valid(self, trained_pipeline, encoded_frame):
        proba = trained_pipeline.predict_proba(select_model_features(encoded_frame))
        assert np.all((proba >= 0) & (proba <= 1))
        assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.mark.integration
class TestPipelineArtifacts:
    """The artefacts `dvc repro` produces must stay consistent with each other."""

    def test_metadata_matches_the_feature_contract(self):
        config = load_config()
        metadata_path = config.get_path("training.metadata_path")
        if not metadata_path.exists():
            pytest.skip("run `dvc repro` first")
        metadata = json.loads(metadata_path.read_text())
        assert metadata["feature_columns"] == MODEL_FEATURES

    def test_evaluation_metrics_meet_the_gate(self):
        config = load_config()
        metrics_path = config.get_path("evaluation.metrics_path")
        if not metrics_path.exists():
            pytest.skip("run `dvc repro` first")
        metrics = json.loads(metrics_path.read_text())
        assert metrics["roc_auc"] >= config.get_nested("evaluation.min_roc_auc")
        assert metrics["f1"] >= config.get_nested("evaluation.min_f1")
