"""Integration tests: run every DVC stage against a temporary workspace.

The stages are driven through their Python entry points with a generated
``params.yaml`` whose paths point inside ``tmp_path``, so the test exercises
the real code the pipeline runs without touching the repository's artefacts.
"""

from __future__ import annotations

import json

import joblib
import pytest
import yaml

from src.config import load_config
from src.ingestion import generate_data
from src.ingestion.ingest import ingest
from src.training.evaluate import ModelQualityError, evaluate_stage
from src.training.train import train_stage
from src.transformation.transform import transform_stage
from src.validation.validate import ValidationError, validate_stage

pytestmark = pytest.mark.integration


def _write_params(tmp_path, **overrides) -> object:
    """Build a small, fully absolute params file inside ``tmp_path``."""
    params = {
        "base": {"project_name": "test-run", "random_state": 42,
                 "target_column": "Loan_Status", "positive_label": 1},
        "data": {
            "raw_path": str(tmp_path / "raw.csv"),
            "n_samples": 400,
            "missing_rate": 0.04,
            "train_path": str(tmp_path / "train.csv"),
            "test_path": str(tmp_path / "test.csv"),
            "test_size": 0.25,
            "stratify": True,
        },
        "validation": {
            "report_path": str(tmp_path / "validation_report.json"),
            "max_missing_fraction": 0.15,
            "min_rows": 100,
            "expected_target_balance": [0.2, 0.9],
        },
        "transformation": {
            "preprocessor_path": str(tmp_path / "preprocessor.joblib"),
            "train_features_path": str(tmp_path / "train_features.csv"),
            "test_features_path": str(tmp_path / "test_features.csv"),
        },
        "training": {
            "model_path": str(tmp_path / "model.joblib"),
            "metadata_path": str(tmp_path / "model_metadata.json"),
            "metrics_path": str(tmp_path / "train_metrics.json"),
            "cv_folds": 3,
            "scoring": "roc_auc",
            "experiment_name": "pytest-loan-approval",
            "registered_model_name": None,
            "candidates": {
                "logistic_regression": {"C": 1.0, "max_iter": 400},
                "random_forest": {"n_estimators": 40, "max_depth": 6, "min_samples_leaf": 5},
            },
        },
        "evaluation": {
            "metrics_path": str(tmp_path / "metrics.json"),
            "confusion_matrix_path": str(tmp_path / "confusion_matrix.json"),
            "roc_curve_path": str(tmp_path / "roc_curve.json"),
            "min_roc_auc": 0.6,
            "min_f1": 0.5,
        },
        "serving": {
            "model_path": str(tmp_path / "model.joblib"),
            "metadata_path": str(tmp_path / "model_metadata.json"),
            "approval_threshold": 0.5,
        },
    }
    for section, values in overrides.items():
        params[section].update(values)

    params_file = tmp_path / "params.yaml"
    params_file.write_text(yaml.safe_dump(params), encoding="utf-8")
    return load_config(params_file)


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """Run ingestion -> validation -> transformation -> training -> evaluation once."""
    tmp_path = tmp_path_factory.mktemp("pipeline")
    import os

    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{tmp_path / 'mlflow.db'}"
    cfg = _write_params(tmp_path)

    generate_data.main(cfg)
    train_path, test_path = ingest(cfg)
    report_path = validate_stage(cfg)
    train_features, test_features, preprocessor_path = transform_stage(cfg)
    training_summary = train_stage(cfg)
    metrics = evaluate_stage(cfg)

    yield {
        "config": cfg,
        "tmp_path": tmp_path,
        "train_path": train_path,
        "test_path": test_path,
        "report_path": report_path,
        "train_features": train_features,
        "test_features": test_features,
        "preprocessor_path": preprocessor_path,
        "training_summary": training_summary,
        "metrics": metrics,
    }
    os.environ.pop("MLFLOW_TRACKING_URI", None)


class TestIngestionStage:
    def test_writes_both_splits(self, pipeline_run):
        assert pipeline_run["train_path"].exists()
        assert pipeline_run["test_path"].exists()

    def test_target_is_encoded_to_integers(self, pipeline_run):
        import pandas as pd

        train = pd.read_csv(pipeline_run["train_path"])
        assert set(train["Loan_Status"].unique()) <= {0, 1}

    def test_missing_raw_file_is_reported(self, tmp_path):
        cfg = _write_params(tmp_path, data={"raw_path": str(tmp_path / "absent.csv")})
        with pytest.raises(FileNotFoundError, match="Raw data not found"):
            ingest(cfg)


class TestValidationStage:
    def test_report_is_written_and_passing(self, pipeline_run):
        report = json.loads(pipeline_run["report_path"].read_text())
        assert report["status"] == "passed"
        assert {d["dataset"] for d in report["datasets"]} == {"train", "test"}

    def test_stage_fails_on_corrupt_data(self, tmp_path):
        import pandas as pd

        cfg = _write_params(tmp_path)
        generate_data.main(cfg)
        ingest(cfg)
        train = pd.read_csv(cfg.get_path("data.train_path"))
        train["Property_Area"] = "Atlantis"
        train.to_csv(cfg.get_path("data.train_path"), index=False)
        with pytest.raises(ValidationError):
            validate_stage(cfg)


class TestTransformationStage:
    def test_feature_files_and_preprocessor_exist(self, pipeline_run):
        assert pipeline_run["train_features"].exists()
        assert pipeline_run["test_features"].exists()
        assert pipeline_run["preprocessor_path"].exists()

    def test_preprocessor_artifact_carries_its_column_list(self, pipeline_run):
        artifact = joblib.load(pipeline_run["preprocessor_path"])
        assert "preprocessor" in artifact
        assert artifact["feature_columns"]

    def test_engineered_columns_reach_the_feature_file(self, pipeline_run):
        import pandas as pd

        columns = pd.read_csv(pipeline_run["train_features"], nrows=1).columns
        assert {"TotalIncome", "DebtToIncomeRatio", "HasCoapplicant"} <= set(columns)


class TestTrainingStage:
    def test_best_model_is_one_of_the_candidates(self, pipeline_run):
        assert pipeline_run["training_summary"]["best_model"] in {
            "logistic_regression", "random_forest"
        }

    def test_every_candidate_is_scored(self, pipeline_run):
        candidates = pipeline_run["training_summary"]["candidates"]
        assert len(candidates) == 2
        assert all("cv_mean" in c and "test_metrics" in c for c in candidates)

    def test_model_and_metadata_are_persisted(self, pipeline_run):
        cfg = pipeline_run["config"]
        assert cfg.get_path("training.model_path").exists()
        metadata = json.loads(cfg.get_path("training.metadata_path").read_text())
        assert metadata["mlflow_run_id"]
        assert metadata["training_rows"] == 300

    def test_saved_model_can_score_new_applications(self, pipeline_run):
        import pandas as pd

        from src.transformation.features import MODEL_FEATURES

        model = joblib.load(pipeline_run["config"].get_path("training.model_path"))
        features = pd.read_csv(pipeline_run["test_features"])[MODEL_FEATURES]
        proba = model.predict_proba(features)[:, 1]
        assert len(proba) == len(features)
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_mlflow_recorded_the_runs(self, pipeline_run):
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{pipeline_run['tmp_path'] / 'mlflow.db'}")
        experiment = mlflow.get_experiment_by_name("pytest-loan-approval")
        assert experiment is not None
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) >= 3            # one parent run + one per candidate


class TestEvaluationStage:
    def test_metrics_file_contents(self, pipeline_run):
        metrics = json.loads(pipeline_run["config"].get_path("evaluation.metrics_path").read_text())
        assert set(metrics) >= {"accuracy", "precision", "recall", "f1", "roc_auc"}
        assert metrics["roc_auc"] >= 0.6

    def test_confusion_matrix_sums_to_the_test_set(self, pipeline_run):
        cfg = pipeline_run["config"]
        matrix = json.loads(cfg.get_path("evaluation.confusion_matrix_path").read_text())
        total = sum(matrix[k] for k in
                    ("true_negative", "false_positive", "false_negative", "true_positive"))
        assert total == matrix["support"] == 100

    def test_roc_curve_points_are_monotonic(self, pipeline_run):
        roc = json.loads(pipeline_run["config"].get_path("evaluation.roc_curve_path").read_text())["roc"]
        fprs = [p["fpr"] for p in roc]
        assert fprs == sorted(fprs)

    def test_quality_gate_blocks_a_weak_model(self, pipeline_run):
        cfg = pipeline_run["config"]
        strict = load_config(cfg.get_path("evaluation.metrics_path").parent / "params.yaml")
        strict["evaluation"]["min_roc_auc"] = 0.999
        with pytest.raises(ModelQualityError, match="quality gate"):
            evaluate_stage(strict)
