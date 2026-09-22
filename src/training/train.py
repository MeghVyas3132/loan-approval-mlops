"""DVC stage 4 - model training with MLflow experiment tracking.

Three candidate models are cross-validated; each gets its own nested MLflow run
carrying its parameters, CV scores and hold-out metrics. The winner (by the
configured scoring metric) is refitted end to end - preprocessor + estimator in
a single sklearn ``Pipeline`` - logged as an MLflow model, and written to
``models/model.joblib`` for the API to load.
"""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import PROJECT_ROOT, Config, load_config
from src.transformation.features import MODEL_FEATURES, build_preprocessor
from src.utils.io import ensure_parent, write_json
from src.utils.logger import get_logger
from src.validation.schema import TARGET_COLUMN

logger = get_logger(__name__)


def build_candidates(
    candidate_params: dict[str, dict[str, Any]], random_state: int
) -> dict[str, BaseEstimator]:
    """Instantiate the candidate estimators declared in ``params.yaml``."""
    factories = {
        "logistic_regression": lambda p: LogisticRegression(random_state=random_state, **p),
        "random_forest": lambda p: RandomForestClassifier(random_state=random_state, n_jobs=-1, **p),
        "gradient_boosting": lambda p: GradientBoostingClassifier(random_state=random_state, **p),
    }
    unknown = set(candidate_params) - set(factories)
    if unknown:
        raise ValueError(f"Unsupported candidate model(s): {sorted(unknown)}")
    return {name: factories[name](params or {}) for name, params in candidate_params.items()}


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> dict[str, float]:
    """Classification metrics used consistently by training, evaluation and tests."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return {k: round(v, 6) for k, v in metrics.items()}


def _configure_mlflow(config: Config) -> str:
    """Point MLflow at a tracking server, or at the local SQLite store.

    ``MLFLOW_TRACKING_URI`` wins when set (that is how CI and the docker-compose
    stack talk to a real tracking server). Otherwise runs go to a local SQLite
    backend - MLflow 3 has retired the plain-file store, and SQLite is what
    enables the model registry used below.
    """
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        uri = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"

    artifact_root = PROJECT_ROOT / "mlartifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(uri)
    experiment_name = config.get_nested("training.experiment_name", "loan-approval")
    if mlflow.get_experiment_by_name(experiment_name) is None:
        # MLflow expands any artifact location to an absolute path when the
        # experiment is created, so the tracking database ends up carrying this
        # machine's paths. scripts/make_mlflow_portable.py rewrites them to
        # relative paths before the project is shared.
        mlflow.create_experiment(experiment_name, artifact_location=artifact_root.as_uri())
    mlflow.set_experiment(experiment_name)

    logger.info("MLflow tracking URI: %s | artifacts: %s", uri, artifact_root)
    return uri


def _load_split(config: Config) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_df = pd.read_csv(config.get_path("transformation.train_features_path"))
    test_df = pd.read_csv(config.get_path("transformation.test_features_path"))
    X_train, y_train = train_df[MODEL_FEATURES], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[MODEL_FEATURES], test_df[TARGET_COLUMN]
    return X_train, y_train, X_test, y_test


def train_stage(config: Config | None = None) -> dict[str, Any]:
    config = config or load_config()
    training_cfg = config.section("training")
    random_state = int(config.get_nested("base.random_state", 42))
    scoring = training_cfg.get("scoring", "roc_auc")

    X_train, y_train, X_test, y_test = _load_split(config)
    candidates = build_candidates(training_cfg["candidates"], random_state)
    cv = StratifiedKFold(
        n_splits=int(training_cfg.get("cv_folds", 5)), shuffle=True, random_state=random_state
    )

    _configure_mlflow(config)
    results: list[dict[str, Any]] = []

    with mlflow.start_run(run_name=f"training-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}") as parent_run:
        mlflow.set_tags(
            {
                "project": config.get_nested("base.project_name", "loan-approval-mlops"),
                "stage": "model_training",
                "python_version": platform.python_version(),
            }
        )
        mlflow.log_params(
            {
                "cv_folds": cv.get_n_splits(),
                "scoring": scoring,
                "n_train_rows": len(X_train),
                "n_test_rows": len(X_test),
                "n_features": len(MODEL_FEATURES),
                "random_state": random_state,
            }
        )

        for name, estimator in candidates.items():
            with mlflow.start_run(run_name=name, nested=True):
                pipeline = Pipeline(
                    steps=[("preprocessor", build_preprocessor()), ("model", estimator)]
                )
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
                pipeline.fit(X_train, y_train)

                y_pred = pipeline.predict(X_test)
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                metrics = compute_metrics(y_test.to_numpy(), y_pred, y_proba)

                hyperparameters = training_cfg["candidates"][name] or {}
                mlflow.log_params(
                    {"model_type": name, **{f"model__{k}": v for k, v in hyperparameters.items()}}
                )
                mlflow.log_metrics(
                    {
                        **metrics,
                        f"cv_{scoring}_mean": float(cv_scores.mean()),
                        f"cv_{scoring}_std": float(cv_scores.std()),
                    }
                )

                logger.info(
                    "%-20s cv_%s=%.4f (+/-%.4f) | test roc_auc=%.4f f1=%.4f",
                    name, scoring, cv_scores.mean(), cv_scores.std(),
                    metrics.get("roc_auc", float("nan")), metrics["f1"],
                )
                results.append(
                    {
                        "model": name,
                        "cv_mean": float(cv_scores.mean()),
                        "cv_std": float(cv_scores.std()),
                        "test_metrics": metrics,
                        "pipeline": pipeline,
                    }
                )

        best = max(results, key=lambda r: r["cv_mean"])
        best_pipeline: Pipeline = best["pipeline"]
        logger.info("Selected best model: %s (cv_%s=%.4f)", best["model"], scoring, best["cv_mean"])

        mlflow.set_tag("best_model", best["model"])
        mlflow.log_metrics(
            {
                "best_cv_score": best["cv_mean"],
                **{f"best_{k}": v for k, v in best["test_metrics"].items()},
            }
        )

        signature = mlflow.models.infer_signature(X_train, best_pipeline.predict(X_train))
        log_model_kwargs = dict(
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            # sklearn ColumnTransformers carry numpy dtypes that the default
            # skops writer refuses to serialise; cloudpickle handles the whole
            # pipeline and is what the API's joblib artifact matches.
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        try:
            mlflow.sklearn.log_model(
                best_pipeline,
                registered_model_name=training_cfg.get("registered_model_name"),
                **log_model_kwargs,
            )
            logger.info("Registered model '%s' in the MLflow registry",
                        training_cfg.get("registered_model_name"))
        except Exception as exc:  # registry needs a database-backed tracking store
            logger.warning("Model registry unavailable (%s); logging model without registration", exc)
            mlflow.sklearn.log_model(best_pipeline, **log_model_kwargs)

        model_path = ensure_parent(config.get_path("training.model_path"))
        joblib.dump(best_pipeline, model_path)

        metadata = {
            "model_name": best["model"],
            "model_path": str(model_path.relative_to(model_path.parents[1])),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "mlflow_run_id": parent_run.info.run_id,
            "mlflow_experiment_id": parent_run.info.experiment_id,
            "feature_columns": MODEL_FEATURES,
            "cv_scoring": scoring,
            "cv_score": best["cv_mean"],
            "test_metrics": best["test_metrics"],
            "sklearn_version": __import__("sklearn").__version__,
            "training_rows": int(len(X_train)),
        }
        write_json(config.get_path("training.metadata_path"), metadata)
        mlflow.log_dict(metadata, "model_metadata.json")

        summary = {
            "best_model": best["model"],
            "scoring": scoring,
            "candidates": [
                {k: v for k, v in r.items() if k != "pipeline"} for r in results
            ],
        }
        write_json(config.get_path("training.metrics_path"), summary)
        mlflow.log_artifact(str(config.get_path("training.metrics_path")))

    logger.info("Training complete. Model saved to %s", model_path)
    return summary


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    print(json.dumps(train_stage()["best_model"], indent=2))
