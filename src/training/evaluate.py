"""DVC stage 5 - hold-out evaluation and quality gate.

Scores the saved model on the untouched test split, writes DVC-tracked metrics
and plot data, and fails the pipeline when the model drops below the thresholds
in ``params.yaml`` - the same gate GitHub Actions enforces before a release.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from src.config import Config, load_config
from src.training.train import compute_metrics
from src.transformation.features import MODEL_FEATURES
from src.utils.io import read_json, write_json
from src.utils.logger import get_logger
from src.validation.schema import TARGET_COLUMN

logger = get_logger(__name__)


class ModelQualityError(RuntimeError):
    """Raised when the trained model is below the release thresholds."""


def evaluate_stage(config: Config | None = None) -> dict[str, Any]:
    config = config or load_config()
    evaluation_cfg = config.section("evaluation")

    model = joblib.load(config.get_path("training.model_path"))
    test_df = pd.read_csv(config.get_path("transformation.test_features_path"))
    X_test, y_test = test_df[MODEL_FEATURES], test_df[TARGET_COLUMN].to_numpy()

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, y_proba)

    metadata = read_json(config.get_path("training.metadata_path"))
    payload = dict(metrics)  # DVC metrics files stay purely numeric

    write_json(config.get_path("evaluation.metrics_path"), payload)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    write_json(
        config.get_path("evaluation.confusion_matrix_path"),
        {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
            "support": int(len(y_test)),
        },
    )

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    # Thin the curve so the plot file stays small and readable in `dvc plots`.
    step = max(1, len(fpr) // 100)
    write_json(
        config.get_path("evaluation.roc_curve_path"),
        {
            "roc": [
                {"fpr": round(float(f), 5), "tpr": round(float(t), 5), "threshold": round(float(th), 5)}
                for f, t, th in zip(
                    fpr[::step], tpr[::step],
                    np.nan_to_num(thresholds[::step], posinf=1.0), strict=True,
                )
            ]
        },
    )

    failures = []
    min_roc_auc = float(evaluation_cfg.get("min_roc_auc", 0.0))
    min_f1 = float(evaluation_cfg.get("min_f1", 0.0))
    if payload.get("roc_auc", 0.0) < min_roc_auc:
        failures.append(f"roc_auc {payload.get('roc_auc'):.4f} < {min_roc_auc}")
    if payload["f1"] < min_f1:
        failures.append(f"f1 {payload['f1']:.4f} < {min_f1}")

    logger.info("Hold-out metrics for %s: %s", metadata["model_name"], payload)
    if failures:
        raise ModelQualityError("Model quality gate failed: " + "; ".join(failures))

    logger.info("Quality gate passed (roc_auc >= %.2f, f1 >= %.2f)", min_roc_auc, min_f1)
    return payload


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    evaluate_stage()
