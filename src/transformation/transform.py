"""DVC stage 3 - feature engineering.

Fits the preprocessing graph on the training split only (so the test split
stays a clean hold-out), persists it for the API, and writes the engineered
frames that the training stage consumes.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.config import Config, load_config
from src.transformation.features import MODEL_FEATURES, build_preprocessor, select_model_features
from src.utils.io import ensure_parent, write_csv
from src.utils.logger import get_logger
from src.validation.schema import TARGET_COLUMN

logger = get_logger(__name__)


def transform_stage(config: Config | None = None) -> tuple[Path, Path, Path]:
    config = config or load_config()

    train_df = pd.read_csv(config.get_path("data.train_path"))
    test_df = pd.read_csv(config.get_path("data.test_path"))

    train_features = select_model_features(train_df)
    test_features = select_model_features(test_df)

    preprocessor = build_preprocessor()
    preprocessor.fit(train_features)
    logger.info(
        "Fitted preprocessor on %d rows -> %d model inputs",
        len(train_features),
        len(preprocessor.get_feature_names_out()),
    )

    train_out = train_features.copy()
    train_out[TARGET_COLUMN] = train_df[TARGET_COLUMN].to_numpy()
    test_out = test_features.copy()
    test_out[TARGET_COLUMN] = test_df[TARGET_COLUMN].to_numpy()

    train_path = write_csv(config.get_path("transformation.train_features_path"), train_out)
    test_path = write_csv(config.get_path("transformation.test_features_path"), test_out)

    preprocessor_path = ensure_parent(config.get_path("transformation.preprocessor_path"))
    joblib.dump(
        {"preprocessor": preprocessor, "feature_columns": MODEL_FEATURES},
        preprocessor_path,
    )
    logger.info("Artifacts: %s | %s | %s", train_path, test_path, preprocessor_path)
    return train_path, test_path, preprocessor_path


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    transform_stage()
