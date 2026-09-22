"""Shared fixtures.

The heavyweight fixtures (a trained pipeline, an API client) are session
scoped: they train once on a small slice of data and are reused, which keeps
the whole suite under a few seconds.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import load_config
from src.ingestion.generate_data import generate_loan_dataset
from src.ingestion.ingest import encode_target
from src.transformation.features import build_preprocessor, select_model_features
from src.validation.schema import TARGET_COLUMN


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    """A small, deterministic raw dataset - same generator the pipeline uses."""
    return generate_loan_dataset(n_samples=400, missing_rate=0.05, random_state=7)


@pytest.fixture(scope="session")
def encoded_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    frame = raw_frame.copy()
    frame[TARGET_COLUMN] = encode_target(frame[TARGET_COLUMN])
    return frame


@pytest.fixture(scope="session")
def trained_pipeline(encoded_frame: pd.DataFrame) -> Pipeline:
    """A real (small) end-to-end pipeline for service-level tests."""
    from sklearn.linear_model import LogisticRegression

    X = select_model_features(encoded_frame)
    y = encoded_frame[TARGET_COLUMN]
    pipeline = Pipeline(
        steps=[("preprocessor", build_preprocessor()), ("model", LogisticRegression(max_iter=500))]
    )
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def sample_application() -> dict:
    return {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5849,
        "CoapplicantIncome": 1500.0,
        "LoanAmount": 128.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 1.0,
        "Property_Area": "Urban",
    }
