"""Feature engineering and the sklearn preprocessing graph.

The same functions are used by the training stage and by the API, so an
application scored in production travels through exactly the transformations
the model was fitted on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.validation.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Domain features a credit analyst would compute by hand.
ENGINEERED_FEATURES = [
    "TotalIncome",
    "LogTotalIncome",
    "LoanAmountPerMonth",
    "DebtToIncomeRatio",
    "IncomeToLoanRatio",
    "HasCoapplicant",
]

NUMERIC_MODEL_FEATURES = NUMERIC_FEATURES + ENGINEERED_FEATURES
CATEGORICAL_MODEL_FEATURES = list(CATEGORICAL_FEATURES)
MODEL_FEATURES = NUMERIC_MODEL_FEATURES + CATEGORICAL_MODEL_FEATURES


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` with the derived affordability features added."""
    out = frame.copy()

    applicant = pd.to_numeric(out.get("ApplicantIncome"), errors="coerce").fillna(0.0)
    coapplicant = pd.to_numeric(out.get("CoapplicantIncome"), errors="coerce").fillna(0.0)
    amount = pd.to_numeric(out.get("LoanAmount"), errors="coerce")
    term = pd.to_numeric(out.get("Loan_Amount_Term"), errors="coerce")

    total_income = applicant + coapplicant
    # Median term of the training distribution is 360 months; use it as the
    # neutral fill so a missing term never produces an infinite instalment.
    safe_term = term.fillna(360.0).clip(lower=1.0)
    monthly_instalment = (amount * 1000.0) / safe_term

    out["TotalIncome"] = total_income
    out["LogTotalIncome"] = np.log1p(total_income)
    out["LoanAmountPerMonth"] = monthly_instalment
    out["DebtToIncomeRatio"] = monthly_instalment / total_income.clip(lower=1.0)
    out["IncomeToLoanRatio"] = total_income / (amount.fillna(amount.median()) * 1000.0).clip(lower=1.0)
    out["HasCoapplicant"] = (coapplicant > 0).astype(int)

    # Cast every numeric model input to float64. Integer columns cannot hold
    # NaN, so a missing value at serving time would otherwise change the dtype
    # and break schema enforcement against the training signature.
    for column in NUMERIC_MODEL_FEATURES:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")

    return out


def build_preprocessor() -> ColumnTransformer:
    """Median-impute + scale numerics, most-frequent-impute + one-hot categoricals."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_MODEL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_MODEL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def select_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Engineer features and keep only the columns the model consumes, in order."""
    enriched = add_engineered_features(frame)
    missing = [c for c in MODEL_FEATURES if c not in enriched.columns]
    if missing:
        raise KeyError(f"Input is missing required feature columns: {missing}")
    return enriched[MODEL_FEATURES]
