"""Create the raw loan-application dataset.

The exam project needs a reproducible data source that anyone can regenerate,
so the raw file is synthesised from a documented generative process instead of
being downloaded. The schema mirrors the classic loan-prediction dataset
(applicant income, co-applicant income, loan amount, term, credit history,
property area, education, marital and employment status), and the label is
produced by a logistic model over those drivers plus noise, so the downstream
pipeline has a genuine - but not perfectly separable - signal to learn.

Run once, then version the output with ``dvc add data/raw/loan_applications.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.utils.io import write_csv
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_COLUMNS = [
    "Loan_ID",
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
    "Loan_Status",
]

# Columns that may contain missing values in the raw feed, mirroring the
# incomplete forms a real lender receives.
NULLABLE_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Self_Employed",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_loan_dataset(
    n_samples: int = 3000,
    missing_rate: float = 0.04,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return ``n_samples`` synthetic loan applications with a learnable label."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not 0.0 <= missing_rate < 0.5:
        raise ValueError("missing_rate must be in [0, 0.5)")

    rng = np.random.default_rng(random_state)

    gender = rng.choice(["Male", "Female"], size=n_samples, p=[0.78, 0.22])
    married = rng.choice(["Yes", "No"], size=n_samples, p=[0.65, 0.35])
    dependents = rng.choice(["0", "1", "2", "3+"], size=n_samples, p=[0.57, 0.17, 0.17, 0.09])
    education = rng.choice(["Graduate", "Not Graduate"], size=n_samples, p=[0.78, 0.22])
    self_employed = rng.choice(["No", "Yes"], size=n_samples, p=[0.86, 0.14])
    property_area = rng.choice(
        ["Urban", "Semiurban", "Rural"], size=n_samples, p=[0.34, 0.38, 0.28]
    )

    # Incomes are heavy tailed: log-normal keeps the shape realistic.
    applicant_income = np.round(rng.lognormal(mean=8.4, sigma=0.55, size=n_samples)).astype(int)
    has_coapplicant = (married == "Yes") & (rng.random(n_samples) < 0.7)
    coapplicant_income = np.where(
        has_coapplicant,
        np.round(rng.lognormal(mean=7.6, sigma=0.7, size=n_samples)),
        0.0,
    )

    total_income = applicant_income + coapplicant_income
    # Requested amount (in thousands) scales with household income.
    loan_amount = np.round(
        np.clip(total_income * rng.uniform(0.010, 0.045, size=n_samples), 15, 700)
    )
    loan_term = rng.choice([360.0, 180.0, 120.0, 300.0, 84.0, 60.0], size=n_samples,
                           p=[0.80, 0.08, 0.04, 0.04, 0.02, 0.02])
    credit_history = rng.choice([1.0, 0.0], size=n_samples, p=[0.84, 0.16])

    # ---- label: a transparent logistic scoring model over the real drivers ----
    emi = loan_amount / np.maximum(loan_term / 12.0, 1.0)          # yearly-ish burden
    debt_to_income = emi / np.maximum(total_income / 1000.0, 0.1)
    logit = (
        -1.15
        + 3.30 * credit_history
        + 0.45 * (education == "Graduate")
        + 0.30 * (property_area == "Semiurban")
        - 0.25 * (property_area == "Rural")
        + 0.20 * (married == "Yes")
        - 0.20 * (self_employed == "Yes")
        + 0.55 * np.log1p(total_income / 5000.0)
        - 1.45 * debt_to_income
        - 0.15 * (dependents == "3+")
    )
    probability = _sigmoid(logit + rng.normal(0.0, 0.45, size=n_samples))
    approved = (rng.random(n_samples) < probability).astype(int)

    frame = pd.DataFrame(
        {
            "Loan_ID": [f"LP{1000000 + i}" for i in range(n_samples)],
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": property_area,
            "Loan_Status": np.where(approved == 1, "Y", "N"),
        }
    )[RAW_COLUMNS]

    if missing_rate > 0:
        for column in NULLABLE_COLUMNS:
            mask = rng.random(n_samples) < missing_rate
            frame.loc[mask, column] = np.nan

    logger.info(
        "Generated %d applications | approval rate=%.3f | missing cells=%d",
        len(frame),
        (frame["Loan_Status"] == "Y").mean(),
        int(frame.isna().sum().sum()),
    )
    return frame


def main(config: Config | None = None) -> Path:
    config = config or load_config()
    data_cfg = config.section("data")
    frame = generate_loan_dataset(
        n_samples=int(data_cfg["n_samples"]),
        missing_rate=float(data_cfg["missing_rate"]),
        random_state=int(config.get_nested("base.random_state", 42)),
    )
    output = write_csv(config.get_path("data.raw_path"), frame)
    logger.info("Raw dataset written to %s", output)
    return output


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
