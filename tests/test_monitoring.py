"""Tests for the drift detector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    categorical_drift,
    classify,
    detect_drift,
    population_stability_index,
    run,
)


@pytest.fixture
def reference() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "ApplicantIncome": rng.normal(5000, 1200, 800),
            "CoapplicantIncome": rng.normal(1500, 600, 800),
            "LoanAmount": rng.normal(140, 40, 800),
            "Loan_Amount_Term": np.full(800, 360.0),
            "Credit_History": rng.choice([0.0, 1.0], 800, p=[0.15, 0.85]),
            "Property_Area": rng.choice(["Urban", "Semiurban", "Rural"], 800),
            "Gender": rng.choice(["Male", "Female"], 800),
            "Married": rng.choice(["Yes", "No"], 800),
            "Dependents": rng.choice(["0", "1", "2", "3+"], 800),
            "Education": rng.choice(["Graduate", "Not Graduate"], 800),
            "Self_Employed": rng.choice(["Yes", "No"], 800),
        }
    )


class TestPsi:
    def test_identical_distributions_have_near_zero_psi(self, reference):
        assert population_stability_index(reference["ApplicantIncome"],
                                          reference["ApplicantIncome"]) < 0.01

    def test_shifted_distribution_raises_psi(self, reference):
        shifted = reference["ApplicantIncome"] * 3.0
        assert population_stability_index(reference["ApplicantIncome"], shifted) > 0.25

    def test_empty_input_is_safe(self, reference):
        assert population_stability_index(reference["ApplicantIncome"], pd.Series(dtype=float)) == 0.0

    def test_constant_reference_is_safe(self):
        constant = pd.Series([1.0] * 100)
        assert population_stability_index(constant, pd.Series([2.0] * 100)) == 0.0

    def test_categorical_psi_detects_a_shift(self, reference):
        skewed = pd.Series(["Urban"] * 500)
        assert categorical_drift(reference["Property_Area"], skewed) > 0.25

    def test_categorical_psi_is_zero_for_the_same_mix(self, reference):
        assert categorical_drift(reference["Property_Area"], reference["Property_Area"]) < 0.01

    @pytest.mark.parametrize("psi,expected", [(0.01, "stable"), (0.15, "warning"), (0.4, "alert")])
    def test_classification_bands(self, psi, expected):
        assert classify(psi, warn=0.1, alert=0.25) == expected


class TestDetectDrift:
    def test_same_data_is_stable(self, reference):
        report = detect_drift(reference, reference)
        assert report["status"] == "stable"
        assert report["drifted_features"] == []

    def test_shifted_data_alerts(self, reference):
        drifted = reference.copy()
        drifted["ApplicantIncome"] = drifted["ApplicantIncome"] * 4
        drifted["Property_Area"] = "Rural"
        report = detect_drift(reference, drifted)
        assert report["status"] == "alert"
        assert "ApplicantIncome" in report["drifted_features"]

    def test_report_is_sorted_by_psi(self, reference):
        drifted = reference.copy()
        drifted["LoanAmount"] = drifted["LoanAmount"] * 2.5
        psis = [f["psi"] for f in detect_drift(reference, drifted)["features"]]
        assert psis == sorted(psis, reverse=True)

    def test_row_counts_are_recorded(self, reference):
        report = detect_drift(reference, reference.head(100))
        assert report["reference_rows"] == 800
        assert report["current_rows"] == 100


class TestRun:
    def test_writes_a_report_file(self, tmp_path, reference, config, monkeypatch):
        reference_path = tmp_path / "reference.csv"
        current_path = tmp_path / "current.csv"
        report_path = tmp_path / "drift_report.json"
        reference.to_csv(reference_path, index=False)
        reference.to_csv(current_path, index=False)

        patched = dict(config)
        patched["monitoring"] = {
            "reference_path": str(reference_path),
            "drift_report_path": str(report_path),
            "psi_warn": 0.1,
            "psi_alert": 0.25,
        }
        from src.config import Config

        report = run(current_path, config=Config(patched))
        assert report_path.exists()
        assert report["status"] == "stable"
