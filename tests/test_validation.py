"""Tests for the data contract and the validation stage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation.schema import CATEGORICAL_FEATURES, LOAN_SCHEMA, NUMERIC_FEATURES
from src.validation.validate import ValidationError, validate_frame


class TestSchema:
    def test_numeric_and_categorical_partition_the_schema(self):
        assert set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES) == set(LOAN_SCHEMA.names())
        assert not set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)

    def test_lookup_of_unknown_column_raises(self):
        with pytest.raises(KeyError):
            LOAN_SCHEMA.get("NotAColumn")

    def test_credit_history_bounds_are_binary(self):
        spec = LOAN_SCHEMA.get("Credit_History")
        assert (spec.minimum, spec.maximum) == (0, 1)


class TestValidateFrame:
    def test_clean_frame_passes(self, encoded_frame: pd.DataFrame):
        report = validate_frame(encoded_frame, min_rows=10, target_balance=(0.2, 0.9))
        assert report["status"] == "passed"
        assert report["rows"] == len(encoded_frame)
        assert report["errors"] == []

    def test_report_records_column_statistics(self, encoded_frame: pd.DataFrame):
        report = validate_frame(encoded_frame, target_balance=(0.0, 1.0))
        income = next(c for c in report["column_checks"] if c["column"] == "ApplicantIncome")
        assert income["min"] >= 0
        assert "mean" in income

    def test_missing_required_column_fails(self, encoded_frame: pd.DataFrame):
        frame = encoded_frame.drop(columns=["Credit_History"])
        with pytest.raises(ValidationError, match="Credit_History"):
            validate_frame(frame, target_balance=(0.0, 1.0))

    def test_out_of_range_value_fails(self, encoded_frame: pd.DataFrame):
        frame = encoded_frame.copy()
        frame.loc[0, "LoanAmount"] = 5000.0          # schema maximum is 1000
        with pytest.raises(ValidationError, match="above maximum"):
            validate_frame(frame, target_balance=(0.0, 1.0))

    def test_unexpected_category_fails(self, encoded_frame: pd.DataFrame):
        frame = encoded_frame.copy()
        frame.loc[0, "Property_Area"] = "Offshore"
        with pytest.raises(ValidationError, match="unexpected categories"):
            validate_frame(frame, target_balance=(0.0, 1.0))

    def test_excess_missing_values_fail(self, encoded_frame: pd.DataFrame):
        frame = encoded_frame.copy()
        frame.loc[frame.index[:200], "LoanAmount"] = np.nan
        with pytest.raises(ValidationError, match="null"):
            validate_frame(frame, max_missing_fraction=0.15, target_balance=(0.0, 1.0))

    def test_null_in_non_nullable_column_fails(self, encoded_frame: pd.DataFrame):
        frame = encoded_frame.copy()
        frame.loc[0, "Education"] = np.nan
        with pytest.raises(ValidationError, match="non-nullable"):
            validate_frame(frame, target_balance=(0.0, 1.0))

    def test_too_few_rows_fails(self, encoded_frame: pd.DataFrame):
        with pytest.raises(ValidationError, match="rows"):
            validate_frame(encoded_frame.head(5), min_rows=100, target_balance=(0.0, 1.0))

    def test_target_imbalance_fails(self, encoded_frame: pd.DataFrame):
        with pytest.raises(ValidationError, match="positive rate"):
            validate_frame(encoded_frame, target_balance=(0.95, 1.0))

    def test_non_binary_target_fails(self, encoded_frame: pd.DataFrame):
        frame = encoded_frame.copy()
        frame.loc[0, "Loan_Status"] = 7
        with pytest.raises(ValidationError, match="0/1"):
            validate_frame(frame, target_balance=(0.0, 1.0))

    def test_target_can_be_optional_for_inference_payloads(self, encoded_frame: pd.DataFrame):
        frame = encoded_frame.drop(columns=["Loan_Status"])
        report = validate_frame(frame, require_target=False)
        assert report["status"] == "passed"
