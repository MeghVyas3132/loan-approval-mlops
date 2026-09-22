"""Tests for feature engineering and the preprocessing graph."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.transformation.features import (
    ENGINEERED_FEATURES,
    MODEL_FEATURES,
    add_engineered_features,
    build_preprocessor,
    select_model_features,
)


class TestEngineeredFeatures:
    def test_all_declared_features_are_created(self, encoded_frame: pd.DataFrame):
        out = add_engineered_features(encoded_frame)
        assert set(ENGINEERED_FEATURES).issubset(out.columns)

    def test_original_frame_is_not_mutated(self, encoded_frame: pd.DataFrame):
        before = encoded_frame.copy()
        add_engineered_features(encoded_frame)
        pd.testing.assert_frame_equal(encoded_frame, before)

    def test_total_income_is_the_household_sum(self):
        frame = pd.DataFrame(
            {
                "ApplicantIncome": [5000.0],
                "CoapplicantIncome": [2000.0],
                "LoanAmount": [100.0],
                "Loan_Amount_Term": [360.0],
            }
        )
        out = add_engineered_features(frame)
        assert out.loc[0, "TotalIncome"] == 7000.0
        assert out.loc[0, "LogTotalIncome"] == pytest.approx(np.log1p(7000.0))

    def test_instalment_and_debt_ratio(self):
        frame = pd.DataFrame(
            {
                "ApplicantIncome": [10_000.0],
                "CoapplicantIncome": [0.0],
                "LoanAmount": [120.0],     # 120k over 120 months -> 1000/month
                "Loan_Amount_Term": [120.0],
            }
        )
        out = add_engineered_features(frame)
        assert out.loc[0, "LoanAmountPerMonth"] == pytest.approx(1000.0)
        assert out.loc[0, "DebtToIncomeRatio"] == pytest.approx(0.1)

    def test_has_coapplicant_flag(self):
        frame = pd.DataFrame(
            {
                "ApplicantIncome": [4000.0, 4000.0],
                "CoapplicantIncome": [0.0, 1200.0],
                "LoanAmount": [80.0, 80.0],
                "Loan_Amount_Term": [360.0, 360.0],
            }
        )
        out = add_engineered_features(frame)
        assert out["HasCoapplicant"].tolist() == [0, 1]

    def test_missing_term_does_not_produce_infinities(self):
        frame = pd.DataFrame(
            {
                "ApplicantIncome": [3000.0],
                "CoapplicantIncome": [0.0],
                "LoanAmount": [100.0],
                "Loan_Amount_Term": [np.nan],
            }
        )
        out = add_engineered_features(frame)
        assert np.isfinite(out.loc[0, "LoanAmountPerMonth"])

    def test_zero_income_does_not_divide_by_zero(self):
        frame = pd.DataFrame(
            {
                "ApplicantIncome": [0.0],
                "CoapplicantIncome": [0.0],
                "LoanAmount": [50.0],
                "Loan_Amount_Term": [360.0],
            }
        )
        out = add_engineered_features(frame)
        assert np.isfinite(out.loc[0, "DebtToIncomeRatio"])

    def test_numeric_columns_are_float64(self, encoded_frame: pd.DataFrame):
        out = add_engineered_features(encoded_frame)
        assert out["ApplicantIncome"].dtype == np.float64


class TestSelectModelFeatures:
    def test_column_order_matches_the_contract(self, encoded_frame: pd.DataFrame):
        assert list(select_model_features(encoded_frame).columns) == MODEL_FEATURES

    def test_missing_input_column_raises(self, encoded_frame: pd.DataFrame):
        with pytest.raises(KeyError):
            select_model_features(encoded_frame.drop(columns=["Property_Area"]))


class TestPreprocessor:
    def test_fit_transform_produces_finite_numeric_matrix(self, encoded_frame: pd.DataFrame):
        features = select_model_features(encoded_frame)
        matrix = build_preprocessor().fit_transform(features)
        assert matrix.shape[0] == len(features)
        assert np.isfinite(matrix).all()

    def test_imputation_removes_nulls(self, encoded_frame: pd.DataFrame):
        features = select_model_features(encoded_frame)
        assert features.isna().any().any()          # the raw feed does have gaps
        matrix = build_preprocessor().fit_transform(features)
        assert not np.isnan(matrix).any()

    def test_numeric_columns_are_standardised(self, encoded_frame: pd.DataFrame):
        features = select_model_features(encoded_frame)
        preprocessor = build_preprocessor().fit(features)
        matrix = preprocessor.transform(features)
        n_numeric = len([c for c in MODEL_FEATURES if c in features.select_dtypes("number").columns])
        assert matrix[:, :n_numeric].mean() == pytest.approx(0.0, abs=1e-6)

    def test_unseen_category_is_ignored_not_fatal(self, encoded_frame: pd.DataFrame):
        features = select_model_features(encoded_frame)
        preprocessor = build_preprocessor().fit(features)
        unseen = features.head(1).copy()
        unseen.loc[unseen.index[0], "Property_Area"] = "Lunar"
        assert preprocessor.transform(unseen).shape[1] == preprocessor.transform(features.head(1)).shape[1]
