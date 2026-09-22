"""Tests for dataset generation and the ingestion stage."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.generate_data import RAW_COLUMNS, generate_loan_dataset
from src.ingestion.ingest import encode_target, split_dataset
from src.validation.schema import TARGET_COLUMN


class TestGenerateData:
    def test_shape_and_columns(self, raw_frame: pd.DataFrame):
        assert len(raw_frame) == 400
        assert list(raw_frame.columns) == RAW_COLUMNS

    def test_is_deterministic(self):
        first = generate_loan_dataset(n_samples=50, random_state=3)
        second = generate_loan_dataset(n_samples=50, random_state=3)
        pd.testing.assert_frame_equal(first, second)

    def test_loan_ids_are_unique(self, raw_frame: pd.DataFrame):
        assert raw_frame["Loan_ID"].is_unique

    def test_label_is_not_degenerate(self, raw_frame: pd.DataFrame):
        approval_rate = (raw_frame["Loan_Status"] == "Y").mean()
        assert 0.3 < approval_rate < 0.9

    def test_missing_values_are_injected(self):
        frame = generate_loan_dataset(n_samples=500, missing_rate=0.1, random_state=1)
        assert frame["Credit_History"].isna().any()

    def test_no_missing_values_when_rate_is_zero(self):
        frame = generate_loan_dataset(n_samples=100, missing_rate=0.0, random_state=1)
        assert frame.isna().sum().sum() == 0

    @pytest.mark.parametrize("bad_kwargs", [{"n_samples": 0}, {"missing_rate": 0.9}])
    def test_rejects_invalid_arguments(self, bad_kwargs):
        with pytest.raises(ValueError):
            generate_loan_dataset(**bad_kwargs)


class TestEncodeTarget:
    def test_maps_y_and_n(self):
        encoded = encode_target(pd.Series(["Y", "N", "Y"]))
        assert encoded.tolist() == [1, 0, 1]

    def test_accepts_already_encoded_values(self):
        assert encode_target(pd.Series([1, 0])).tolist() == [1, 0]

    def test_rejects_unknown_labels(self):
        with pytest.raises(ValueError, match="Unrecognised target values"):
            encode_target(pd.Series(["Y", "MAYBE"]))


class TestSplitDataset:
    def test_split_sizes_and_disjointness(self, encoded_frame: pd.DataFrame):
        train, test = split_dataset(encoded_frame, TARGET_COLUMN, test_size=0.25, random_state=42)
        assert len(test) == pytest.approx(len(encoded_frame) * 0.25, abs=1)
        assert len(train) + len(test) == len(encoded_frame)
        assert set(train["Loan_ID"]).isdisjoint(set(test["Loan_ID"]))

    def test_split_is_stratified(self, encoded_frame: pd.DataFrame):
        train, test = split_dataset(encoded_frame, TARGET_COLUMN, test_size=0.2, random_state=42)
        assert train[TARGET_COLUMN].mean() == pytest.approx(test[TARGET_COLUMN].mean(), abs=0.05)

    def test_split_is_reproducible(self, encoded_frame: pd.DataFrame):
        a, _ = split_dataset(encoded_frame, TARGET_COLUMN, test_size=0.2, random_state=11)
        b, _ = split_dataset(encoded_frame, TARGET_COLUMN, test_size=0.2, random_state=11)
        pd.testing.assert_frame_equal(a, b)
