"""Request-validation tests - the API's first line of defence."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.prediction.schemas import BatchPredictionRequest, LoanApplication


class TestLoanApplication:
    def test_valid_payload(self, sample_application):
        application = LoanApplication(**sample_application)
        assert application.ApplicantIncome == 5849

    def test_optional_fields_default_to_none(self, sample_application):
        optional = {"Gender", "Married", "Dependents", "Self_Employed",
                    "Credit_History", "CoapplicantIncome"}
        payload = {k: v for k, v in sample_application.items() if k not in optional}
        application = LoanApplication(**payload)
        assert application.Gender is None
        assert application.CoapplicantIncome == 0.0

    @pytest.mark.parametrize(
        "field,value",
        [
            ("ApplicantIncome", -100),
            ("LoanAmount", 0),
            ("LoanAmount", 5000),
            ("Loan_Amount_Term", 5),
            ("Credit_History", 0.5),
            ("Property_Area", "Offshore"),
            ("Education", "PhD"),
            ("Dependents", "7"),
        ],
    )
    def test_invalid_values_are_rejected(self, sample_application, field, value):
        with pytest.raises(ValidationError):
            LoanApplication(**{**sample_application, field: value})

    def test_required_fields_are_enforced(self, sample_application):
        payload = dict(sample_application)
        payload.pop("Education")
        with pytest.raises(ValidationError):
            LoanApplication(**payload)


class TestBatchRequest:
    def test_batch_must_not_be_empty(self):
        with pytest.raises(ValidationError):
            BatchPredictionRequest(applications=[])

    def test_batch_is_capped(self, sample_application):
        with pytest.raises(ValidationError):
            BatchPredictionRequest(applications=[sample_application] * 1001)

    def test_valid_batch(self, sample_application):
        request = BatchPredictionRequest(applications=[sample_application] * 3)
        assert len(request.applications) == 3
