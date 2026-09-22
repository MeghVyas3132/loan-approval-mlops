"""Pydantic request/response models - the public contract of the API.

Field constraints mirror :mod:`src.validation.schema`, so an application that
would have failed data validation during training is rejected at the edge with
a 422 instead of producing a meaningless score.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoanApplication(BaseModel):
    """A single loan application submitted for scoring."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    Gender: Literal["Male", "Female"] | None = Field(None, description="Applicant gender")
    Married: Literal["Yes", "No"] | None = Field(None, description="Marital status")
    Dependents: Literal["0", "1", "2", "3+"] | None = Field(None, description="Number of dependents")
    Education: Literal["Graduate", "Not Graduate"] = Field(..., description="Education level")
    Self_Employed: Literal["Yes", "No"] | None = Field(None, description="Employment status")
    ApplicantIncome: float = Field(..., ge=0, le=1_000_000, description="Monthly applicant income")
    CoapplicantIncome: float = Field(0.0, ge=0, le=1_000_000, description="Monthly co-applicant income")
    LoanAmount: float | None = Field(None, gt=0, le=1_000, description="Requested amount in thousands")
    Loan_Amount_Term: float | None = Field(None, ge=12, le=480, description="Loan term in months")
    Credit_History: float | None = Field(None, ge=0, le=1, description="1 = meets credit guidelines")
    Property_Area: Literal["Urban", "Semiurban", "Rural"] = Field(..., description="Property location")

    @field_validator("Credit_History")
    @classmethod
    def _credit_history_is_binary(cls, value: float | None) -> float | None:
        if value is not None and value not in (0.0, 1.0):
            raise ValueError("Credit_History must be 0 or 1")
        return value


class PredictionResponse(BaseModel):
    loan_status: Literal["Approved", "Rejected"]
    prediction: int = Field(..., description="1 = approved, 0 = rejected")
    approval_probability: float = Field(..., ge=0, le=1)
    threshold: float = Field(..., ge=0, le=1)
    model_name: str
    model_version: str
    request_id: str


class BatchPredictionRequest(BaseModel):
    applications: list[LoanApplication] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
    approved_count: int


class HealthResponse(BaseModel):
    status: Literal["healthy", "unhealthy"]
    model_loaded: bool
    version: str


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    trained_at: str | None = None
    mlflow_run_id: str | None = None
    feature_columns: list[str]
    test_metrics: dict[str, float] = {}
    threshold: float
