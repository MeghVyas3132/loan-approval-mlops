"""Tests for the framework-independent prediction service."""

from __future__ import annotations

import joblib
import pytest

from src.prediction.schemas import LoanApplication
from src.prediction.service import ModelNotLoadedError, PredictionService


@pytest.fixture
def service(tmp_path, trained_pipeline, config) -> PredictionService:
    model_path = tmp_path / "model.joblib"
    joblib.dump(trained_pipeline, model_path)
    svc = PredictionService(config=config, model_path=model_path)
    svc.load()
    return svc


class TestLifecycle:
    def test_service_is_not_ready_before_load(self, tmp_path, trained_pipeline, config):
        model_path = tmp_path / "model.joblib"
        joblib.dump(trained_pipeline, model_path)
        assert PredictionService(config=config, model_path=model_path).is_ready is False

    def test_missing_model_file_raises(self, tmp_path, config):
        svc = PredictionService(config=config, model_path=tmp_path / "absent.joblib")
        with pytest.raises(FileNotFoundError, match="No model at"):
            svc.load()

    def test_predicting_before_load_raises(self, tmp_path, config, sample_application):
        svc = PredictionService(config=config, model_path=tmp_path / "absent.joblib")
        with pytest.raises(ModelNotLoadedError):
            svc.predict([LoanApplication(**sample_application)])

    def test_load_is_idempotent(self, service):
        service.load()
        assert service.is_ready


class TestPredictions:
    def test_single_prediction_contract(self, service, sample_application):
        response = service.predict([LoanApplication(**sample_application)])[0]
        assert response.loan_status in {"Approved", "Rejected"}
        assert response.prediction in {0, 1}
        assert 0.0 <= response.approval_probability <= 1.0
        assert response.request_id

    def test_decision_follows_the_threshold(self, service, sample_application):
        response = service.predict([LoanApplication(**sample_application)])[0]
        expected = int(response.approval_probability >= service.threshold)
        assert response.prediction == expected

    def test_batch_returns_one_response_per_application(self, service, sample_application):
        applications = [LoanApplication(**sample_application) for _ in range(5)]
        assert len(service.predict(applications)) == 5

    def test_request_ids_are_unique(self, service, sample_application):
        applications = [LoanApplication(**sample_application) for _ in range(5)]
        ids = {r.request_id for r in service.predict(applications)}
        assert len(ids) == 5

    def test_optional_fields_may_be_missing(self, service, sample_application):
        sparse = dict(sample_application)
        for field in ("Gender", "Married", "Dependents", "Self_Employed", "Credit_History"):
            sparse.pop(field)
        response = service.predict([LoanApplication(**sparse)])[0]
        assert 0.0 <= response.approval_probability <= 1.0

    def test_strong_applicant_scores_above_weak_applicant(self, service, sample_application):
        strong = dict(sample_application, Credit_History=1.0, ApplicantIncome=20000, LoanAmount=60.0)
        weak = dict(sample_application, Credit_History=0.0, ApplicantIncome=1500, LoanAmount=600.0)
        strong_p = service.predict([LoanApplication(**strong)])[0].approval_probability
        weak_p = service.predict([LoanApplication(**weak)])[0].approval_probability
        assert strong_p > weak_p

    def test_predictions_are_deterministic(self, service, sample_application):
        first = service.predict([LoanApplication(**sample_application)])[0].approval_probability
        second = service.predict([LoanApplication(**sample_application)])[0].approval_probability
        assert first == second
