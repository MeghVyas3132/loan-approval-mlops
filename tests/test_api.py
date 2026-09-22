"""End-to-end API tests driven through FastAPI's TestClient.

The app's dependency on the model singleton is overridden with a service
backed by the small fixture pipeline, so the suite never depends on artefacts
produced by a previous `dvc repro`.
"""

from __future__ import annotations

import joblib
import pytest
from fastapi.testclient import TestClient

from src.prediction.api import app
from src.prediction.service import PredictionService, get_service


@pytest.fixture(scope="module")
def client(tmp_path_factory, request):
    trained_pipeline = request.getfixturevalue("trained_pipeline")
    config = request.getfixturevalue("config")

    model_path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(trained_pipeline, model_path)

    service = PredictionService(config=config, model_path=model_path)
    service.load()
    app.dependency_overrides[get_service] = lambda: service

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


class TestMetaEndpoints:
    def test_root_banner(self, client):
        body = client.get("/").json()
        assert body["service"]
        assert body["docs"] == "/docs"

    def test_health_is_green(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_readiness_reports_a_loaded_model(self, client):
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True

    def test_model_info_exposes_the_feature_contract(self, client):
        body = client.get("/model-info").json()
        assert body["threshold"] == 0.5
        assert "model_name" in body

    def test_openapi_schema_is_served(self, client):
        schema = client.get("/openapi.json").json()
        assert "/predict" in schema["paths"]
        assert "/predict/batch" in schema["paths"]


class TestPredictEndpoint:
    def test_returns_a_decision(self, client, sample_application):
        response = client.post("/predict", json=sample_application)
        assert response.status_code == 200
        body = response.json()
        assert body["loan_status"] in {"Approved", "Rejected"}
        assert 0.0 <= body["approval_probability"] <= 1.0
        assert body["request_id"]

    def test_rejects_an_invalid_payload(self, client, sample_application):
        response = client.post("/predict", json={**sample_application, "ApplicantIncome": -5})
        assert response.status_code == 422

    def test_rejects_a_missing_required_field(self, client, sample_application):
        payload = dict(sample_application)
        payload.pop("Property_Area")
        assert client.post("/predict", json=payload).status_code == 422

    def test_accepts_nulls_for_optional_fields(self, client, sample_application):
        payload = {**sample_application, "Gender": None, "Credit_History": None}
        assert client.post("/predict", json=payload).status_code == 200

    def test_high_risk_application_is_scored_lower(self, client, sample_application):
        good = client.post("/predict", json={**sample_application, "Credit_History": 1.0}).json()
        bad = client.post("/predict", json={**sample_application, "Credit_History": 0.0}).json()
        assert good["approval_probability"] > bad["approval_probability"]


class TestBatchEndpoint:
    def test_scores_every_application(self, client, sample_application):
        payload = {"applications": [sample_application, sample_application, sample_application]}
        body = client.post("/predict/batch", json=payload).json()
        assert body["count"] == 3
        assert len(body["predictions"]) == 3
        assert body["approved_count"] == sum(p["prediction"] for p in body["predictions"])

    def test_empty_batch_is_rejected(self, client):
        assert client.post("/predict/batch", json={"applications": []}).status_code == 422


class TestMetricsEndpoint:
    def test_exposes_prometheus_text_format(self, client, sample_application):
        client.post("/predict", json=sample_application)
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

    def test_custom_model_metrics_are_published(self, client, sample_application):
        client.post("/predict", json=sample_application)
        body = client.get("/metrics").text
        for metric in (
            "loan_predictions_total",
            "loan_prediction_latency_seconds",
            "loan_prediction_probability",
            "loan_prediction_batch_size",
        ):
            assert metric in body

    def test_prediction_counter_increases(self, client, sample_application):
        def counter_value(text: str) -> float:
            values = [
                float(line.rsplit(" ", 1)[1])
                for line in text.splitlines()
                if line.startswith("loan_predictions_total{")
            ]
            return sum(values)

        before = counter_value(client.get("/metrics").text)
        client.post("/predict", json=sample_application)
        after = counter_value(client.get("/metrics").text)
        assert after == pytest.approx(before + 1)


class TestServiceUnavailable:
    def test_predict_returns_503_without_a_model(self, sample_application, config, tmp_path):
        empty_service = PredictionService(config=config, model_path=tmp_path / "missing.joblib")
        app.dependency_overrides[get_service] = lambda: empty_service
        try:
            with TestClient(app) as unloaded_client:
                assert unloaded_client.get("/ready").status_code == 503
                assert unloaded_client.post("/predict", json=sample_application).status_code == 503
        finally:
            app.dependency_overrides.clear()
