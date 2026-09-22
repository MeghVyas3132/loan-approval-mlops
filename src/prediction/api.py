"""FastAPI application exposing the loan-approval model.

Endpoints
---------
GET  /health         liveness  - always answers while the process is up
GET  /ready          readiness - 503 until a model is loaded (k8s gate)
GET  /model-info     model card: name, version, training metrics
POST /predict        score one application
POST /predict/batch  score up to 1000 applications in one call
GET  /metrics        Prometheus exposition (default + custom model metrics)
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from src.config import get_config
from src.prediction.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    LoanApplication,
    ModelInfo,
    PredictionResponse,
)
from src.prediction.service import ModelNotLoadedError, PredictionService, get_service
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()

# ---- custom Prometheus metrics (scraped by Prometheus, charted in Grafana) ----
PREDICTIONS_TOTAL = Counter(
    "loan_predictions_total",
    "Loan predictions served, labelled by decision.",
    ["decision", "model_name"],
)
PREDICTION_LATENCY = Histogram(
    "loan_prediction_latency_seconds",
    "Wall-clock time spent inside the model for one request.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
PREDICTION_PROBABILITY = Histogram(
    "loan_prediction_probability",
    "Distribution of approval probabilities - a shift here signals model drift.",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
PREDICTION_ERRORS = Counter(
    "loan_prediction_errors_total",
    "Failed prediction requests, labelled by error type.",
    ["error_type"],
)
MODEL_LOADED = Gauge("loan_model_loaded", "1 when a model is loaded and servable.")
BATCH_SIZE = Histogram(
    "loan_prediction_batch_size",
    "Number of applications per batch request.",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at start-up instead of on the first request."""
    service = get_service()
    try:
        service.load()
        MODEL_LOADED.set(1)
    except FileNotFoundError as exc:
        MODEL_LOADED.set(0)
        # Start anyway so /health and /metrics work; /ready stays red until a
        # model appears, which is exactly what the k8s readiness probe wants.
        logger.error("Starting without a model: %s", exc)
    yield
    logger.info("Shutting down prediction service")


app = FastAPI(
    title=config.get_nested("serving.api_title", "Loan Approval Prediction API"),
    version=config.get_nested("serving.api_version", "1.0.0"),
    description=(
        "Predicts whether a loan application should be approved. "
        "Part of the MLOps end-term project: DVC + MLflow + FastAPI + Docker + "
        "GitHub Actions + Kubernetes + Prometheus/Grafana."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Default HTTP metrics (request count, latency, size) on /metrics.
Instrumentator(
    should_group_status_codes=False,
    excluded_handlers=["/metrics", "/health"],
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)


@app.exception_handler(ModelNotLoadedError)
async def _model_not_loaded_handler(_: Request, exc: ModelNotLoadedError) -> JSONResponse:
    PREDICTION_ERRORS.labels(error_type="model_not_loaded").inc()
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": f"Model unavailable: {exc}"},
    )


@app.get("/", tags=["meta"], summary="Service banner")
async def root() -> dict[str, str]:
    return {
        "service": app.title,
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"], summary="Liveness probe")
async def health(service: PredictionService = Depends(get_service)) -> HealthResponse:
    return HealthResponse(status="healthy", model_loaded=service.is_ready, version=app.version)


@app.get("/ready", response_model=HealthResponse, tags=["meta"], summary="Readiness probe")
async def ready(service: PredictionService = Depends(get_service)) -> HealthResponse:
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )
    return HealthResponse(status="healthy", model_loaded=True, version=app.version)


@app.get("/model-info", response_model=ModelInfo, tags=["meta"], summary="Deployed model card")
async def model_info(service: PredictionService = Depends(get_service)) -> ModelInfo:
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )
    metadata = service.metadata
    return ModelInfo(
        model_name=service.model_name,
        model_version=service.model_version,
        trained_at=metadata.get("trained_at"),
        mlflow_run_id=metadata.get("mlflow_run_id"),
        feature_columns=metadata.get("feature_columns", []),
        test_metrics=metadata.get("test_metrics", {}),
        threshold=service.threshold,
    )


def _record(responses: list[PredictionResponse], model_name: str, elapsed: float) -> None:
    PREDICTION_LATENCY.observe(elapsed)
    for response in responses:
        PREDICTIONS_TOTAL.labels(decision=response.loan_status, model_name=model_name).inc()
        PREDICTION_PROBABILITY.observe(response.approval_probability)


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"],
          summary="Score a single application")
async def predict(
    application: LoanApplication, service: PredictionService = Depends(get_service)
) -> PredictionResponse:
    started = time.perf_counter()
    try:
        response = service.predict([application])[0]
    except ModelNotLoadedError:
        raise
    except Exception as exc:  # pragma: no cover - unexpected inference failure
        PREDICTION_ERRORS.labels(error_type=type(exc).__name__).inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    _record([response], service.model_name, time.perf_counter() - started)
    BATCH_SIZE.observe(1)
    return response


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"],
          summary="Score many applications")
async def predict_batch(
    request: BatchPredictionRequest, service: PredictionService = Depends(get_service)
) -> BatchPredictionResponse:
    started = time.perf_counter()
    try:
        responses = service.predict(request.applications)
    except ModelNotLoadedError:
        raise
    except Exception as exc:  # pragma: no cover - unexpected inference failure
        PREDICTION_ERRORS.labels(error_type=type(exc).__name__).inc()
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    _record(responses, service.model_name, time.perf_counter() - started)
    BATCH_SIZE.observe(len(responses))
    return BatchPredictionResponse(
        predictions=responses,
        count=len(responses),
        approved_count=sum(r.prediction for r in responses),
    )
