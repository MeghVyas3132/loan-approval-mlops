"""Model loading and inference, isolated from the web framework.

Keeping this layer free of FastAPI means the exact same code path is exercised
by the unit tests, by the API, and by any future batch-scoring job.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import Config, get_config
from src.prediction.schemas import LoanApplication, PredictionResponse
from src.transformation.features import add_engineered_features
from src.utils.io import read_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested before a model is available."""


class PredictionService:
    """Loads the trained pipeline once and scores applications against it."""

    def __init__(self, config: Config | None = None, model_path: str | Path | None = None) -> None:
        self._config = config or get_config()
        self._model_path = Path(model_path) if model_path else self._config.get_path("serving.model_path")
        self._metadata_path = self._config.get_path("serving.metadata_path")
        self._threshold = float(self._config.get_nested("serving.approval_threshold", 0.5))
        self._model: Any | None = None
        self._metadata: dict[str, Any] = {}
        self._lock = threading.Lock()

    # ---- lifecycle -------------------------------------------------------
    def load(self) -> None:
        """Load model + metadata from disk. Safe to call repeatedly."""
        with self._lock:
            if not self._model_path.exists():
                raise FileNotFoundError(
                    f"No model at {self._model_path}. Run `dvc repro` or `dvc pull` first."
                )
            self._model = joblib.load(self._model_path)
            self._metadata = read_json(self._metadata_path) if self._metadata_path.exists() else {}
            logger.info(
                "Loaded model '%s' (run %s) from %s",
                self._metadata.get("model_name", "unknown"),
                self._metadata.get("mlflow_run_id", "n/a"),
                self._model_path,
            )

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def model_name(self) -> str:
        return str(self._metadata.get("model_name", "unknown"))

    @property
    def model_version(self) -> str:
        run_id = self._metadata.get("mlflow_run_id")
        return str(run_id)[:8] if run_id else "local"

    # ---- inference -------------------------------------------------------
    def _to_frame(self, applications: Iterable[LoanApplication]) -> pd.DataFrame:
        rows = [app.model_dump() for app in applications]
        return add_engineered_features(pd.DataFrame(rows))

    def predict_proba(self, applications: Iterable[LoanApplication]) -> list[float]:
        if self._model is None:
            raise ModelNotLoadedError("Model is not loaded")
        frame = self._to_frame(applications)
        return [float(p) for p in self._model.predict_proba(frame)[:, 1]]

    def predict(self, applications: Iterable[LoanApplication]) -> list[PredictionResponse]:
        """Score applications and wrap the results in the response contract."""
        applications = list(applications)
        probabilities = self.predict_proba(applications)
        responses = []
        for probability in probabilities:
            approved = probability >= self._threshold
            responses.append(
                PredictionResponse(
                    loan_status="Approved" if approved else "Rejected",
                    prediction=int(approved),
                    approval_probability=round(probability, 6),
                    threshold=self._threshold,
                    model_name=self.model_name,
                    model_version=self.model_version,
                    request_id=str(uuid.uuid4()),
                )
            )
        return responses


_service: PredictionService | None = None
_service_lock = threading.Lock()


def get_service() -> PredictionService:
    """Process-wide singleton used as the FastAPI dependency.

    It takes no arguments on purpose: FastAPI inspects a dependency's signature
    and would otherwise try to bind ``config`` to a query parameter. Tests
    inject their own instance through ``app.dependency_overrides`` or
    :func:`set_service`.
    """
    global _service
    with _service_lock:
        if _service is None:
            _service = PredictionService()
    return _service


def set_service(service: PredictionService) -> None:
    """Replace the singleton (used by tests and by warm-reload tooling)."""
    global _service
    with _service_lock:
        _service = service


def reset_service() -> None:
    """Drop the singleton so the next call rebuilds it from configuration."""
    global _service
    with _service_lock:
        _service = None
