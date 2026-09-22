"""Application entry point.

``app.py`` at the repository root is what the Dockerfile, uvicorn and the
Kubernetes deployment all reference: ``uvicorn app:app``.
"""

from __future__ import annotations

import os

import uvicorn

from src.prediction.api import app

__all__ = ["app"]


if __name__ == "__main__":  # pragma: no cover - local dev server
    uvicorn.run(
        "app:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
    )
