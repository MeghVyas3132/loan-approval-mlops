# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Multi-stage build: wheels are compiled in the builder, the runtime image only
# carries the virtualenv, the source and the model artefact.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Only the serving dependencies - DVC/MLflow are build-time tools, not runtime.
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install \
        "pandas>=2.1" "numpy>=1.26" "scikit-learn>=1.4" "joblib>=1.3" "PyYAML>=6.0" \
        "fastapi>=0.110" "uvicorn[standard]>=0.29" "pydantic>=2.6" \
        "prometheus-client>=0.20" "prometheus-fastapi-instrumentator>=7.0"

# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="loan-approval-api" \
      org.opencontainers.image.description="Loan approval prediction service (MLOps end-term project)" \
      org.opencontainers.image.source="https://github.com/<your-org>/loan-approval-mlops"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Run as an unprivileged user - required by the Kubernetes securityContext.
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --create-home appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser app.py params.yaml ./
# The trained artefact: produced by `dvc repro`/`dvc pull` before the build.
COPY --chown=appuser:appuser models/ ./models/

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=2).status==200 else 1)"

# One worker per container on purpose: prometheus_client keeps its registry in
# process memory, so multiple workers behind one port would each report a slice
# of the traffic. Scale horizontally with Kubernetes replicas/HPA instead -
# every pod is then scraped as its own target.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
