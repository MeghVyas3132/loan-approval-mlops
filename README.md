# Loan Approval Prediction - End-to-End MLOps System

[![CI](https://github.com/MeghVyas3132/loan-approval-mlops/actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
[![CD](https://github.com/MeghVyas3132/loan-approval-mlops/actions/workflows/cd.yml/badge.svg)](../../actions/workflows/cd.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![Coverage](https://img.shields.io/badge/coverage-96%25-brightgreen)

A production-style machine learning system that decides whether a loan
application should be **approved** or **rejected**, covering the full lifecycle:
data versioning, validation, experiment tracking, testing, a REST API,
containerisation, CI/CD, Kubernetes deployment, and live monitoring.

*MLOps end-term project - Course STDE 301, Vijaybhoomi School of Science and Technology.*

---

## Table of contents

- [What this system does](#what-this-system-does)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [1. Data versioning with DVC](#1-data-versioning-with-dvc)
- [2. The data pipeline](#2-the-data-pipeline)
- [3. Experiment tracking with MLflow](#3-experiment-tracking-with-mlflow)
- [4. Testing](#4-testing)
- [5. The FastAPI service](#5-the-fastapi-service)
- [6. Docker](#6-docker)
- [7. CI/CD with GitHub Actions](#7-cicd-with-github-actions)
- [8. Kubernetes deployment](#8-kubernetes-deployment)
- [9. Monitoring and dashboards](#9-monitoring-and-dashboards)
- [Results](#results)
- [Design decisions](#design-decisions)
- [Rubric map](#rubric-map)

---

## What this system does

A lender receives thousands of applications a day, and manual review is slow and
inconsistent. This system scores an application in milliseconds from the
attributes on the form:

| Field | Type | Notes |
| --- | --- | --- |
| `ApplicantIncome` | numeric | monthly income |
| `CoapplicantIncome` | numeric | 0 when there is no co-applicant |
| `LoanAmount` | numeric | requested amount, in thousands |
| `Loan_Amount_Term` | numeric | months (12-480) |
| `Credit_History` | numeric | 1 = meets guidelines, 0 = does not |
| `Property_Area` | categorical | Urban / Semiurban / Rural |
| `Education` | categorical | Graduate / Not Graduate |
| `Married` | categorical | Yes / No |
| `Self_Employed` | categorical | Yes / No |
| `Gender`, `Dependents` | categorical | optional |

The decision, the approval probability, and the model version that produced it
come back over a REST API that is deployed to Kubernetes and monitored in
Grafana.

---

## Architecture

```
                    ┌──────────────────────────┐
                    │  Raw data (DVC tracked)  │
                    └────────────┬─────────────┘
                                 ▼
                    ┌──────────────────────────┐
     dvc repro      │  1. Data ingestion       │  stratified train/test split
     drives every   ├──────────────────────────┤
     stage below    │  2. Data validation      │  schema contract, fails fast
                    ├──────────────────────────┤
                    │  3. Feature engineering  │  affordability features + preprocessor
                    ├──────────────────────────┤
                    │  4. Model training       │──────► MLflow (params, metrics,
                    │     3 candidates, CV     │        artefacts, registry)
                    ├──────────────────────────┤
                    │  5. Evaluation + gate    │  ROC-AUC / F1 thresholds
                    └────────────┬─────────────┘
                                 ▼
                        models/model.joblib
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
   pytest (150 tests)      Docker image            GitHub Actions
                                 │                  CI -> CD -> GHCR
                                 ▼
                    ┌──────────────────────────┐
                    │  Kubernetes (namespace   │
                    │  mlops): Deployment,     │
                    │  Service, HPA, Ingress   │
                    └────────────┬─────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │  FastAPI  /predict       │
                    │           /metrics       │
                    └────────┬────────┬────────┘
                             ▼        ▼
                     Predictions   Prometheus ──► Grafana dashboard
                                        │
                                        └──► alert rules + nightly PSI drift check
```

---

## Reviewing this submission

The submitted archive is the complete working repository, not just the source:

| Included | Why it is there |
| --- | --- |
| `.git/` | 17 commits - the history itself is part of the deliverable |
| `.dvc/cache/` | the DVC object store, so `dvc checkout` and `dvc status` work offline with no remote |
| `data/`, `models/` | the exact data and model the reported metrics came from |
| `mlflow.db`, `mlartifacts/` | the recorded experiment runs, openable in the MLflow UI (paths already made relative) |
| `reports/` | validation report, metrics, confusion matrix and ROC curve |

So the project runs immediately after unzipping:

```bash
pip install -r requirements-dev.txt
pytest                                                  # 150 tests
dvc status                                              # pipeline up to date
python app.py                                           # http://localhost:8000/docs
mlflow ui --backend-store-uri sqlite:///mlflow.db       # http://127.0.0.1:5000 (macOS AirPlay owns localhost:5000)
```

Python caches (`__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `coverage.xml`)
are excluded - they are regenerated by the commands above.

---

## Quick start

```bash
git clone https://github.com/MeghVyas3132/loan-approval-mlops.git && cd loan-approval-mlops

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# 1. Get the data (from DVC remote, or regenerate it deterministically)
dvc pull || python -m src.ingestion.generate_data

# 2. Run the whole pipeline: ingest -> validate -> transform -> train -> evaluate
dvc repro

# 3. Check what it produced
dvc metrics show
pytest

# 4. Serve it
python app.py            # http://localhost:8000/docs
```

Everything above is also available through `make`:

```bash
make help        # list targets
make pipeline    # dvc repro
make test        # pytest with coverage
make compose-up  # API + Prometheus + Grafana
```

---

## Repository layout

```
loan-approval-mlops/
├── data/
│   ├── raw/                     # DVC-tracked source data (.dvc pointer in git)
│   └── processed/               # pipeline outputs: splits + engineered features
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── src/
│   ├── ingestion/               # dataset generation + train/test split
│   ├── validation/              # schema contract + validation stage
│   ├── transformation/          # feature engineering + preprocessing graph
│   ├── training/                # MLflow-tracked training + evaluation gate
│   ├── prediction/              # FastAPI app, request schemas, model service
│   ├── monitoring/              # PSI drift detection
│   ├── utils/                   # logging + IO helpers
│   └── config.py                # typed access to params.yaml
├── scripts/                     # make_mlflow_portable.py
├── tests/                       # 150 tests, 96% coverage
├── deployment/
│   ├── docker/                  # docker-compose stack (API + Prometheus + Grafana)
│   └── kubernetes/              # namespace, deployment, service, HPA, ingress, PDB,
│                                # ServiceMonitor, drift CronJob, monitoring stack
├── monitoring/
│   ├── prometheus/              # scrape config + alert rules
│   ├── grafana/                 # provisioning + the dashboard JSON
│   └── generate_traffic.py      # demo traffic so the dashboards have data
├── .github/workflows/           # ci.yml, cd.yml, monitoring.yml
├── dvc.yaml                     # the 5-stage pipeline
├── params.yaml                  # every tunable parameter, in one place
├── requirements.txt             # runtime deps (requirements-dev.txt adds tooling)
├── Dockerfile                   # multi-stage, non-root, health-checked
├── Makefile
├── app.py                       # uvicorn entry point
└── README.md
```

---

## 1. Data versioning with DVC

The raw CSV never enters git - git stores a 3-line pointer, DVC stores the data.

```bash
python -m src.ingestion.generate_data      # writes data/raw/loan_applications.csv
dvc add data/raw/loan_applications.csv     # creates the .dvc pointer
git add data/raw/loan_applications.csv.dvc data/raw/.gitignore
dvc remote add -d localremote /tmp/loan-approval-dvcstore
dvc push                                   # data to the remote, pointer to git
```

`data/raw/loan_applications.csv.dvc`:

```yaml
outs:
  - md5: 0dd218179ccb1b0c2b4483a7acb979c8
    size: 206632
    hash: md5
    path: loan_applications.csv
```

The configured remote is a local directory, which stands in for the S3 / GCS /
Azure bucket a production project would use - swapping it is a one-line change
in `.dvc/config`. Because this submission bundles `.dvc/cache`, `dvc checkout`
restores every tracked file without reaching a remote at all.

Anyone who clones the repo runs `dvc pull` and gets byte-identical data. Every
pipeline output (splits, features, preprocessor, model) is likewise tracked by
DVC through `dvc.lock`, so `git checkout <old-commit> && dvc checkout` restores
the exact data *and* model of that commit.

Useful commands:

```bash
dvc dag              # print the stage graph
dvc status           # what is out of date
dvc metrics show     # metrics from the tracked JSON files
dvc metrics diff     # compare against another commit
dvc plots show reports/roc_curve.json
```

### About the dataset

The raw file is generated by `src/ingestion/generate_data.py` rather than
downloaded, so the project is reproducible with no external dependency. The
schema mirrors the standard loan-prediction dataset, and the label comes from a
documented logistic scoring function over credit history, affordability,
education and property area, plus noise - the signal is real but not perfectly
separable. Point `data.raw_path` at a real CSV with the same columns and every
downstream stage works unchanged.

---

## 2. The data pipeline

Five DVC stages, each a plain Python module you can also run by hand:

| Stage | Command | Output |
| --- | --- | --- |
| `data_ingestion` | `python -m src.ingestion.ingest` | `data/processed/train.csv`, `test.csv` |
| `data_validation` | `python -m src.validation.validate` | `reports/validation_report.json` |
| `data_transformation` | `python -m src.transformation.transform` | feature CSVs + `models/preprocessor.joblib` |
| `model_training` | `python -m src.training.train` | `models/model.joblib`, `reports/train_metrics.json` |
| `model_evaluation` | `python -m src.training.evaluate` | `reports/metrics.json`, confusion matrix, ROC curve |

**Validation** enforces the contract declared once in
`src/validation/schema.py` - required columns, dtypes, value ranges, allowed
categories, missing-value budget, and target balance. A breach raises
`ValidationError` and the pipeline stops before a bad batch reaches training.

**Feature engineering** adds what a credit analyst would compute by hand:

| Feature | Definition | Why |
| --- | --- | --- |
| `TotalIncome` | applicant + co-applicant | household capacity, not individual |
| `LogTotalIncome` | `log1p(TotalIncome)` | incomes are heavy-tailed |
| `LoanAmountPerMonth` | `amount × 1000 / term` | the actual instalment |
| `DebtToIncomeRatio` | instalment / total income | affordability - the strongest engineered signal |
| `IncomeToLoanRatio` | income / amount | inverse leverage |
| `HasCoapplicant` | co-applicant income > 0 | a second earner changes the risk |

The preprocessor (median impute + scale for numerics, most-frequent impute +
one-hot for categoricals) is **fitted on the training split only** and saved as
part of the model pipeline, so training and serving apply identical transforms.

---

## 3. Experiment tracking with MLflow

Every training run opens a parent MLflow run and one nested run per candidate:

```bash
python -m src.training.train
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000 (macOS AirPlay owns localhost:5000)
```

Logged per run: model type and hyperparameters, cross-validated ROC-AUC (mean
and std), hold-out accuracy / precision / recall / F1 / ROC-AUC, the fitted
pipeline with its input signature and an input example, `model_metadata.json`,
and tags for the project, stage and Python version. The winning model is
registered as `loan-approval-classifier` in the MLflow Model Registry.

Point the runs at a tracking server instead of the local SQLite store with:

```bash
export MLFLOW_TRACKING_URI=http://mlflow.example.com:5000
```

The run id is written into `models/model_metadata.json` and served by the API at
`GET /model-info`, so a prediction in production is traceable back to the exact
experiment that produced the model.

MLflow records an absolute artifact path for each run, which would break the UI
on anyone else's checkout. `scripts/make_mlflow_portable.py` rewrites those
paths to be relative to the project root, so the tracking database travels with
the repository:

```bash
python scripts/make_mlflow_portable.py --dry-run   # report what would change
python scripts/make_mlflow_portable.py             # rewrite in place
```

---

## 4. Testing

```bash
pytest                       # 150 tests, ~12s, coverage gate at 85%
pytest -m "not integration"  # unit tests only
pytest --cov-report=html     # browsable report in htmlcov/
```

| File | What it covers |
| --- | --- |
| `test_ingestion.py` | generator determinism, target encoding, stratified splitting |
| `test_validation.py` | every rule in the data contract, each failure mode |
| `test_transformation.py` | feature formulas, division-by-zero, imputation, unseen categories |
| `test_training.py` | candidate construction, metric correctness, learned behaviour |
| `test_service.py` | model loading, thresholding, batch scoring, error paths |
| `test_schemas.py` | request validation - the API's first line of defence |
| `test_api.py` | every endpoint, 422/503 handling, Prometheus metric emission |
| `test_monitoring.py` | PSI maths, drift classification bands, report generation |
| `test_pipeline_stages.py` | all five stages end to end in a temp workspace |
| `test_config_utils.py` | configuration resolution and IO helpers |
| `test_mlflow_portability.py` | the artifact-path rewrite, including idempotency |

Coverage is currently **96%** against an 85% gate enforced in `pyproject.toml`,
so CI fails if coverage regresses.

---

## 5. The FastAPI service

```bash
python app.py                     # or: uvicorn app:app --reload
```

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | liveness - answers while the process is up |
| `GET` | `/ready` | readiness - 503 until a model is loaded |
| `GET` | `/model-info` | model card: name, version, training metrics, features |
| `POST` | `/predict` | score one application |
| `POST` | `/predict/batch` | score up to 1000 applications |
| `GET` | `/metrics` | Prometheus exposition |
| `GET` | `/docs` | interactive OpenAPI documentation |

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "Gender": "Male", "Married": "Yes", "Dependents": "1",
    "Education": "Graduate", "Self_Employed": "No",
    "ApplicantIncome": 5849, "CoapplicantIncome": 1500,
    "LoanAmount": 128, "Loan_Amount_Term": 360,
    "Credit_History": 1, "Property_Area": "Urban"
  }'
```

```json
{
  "loan_status": "Approved",
  "prediction": 1,
  "approval_probability": 0.892341,
  "threshold": 0.5,
  "model_name": "logistic_regression",
  "model_version": "a1b2c3d4",
  "request_id": "7f3c9e2a-..."
}
```

The model is loaded once at start-up (FastAPI lifespan), not per request.
Invalid payloads are rejected with 422 by the Pydantic contract before they
reach the model; if no model is present the service still starts, `/health`
stays green, and `/ready` returns 503 - which is exactly what lets Kubernetes
hold traffic back instead of crash-looping.

---

## 6. Docker

```bash
docker build -t loan-approval-api:local .
docker run --rm -p 8000:8000 loan-approval-api:local
```

The image is multi-stage (wheels built in a builder stage, only the virtualenv
copied into the runtime), installs serving dependencies only - no DVC or MLflow,
runs as the non-root user `appuser` (uid 10001), and declares a `HEALTHCHECK`
against `/health`. `.dockerignore` keeps data, caches and tests out of the
context.

The full local stack:

```bash
docker compose -f deployment/docker/docker-compose.yml up --build
```

| Service | URL |
| --- | --- |
| API | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |

---

## 7. CI/CD with GitHub Actions

**`ci.yml`** - on every push and pull request:

1. **lint** - `ruff check` over `src`, `tests`, `app.py`
2. **test** - pytest with coverage on Python 3.10, 3.11 and 3.12
3. **pipeline** - `dvc repro` with a cached `.dvc/cache`, then the quality gate;
   metrics are written to the job summary and the model is uploaded as an artefact
4. **docker** - build the image, start the container, and smoke-test
   `/health`, `/ready`, `/predict` and `/metrics`

**`cd.yml`** - on `main` and on `v*.*.*` tags: rebuild the model, push a tagged
image to GHCR, apply the Kubernetes manifests, wait for the rollout, smoke-test
the live service, and **roll back automatically** if any step fails. The cluster
step is skipped with a warning when the `KUBE_CONFIG` secret is absent, so the
workflow stays green in a fork.

**`monitoring.yml`** - nightly PSI drift check that publishes a per-feature
table to the job summary.

Secrets used: `GITHUB_TOKEN` (automatic, pushes to GHCR) and `KUBE_CONFIG`
(base64-encoded kubeconfig).

---

## 8. Kubernetes deployment

```bash
make docker-build                                  # local image: loan-approval-api:local
make k8s-deploy                                    # namespace, config, deployment, service, HPA, PDB
kubectl -n mlops set image deployment/loan-approval-api api=loan-approval-api:local   # local cluster only
kubectl -n mlops port-forward svc/loan-approval-api 8000:80
make k8s-monitoring                                # in-cluster Prometheus + Grafana (creates its secret)
```

`servicemonitor.yaml` only applies on clusters running the Prometheus Operator,
and the HPA needs metrics-server to read CPU/memory (`<unknown>` without it).
Docker Desktop does not forward NodePorts to localhost - use `port-forward`.

| Manifest | Purpose |
| --- | --- |
| `namespace.yaml` | the `mlops` namespace |
| `configmap.yaml` | log level, port, MLflow URI |
| `deployment.yaml` | 3 replicas, rolling update with `maxUnavailable: 0`, liveness/readiness/startup probes, CPU+memory requests and limits, non-root read-only root filesystem, Prometheus scrape annotations, topology spread |
| `service.yaml` | ClusterIP, plus a NodePort for minikube/kind |
| `hpa.yaml` | autoscale 2-10 replicas on 70% CPU / 80% memory |
| `ingress.yaml` | nginx ingress at `loan-approval.local` |
| `pdb.yaml` | keep at least one pod during voluntary disruptions |
| `servicemonitor.yaml` | scrape target for the Prometheus Operator |
| `drift-cronjob.yaml` | nightly drift check as a CronJob |
| `monitoring-stack.yaml` | in-cluster Prometheus + Grafana for clusters without kube-prometheus-stack |

All manifests were validated with `kubectl apply --dry-run=client`.

---

## 9. Monitoring and dashboards

**Custom metrics** exported by the service on `/metrics`, alongside the standard
HTTP request-rate/latency/size metrics:

| Metric | Type | Question it answers |
| --- | --- | --- |
| `loan_predictions_total{decision,model_name}` | counter | how many approvals vs rejections? |
| `loan_prediction_latency_seconds` | histogram | how fast is inference (p50/p95/p99)? |
| `loan_prediction_probability` | histogram | is the score distribution shifting? |
| `loan_prediction_errors_total{error_type}` | counter | what is failing? |
| `loan_prediction_batch_size` | histogram | how are clients calling the API? |
| `loan_model_loaded` | gauge | is this pod actually servable? |

**Grafana dashboard** (`monitoring/grafana/dashboards/loan-approval-overview.json`,
auto-provisioned) in four rows: service health, traffic and latency, model
behaviour (approval rate + score-distribution heatmap), and errors/batching.

**Alert rules** (`monitoring/prometheus/alert_rules.yml`): API down, model not
loaded, p95 latency > 500 ms, 5xx rate > 5%, accumulating inference errors,
approval rate above 90% or below 20% for 15 minutes, and no traffic at all.

**Data drift** - `src/monitoring/drift.py` computes the Population Stability
Index per feature against the training distribution (PSI < 0.1 stable,
0.1-0.25 warning, > 0.25 alert):

```bash
python -m src.monitoring.drift --current data/processed/test.csv
```

To see the dashboards with live data:

```bash
make compose-up
python monitoring/generate_traffic.py --requests 300
```

---

## Results

Hold-out performance of the selected model (600 applications never seen during
training or selection):

| Metric | Value |
| --- | --- |
| ROC-AUC | 0.805 |
| Accuracy | 0.788 |
| Precision | 0.790 |
| Recall | 0.927 |
| F1 | 0.853 |

Cross-validated ROC-AUC across candidates (5-fold on the training split):

| Model | CV ROC-AUC | Hold-out ROC-AUC |
| --- | --- | --- |
| **Logistic regression** (selected) | **0.814 ± 0.029** | 0.805 |
| Random forest | 0.812 ± 0.028 | 0.780 |
| Gradient boosting | 0.805 ± 0.025 | 0.783 |

Logistic regression wins on cross-validated ROC-AUC and generalises best, which
is the right outcome for lending: the decision stays explainable, and a
coefficient per feature is far easier to defend to a credit committee than a
tree ensemble. Recall (0.93) is deliberately higher than precision - the gate
selects on ROC-AUC, and at the default 0.5 threshold the model errs towards
approving. `serving.approval_threshold` in `params.yaml` moves that trade-off
without retraining.

---

## Design decisions

**Why generate the raw data?** A marked project must run on any machine with no
credentials or downloads. The generator is versioned, seeded and documented, and
the pipeline treats its output exactly like an external feed - `dvc add`, then
every stage reads the tracked file.

**Why SQLite for MLflow rather than `./mlruns`?** MLflow 3 has retired the plain
file store, and a database backend is what enables the Model Registry. One
environment variable switches the whole project to a remote tracking server.

**Why is the preprocessor inside the model pipeline?** A separate preprocessor
is the classic source of training/serving skew. Saving one sklearn `Pipeline`
means the API cannot apply a different transform than training did - and
`tests/test_api.py` scores through the same object.

**Why a quality gate in the pipeline instead of only in CI?** `dvc repro` fails
on a weak model, so a bad model never reaches `models/model.joblib` locally
either. CI re-runs the same check, so there is one definition of "good enough".

**Why PSI for drift?** It is the standard in credit risk, needs no labels (which
arrive months later, when the loan matures or defaults), and is cheap enough to
run nightly on a CronJob.

---

## Rubric map

| Component | Marks | Where to look |
| --- | --- | --- |
| Git practices | 4 | granular commits, `.gitignore`, no data or artefacts in git, branch-protected CI |
| DVC usage | 4 | `data/raw/*.dvc`, `dvc.yaml` (5 stages), `dvc.lock`, remote + `dvc push`, params/metrics/plots tracking |
| Data pipeline | 4 | `src/ingestion`, `src/validation` (schema contract), `src/transformation` (6 engineered features) |
| Pytest coverage | 4 | `tests/` - 150 tests, 96% coverage, 85% gate in `pyproject.toml` |
| MLflow tracking | 4 | `src/training/train.py` - nested runs, params, metrics, artefacts, signature, model registry |
| FastAPI service | 4 | `src/prediction/` - 7 endpoints, Pydantic contract, lifespan model load, OpenAPI docs |
| Dockerization | 4 | `Dockerfile` (multi-stage, non-root, healthcheck), `.dockerignore`, `docker-compose.yml` |
| GitHub Actions | 4 | `.github/workflows/` - CI (lint/test/pipeline/docker), CD (GHCR + rollout + rollback), nightly monitoring |
| Kubernetes deployment | 4 | `deployment/kubernetes/` - 10 manifests, probes, HPA, PDB, ingress, all dry-run validated |
| Monitoring & dashboards | 4 | 6 custom metrics, Grafana dashboard (14 panels), 8 alert rules, PSI drift detector + CronJob |
