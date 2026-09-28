# Convenience targets for the loan-approval MLOps project.
# Run `make help` for the list.

.DEFAULT_GOAL := help
PYTHON ?= python
COMPOSE := docker compose -f deployment/docker/docker-compose.yml
IMAGE ?= loan-approval-api:local
NAMESPACE ?= mlops

.PHONY: help install data pipeline repro metrics test lint format api docker-build docker-run k8s-monitoring \
        compose-up compose-down mlflow k8s-deploy k8s-delete k8s-status drift clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime + dev dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt

data:  ## Regenerate the raw dataset and track it with DVC
	$(PYTHON) -m src.ingestion.generate_data
	dvc add data/raw/loan_applications.csv

pipeline:  ## Run the full DVC pipeline
	dvc repro

repro: pipeline  ## Alias for `make pipeline`

metrics:  ## Show DVC-tracked metrics
	dvc metrics show
	dvc plots show reports/roc_curve.json || true

test:  ## Run the test suite with coverage
	pytest

lint:  ## Lint the codebase
	ruff check src tests app.py

format:  ## Auto-format the codebase
	ruff format src tests app.py
	ruff check --fix src tests app.py

api:  ## Run the API locally with hot reload
	API_RELOAD=true $(PYTHON) app.py

mlflow:  ## Open the MLflow UI on http://localhost:5000
	mlflow ui --backend-store-uri "sqlite:///$(CURDIR)/mlflow.db" --port 5000

docker-build:  ## Build the serving image
	docker build -t $(IMAGE) .

docker-run:  ## Run the serving image on http://localhost:8000
	docker run --rm -p 8000:8000 --name loan-approval-api $(IMAGE)

compose-up:  ## Start API + Prometheus + Grafana
	$(COMPOSE) up --build -d
	@echo "API        http://localhost:8000/docs"
	@echo "Prometheus http://localhost:9090"
	@echo "Grafana    http://localhost:3000 (admin/admin)"

compose-down:  ## Stop the local stack
	$(COMPOSE) down -v

k8s-deploy:  ## Apply every Kubernetes manifest
	kubectl apply -f deployment/kubernetes/namespace.yaml
	kubectl apply -f deployment/kubernetes/configmap.yaml
	kubectl apply -f deployment/kubernetes/deployment.yaml
	kubectl apply -f deployment/kubernetes/service.yaml
	kubectl apply -f deployment/kubernetes/hpa.yaml
	kubectl apply -f deployment/kubernetes/pdb.yaml
	kubectl -n $(NAMESPACE) rollout status deployment/loan-approval-api

GRAFANA_PASSWORD ?= admin

k8s-monitoring:  ## Deploy in-cluster Prometheus + Grafana with the dashboard
	kubectl -n $(NAMESPACE) create secret generic grafana-admin \
		--from-literal=password=$(GRAFANA_PASSWORD) --dry-run=client -o yaml | kubectl apply -f -
	kubectl -n $(NAMESPACE) create configmap grafana-dashboard-provider \
		--from-file=monitoring/grafana/provisioning/dashboards/dashboards.yml --dry-run=client -o yaml | kubectl apply -f -
	kubectl -n $(NAMESPACE) create configmap grafana-dashboards \
		--from-file=monitoring/grafana/dashboards/ --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -f deployment/kubernetes/monitoring-stack.yaml
	kubectl -n $(NAMESPACE) rollout status deployment/grafana

k8s-status:  ## Show what is running in the cluster
	kubectl -n $(NAMESPACE) get pods,svc,hpa

k8s-delete:  ## Remove the deployment
	kubectl delete -f deployment/kubernetes/ --ignore-not-found

drift:  ## Run the drift detector against the test split
	$(PYTHON) -m src.monitoring.drift --current data/processed/test.csv

traffic:  ## Send sample traffic so the dashboards have data
	$(PYTHON) monitoring/generate_traffic.py --requests 300

clean:  ## Remove caches and generated reports
	rm -rf .pytest_cache .ruff_cache htmlcov coverage.xml junit.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
