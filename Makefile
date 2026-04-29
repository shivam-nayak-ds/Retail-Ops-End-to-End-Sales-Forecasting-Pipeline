# ─────────────────────────────────────────────────────────────
#  Retail-Ops Sales Forecasting Pipeline — Makefile
#  Run `make help` to see all available commands
# ─────────────────────────────────────────────────────────────

.PHONY: help install install-dev clean train predict serve test lint format docker-build docker-run mlflow zenml

# ── Colors ───────────────────────────────────────────────────
BOLD  := \033[1m
RESET := \033[0m
GREEN := \033[32m
CYAN  := \033[36m

help:  ## Show all available commands
	@echo ""
	@echo "$(BOLD)Retail-Ops Sales Forecasting Pipeline$(RESET)"
	@echo "$(CYAN)──────────────────────────────────────$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ── Setup ─────────────────────────────────────────────────────
install:  ## Install the package and all dependencies
	pip install -e .
	@echo "$(GREEN)✓ Package installed successfully$(RESET)"

install-dev:  ## Install with dev/test dependencies
	pip install -e ".[dev]"
	pip install ruff pytest pytest-cov httpx
	@echo "$(GREEN)✓ Dev dependencies installed$(RESET)"

clean:  ## Remove build artifacts, cache, and temp files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned up$(RESET)"

# ── Pipeline ──────────────────────────────────────────────────
train:  ## Run the full training pipeline
	python main.py --mode train

predict:  ## Run the prediction pipeline
	python main.py --mode predict

# ── API ───────────────────────────────────────────────────────
serve:  ## Start the FastAPI server (dev mode)
	uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

serve-prod:  ## Start the FastAPI server (production mode)
	uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 2

# ── Testing ───────────────────────────────────────────────────
test:  ## Run all tests
	pytest tests/ -v --tb=short

test-cov:  ## Run tests with coverage report
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# ── Code Quality ──────────────────────────────────────────────
lint:  ## Lint code with ruff
	ruff check src/ tests/ api/

format:  ## Format code with ruff
	ruff format src/ tests/ api/

lint-fix:  ## Auto-fix lint issues
	ruff check --fix src/ tests/ api/

# ── Experiment Tracking ───────────────────────────────────────
mlflow:  ## Start the MLflow tracking server
	mlflow server \
		--host 0.0.0.0 \
		--port 5000 \
		--backend-store-uri sqlite:///mlruns/mlflow.db \
		--default-artifact-root ./mlruns/artifacts

zenml:  ## Initialize ZenML and start dashboard
	zenml init
	zenml up

# ── Docker ────────────────────────────────────────────────────
docker-build:  ## Build the Docker image
	docker build -t retail-ops-forecaster:latest .

docker-run:  ## Run the full stack with Docker Compose
	docker compose -f docker/docker-compose.yml up -d

docker-stop:  ## Stop all Docker containers
	docker compose -f docker/docker-compose.yml down

docker-logs:  ## View container logs
	docker compose -f docker/docker-compose.yml logs -f

# ── Data ──────────────────────────────────────────────────────
download-data:  ## Download Rossmann dataset (requires kaggle CLI)
	@echo "Downloading Rossmann Store Sales dataset..."
	kaggle competitions download -c rossmann-store-sales -p data/raw/
	cd data/raw && unzip -o rossmann-store-sales.zip
	@echo "$(GREEN)✓ Data downloaded to data/raw/$(RESET)"

# ── Monitoring ────────────────────────────────────────────────
drift-check:  ## Run drift detection on latest predictions
	python -m monitoring.drift_detection

dashboard:  ## Open the monitoring dashboard
	python monitoring/dashboard.py
