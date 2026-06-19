# ==============================================================================
# Autonomous Code Review Agent - Makefile
# Common development and deployment commands
# ==============================================================================

.PHONY: help dev up down build migrate test lint format clean

# Default target
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Development ---

dev: ## Start full development stack
	docker compose -f docker-compose.yml -f docker/docker-compose.dev.yml up --build

up: ## Start production stack
	docker compose up -d --build

down: ## Stop all services
	docker compose down

build: ## Build Docker images
	docker compose build

logs: ## Tail logs from all services
	docker compose logs -f

logs-api: ## Tail API server logs
	docker compose logs -f api

logs-worker: ## Tail worker logs
	docker compose logs -f worker

# --- Database ---

migrate: ## Run database migrations
	docker compose run --rm migrations

migrate-local: ## Run migrations locally
	alembic upgrade head

migrate-new: ## Create a new migration (usage: make migrate-new msg="description")
	alembic revision --autogenerate -m "$(msg)"

# --- Testing ---

test: ## Run test suite
	pytest tests/ -v --cov=code_review_agent --cov-report=term-missing

test-fast: ## Run tests without coverage
	pytest tests/ -x -q

# --- Code Quality ---

lint: ## Run linter
	ruff check src/ tests/

format: ## Format code
	ruff format src/ tests/

typecheck: ## Run type checker
	mypy src/

check: lint typecheck ## Run all code quality checks

# --- Utilities ---

clean: ## Clean build artifacts and caches
	rm -rf build/ dist/ *.egg-info .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

shell: ## Open a shell in the API container
	docker compose exec api bash

worker-shell: ## Open a shell in the worker container
	docker compose exec worker bash

redis-cli: ## Connect to Redis CLI
	docker compose exec redis redis-cli

psql: ## Connect to PostgreSQL
	docker compose exec postgres psql -U postgres code_review_agent

# --- Installation ---

install: ## Install package in development mode
	pip install -e ".[dev]"

install-hooks: ## Install pre-commit hooks
	pre-commit install
