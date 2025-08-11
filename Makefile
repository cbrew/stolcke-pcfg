PY = uv run

.PHONY: setup venv sync run lint fmt test coverage precommit hooks docs-serve docs-build docs-deploy clean

setup: venv sync hooks ## Create venv, install deps, install hooks

venv: ## Create a local virtual environment via uv
	uv venv

sync: ## Install project and dev dependencies via uv
	uv sync

run: ## Run the CLI entry point
	$(PY) stolcke-parser

lint: ## Lint with Ruff
	$(PY) ruff check .

fmt: ## Format with Ruff
	$(PY) ruff format .

test: ## Run tests
	$(PY) pytest -q

coverage: ## Coverage report
	$(PY) pytest --cov=stolcke_pcfg --cov-report=term-missing

precommit: ## Run all pre-commit hooks on the repo
	$(PY) pre-commit run --all-files --show-diff-on-failure

hooks: ## Install pre-commit hooks
	$(PY) pre-commit install

clean: ## Remove caches and build artifacts
	rm -rf .ruff_cache .pytest_cache .coverage htmlcov build dist *.egg-info

docs-serve: ## Serve docs locally with MkDocs
	$(PY) mkdocs serve

docs-build: ## Build static docs site
	$(PY) mkdocs build --strict

docs-deploy: ## Deploy docs to GitHub Pages (gh-pages branch)
	$(PY) mkdocs gh-deploy --force --no-history
