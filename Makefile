.PHONY: help install dev-install setup update app test lint fmt typecheck clean rebuild

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with app + embedding extras
	uv pip install -e ".[embed,app]"

dev-install:  ## Install everything including dev tooling
	uv pip install -e ".[embed,app,dev]"

setup:  ## First-time build: schema, catalog download, ingest, embed, train
	movierec setup

update:  ## Incremental update from the newest Letterboxd export
	movierec update

app:  ## Launch the Streamlit interface
	streamlit run app/streamlit_app.py

test:  ## Run the test suite
	pytest

lint:  ## Lint and check formatting
	ruff check src tests app
	ruff format --check src tests app

fmt:  ## Auto-format
	ruff format src tests app
	ruff check --fix src tests app

typecheck:  ## Static type check
	mypy src/movierec

clean:  ## Remove caches (keeps the database)
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

rebuild:  ## Destroy and rebuild the database from scratch
	movierec rebuild --yes
