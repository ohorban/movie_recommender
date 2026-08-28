# movierec — everything runs inside a local .venv, which is created on demand.
# No `source .venv/bin/activate` needed: targets call the venv's binaries directly.

VENV           ?= .venv
PYTHON_VERSION ?= 3.12
PY             := $(VENV)/bin/python
UVPIP          := uv pip install --python $(PY)
MOVIEREC       := $(VENV)/bin/movierec
STREAMLIT      := $(VENV)/bin/streamlit

.DEFAULT_GOAL := help
.PHONY: help venv install dev-install setup update app test lint fmt typecheck doctor clean rebuild distclean

help:  ## Show this help
	@echo "movierec — make targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "First run:  make dev-install && make setup && make app"

# --- environment -----------------------------------------------------------
# `uv venv` writes .venv/bin/python, so make can treat it as a file target and
# skip this entirely once the environment exists.
$(PY):
	@command -v uv >/dev/null 2>&1 || { \
		echo "uv is not installed. Install it with:"; \
		echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		echo "then restart your shell (or: export PATH=\"$$HOME/.local/bin:$$PATH\")."; \
		exit 1; }
	uv venv --python $(PYTHON_VERSION) $(VENV)

venv: $(PY)  ## Create the virtual environment only

# Reinstalls whenever pyproject.toml changes; the console script is the marker.
$(MOVIEREC): pyproject.toml | $(PY)
	$(UVPIP) -e ".[embed,app]"
	@touch $(MOVIEREC)

install: $(MOVIEREC)  ## Install with app + embedding extras

dev-install: | $(PY)  ## Install everything, including dev tooling
	$(UVPIP) -e ".[embed,app,dev]"

# --- pipeline --------------------------------------------------------------
setup: $(MOVIEREC)  ## First-time build: catalog, ingest, embed, train
	$(MOVIEREC) setup

update: $(MOVIEREC)  ## Incremental update from the newest Letterboxd export
	$(MOVIEREC) update

rebuild: $(MOVIEREC)  ## Destroy and rebuild the database (keeps manual match fixes)
	$(MOVIEREC) rebuild --yes

app: $(MOVIEREC)  ## Launch the web interface
	@test -x $(STREAMLIT) || { \
		echo "streamlit is not installed in $(VENV). Run:  make dev-install"; exit 1; }
	$(STREAMLIT) run app/streamlit_app.py

doctor: $(MOVIEREC)  ## Check the environment, keys and database
	@$(PY) -c "import sys; print('python     ', sys.version.split()[0])"
	@$(PY) -c "import movierec; print('movierec   ', movierec.__version__)"
	@$(PY) -c "import importlib.util as u; print('embeddings ', 'sentence-transformers ok' if u.find_spec('sentence_transformers') else 'MISSING -> make dev-install')"
	@$(PY) -c "import importlib.util as u; print('web ui     ', 'streamlit ok' if u.find_spec('streamlit') else 'MISSING -> make dev-install')"
	@$(PY) -c "from movierec.config import load_config as c; m=c().missing_credentials(); print('keys       ', 'all set' if not m else 'MISSING: '+', '.join(m))"
	@$(MOVIEREC) status

# --- development -----------------------------------------------------------
test: | $(PY)  ## Run the test suite
	$(VENV)/bin/pytest

lint: | $(PY)  ## Lint and check formatting
	$(VENV)/bin/ruff check src tests app
	$(VENV)/bin/ruff format --check src tests app

fmt: | $(PY)  ## Auto-format and apply safe fixes
	$(VENV)/bin/ruff format src tests app
	$(VENV)/bin/ruff check --fix src tests app

typecheck: | $(PY)  ## Static type check
	$(VENV)/bin/mypy src/movierec

clean:  ## Remove caches (keeps the database and the venv)
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

distclean: clean  ## Also remove the virtual environment
	rm -rf $(VENV)
