.PHONY: help venv install test test-cov lint typecheck check pre-commit

VENV_DIR      ?= venv
PYTHON        ?= python3

.DEFAULT: help

help:
	@echo "Make Help for timeseries-zarr-edf"
	@echo ""
	@echo "make venv       - create virtual environment and install all dependencies"
	@echo "make install    - install dependencies into existing venv"
	@echo "make pre-commit - install pre-commit hooks"
	@echo "make test       - run tests"
	@echo "make test-cov   - run tests with coverage report"
	@echo "make lint       - run linter + formatter with auto-fix"
	@echo "make typecheck  - run mypy --strict"
	@echo "make check      - run lint check, typecheck, and tests"

venv:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -e ".[test,dev]"
	@echo ""
	@echo "Virtual environment created. Activate with:"
	@echo "  source $(VENV_DIR)/bin/activate"

install:
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -e ".[test,dev]"

test:
	$(VENV_DIR)/bin/python -m pytest tests/ -v

test-cov:
	$(VENV_DIR)/bin/python -m pytest tests/ -v --cov=timeseries_zarr_edf --cov-report=term-missing

lint:
	$(VENV_DIR)/bin/ruff check --fix timeseries_zarr_edf/ tests/
	$(VENV_DIR)/bin/ruff format timeseries_zarr_edf/ tests/

typecheck:
	$(VENV_DIR)/bin/mypy timeseries_zarr_edf/

check:
	$(VENV_DIR)/bin/ruff check timeseries_zarr_edf/ tests/
	$(VENV_DIR)/bin/ruff format --check timeseries_zarr_edf/ tests/
	$(VENV_DIR)/bin/mypy timeseries_zarr_edf/
	$(VENV_DIR)/bin/python -m pytest tests/

pre-commit:
	$(VENV_DIR)/bin/pre-commit install
