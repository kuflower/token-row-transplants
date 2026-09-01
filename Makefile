PYTHON ?= python
PYTHON_BOOTSTRAP ?= python3.12
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
PYTHON_DIRS := src tests experiments scripts
NOTEBOOKS := notebooks/results_overview.ipynb

.PHONY: \
  check environment environment-check \
  format format-check lint test \
  typecheck verify-results \
  notebook-check notebook-normalize notebook-refresh notebook-render \
  analysis-figures analysis-figures-check

environment:
	$(PYTHON_BOOTSTRAP) scripts/check_environment.py --python-only
	$(PYTHON_BOOTSTRAP) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade "pip==26.2.1"
	$(VENV_PYTHON) -m pip install -r requirements-ci.txt
	$(VENV_PYTHON) -m pip install -e . --no-deps
	$(VENV_PYTHON) scripts/check_environment.py
	$(VENV_PYTHON) -m pip check

environment-check:
	$(PYTHON) scripts/check_environment.py
	$(PYTHON) -m pip check

test: environment-check
	$(PYTHON) -m pytest -q

lint: environment-check
	$(PYTHON) -m ruff check $(PYTHON_DIRS)

format: environment-check
	$(PYTHON) -m black $(PYTHON_DIRS) $(NOTEBOOKS)

format-check: environment-check
	$(PYTHON) -m black --check $(PYTHON_DIRS) $(NOTEBOOKS)

notebook-check: environment-check
	$(PYTHON) -m nbconvert --to notebook --execute \
		--ExecutePreprocessor.timeout=120 --output-dir build/notebooks \
		$(NOTEBOOKS)

notebook-normalize: environment-check
	$(PYTHON) scripts/normalize_notebook.py $(NOTEBOOKS)

notebook-refresh: environment-check
	$(PYTHON) scripts/render_analysis_figures.py
	$(PYTHON) -m nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=120 $(NOTEBOOKS)
	$(MAKE) notebook-normalize
	$(PYTHON) -m black $(NOTEBOOKS)

notebook-render: notebook-check
	$(PYTHON) -m nbconvert --to html --output-dir build/notebooks \
		build/notebooks/results_overview.ipynb

analysis-figures: environment-check
	$(PYTHON) scripts/render_analysis_figures.py

analysis-figures-check: environment-check
	$(PYTHON) scripts/render_analysis_figures.py --check

verify-results: analysis-figures-check notebook-check

typecheck: environment-check
	$(PYTHON) -m mypy

check: test lint format-check typecheck verify-results
