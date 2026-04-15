.PHONY: help install install-dev lint typecheck format test test-fast coverage clean

# Detect virtual environment
VENV_BIN := $(shell if [ -d ".venv/bin" ]; then echo ".venv/bin/"; else echo ""; fi)
PYTHON := $(VENV_BIN)python
PIP := $(shell if [ -f ".venv/bin/pip" ]; then echo ".venv/bin/pip"; else echo "$(PYTHON) -m pip"; fi)

# Default target
help:
	@echo "Available targets:"
	@echo "  install         - Install the package"
	@echo "  install-dev     - Install the package with development dependencies"
	@echo "  lint            - Run code linters (ruff, mypy, black)"
	@echo "  typecheck       - Run mypy type checker only"
	@echo "  format          - Format code with black and ruff"
	@echo "  test            - Run all tests with pytest"
	@echo "  test-fast       - Run tests excluding slow gradient tests"
	@echo "  coverage        - Run tests with coverage report"
	@echo "  clean           - Remove build artifacts and caches"
	@echo ""
	@echo "Note: If .venv/ exists, it will be used automatically"

# Install the package
install:
	$(PIP) install -e .

# Install with development dependencies
install-dev:
	$(PIP) install -e ".[dev]"

# Run linters
lint:
	@echo "Running ruff..."
	$(PYTHON) -m ruff check smooth_mcp/ tests/ demos/
	@echo "Running mypy..."
	$(PYTHON) -m mypy smooth_mcp/ demos/
	@echo "Checking formatting with black..."
	$(PYTHON) -m black --check smooth_mcp/ tests/ demos/

# Run type checker
typecheck:
	@echo "Running mypy type checker..."
	$(PYTHON) -m mypy smooth_mcp/ demos/

# Format code
format:
	@echo "Formatting with black..."
	$(PYTHON) -m black smooth_mcp/ tests/ demos/
	@echo "Sorting imports with ruff..."
	$(PYTHON) -m ruff check --fix --select I smooth_mcp/ tests/ demos/

# Run all tests
test:
	$(PYTHON) -m pytest tests/ -v

# Run fast tests only (skip slow gradient tests)
test-fast:
	$(PYTHON) -m pytest tests/ -v --ignore=tests/test_gradients.py

# Run tests with coverage
coverage:
	@echo "Running tests with coverage..."
	$(PYTHON) -m pytest --cov=smooth_mcp tests/

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.swp" -delete
	@echo "Clean complete!"
