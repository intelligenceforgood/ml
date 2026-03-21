.PHONY: install dev test lint fmt clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/unit -x

lint:
	ruff check src/ tests/
	black --check src/ tests/

fmt:
	black src/ tests/
	isort src/ tests/
	ruff check --fix src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info src/*.egg-info
