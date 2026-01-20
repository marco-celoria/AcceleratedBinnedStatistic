.PHONY: install test lint format

install:
	uv pip install -e .

test:
	pytest -v --capture=no  tests/

lint:
	flake8 src tests
	mypy src tests
	black --check src tests
	isort --check-only src tests

format:
	black src tests
	isort src tests

