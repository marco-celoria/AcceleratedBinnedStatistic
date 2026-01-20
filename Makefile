# SPDX-FileCopyrightText: (C) 2026 Marco Celoria <celoria.marco@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

.PHONY: install test lint format

install:
	uv pip install -e .

test:
	pytest -v --capture=no  tests/

lint:
	flake8 src tests benchmarks examples
	mypy src tests benchmarks examples 
	isort --check-only src tests benchmarks examples

format:
	black src tests benchmarks examples
	isort src tests benchmarks examples

