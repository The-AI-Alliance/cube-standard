.PHONY: help install ci-install update format lint lint-check test

help:
	@echo "make install    - Install dependencies in editable mode"
	@echo "make ci-install - Install dependencies with locked versions (for CI)"
	@echo "make update     - Update dependencies"
	@echo "make format     - Format code"
	@echo "make lint       - Lint and auto-fix"
	@echo "make lint-check - Check linting without fixing"
	@echo "make test       - Run unit tests"

install:
	uv sync --all-extras --all-packages

ci-install:
	uv sync --frozen --all-extras --all-packages

update:
	uv sync --all-extras --upgrade

lint:
	uv run ruff check --fix .
	uv run ruff format .

lint-check:
	uv run ruff check --diff .
	uv run ruff format --diff .

test:
	uv run pytest -n 10
