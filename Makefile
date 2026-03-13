.PHONY: help install format lint run test

help:
	@echo "make install    - Install dependencies in editable mode"
	@echo "make format     - Format code"
	@echo "make lint       - Lint and auto-fix"
	@echo "make run        - Run the CUBE server"

install:
	uv sync --all-extras

lint:
	uv run ruff check --fix .
	uv run ruff format .

lint-check:
	uvx ruff check --diff .
	uvx ruff format --diff .

run:
	uv run python -m cube

test:
	uv run pytest tests/
