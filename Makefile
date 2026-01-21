.PHONY: help install format lint run

help:
	@echo "make install    - Install dependencies in editable mode"
	@echo "make format     - Format code"
	@echo "make lint       - Lint and auto-fix"
	@echo "make run        - Run the CUBE server"

install:
	uv sync --all-extras
	uv pip install -e .

format:
	uv run ruff format .

lint:
	uv run ruff check --fix .

run:
	uv run python -m cube