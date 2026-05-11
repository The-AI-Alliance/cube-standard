.PHONY: install update test lint


install:
	uv sync --all-extras


update:
	uv sync --all-extras --upgrade


test:
	uv run pytest tests/


lint:
	uv run ruff check --fix .
	uv run ruff format .
