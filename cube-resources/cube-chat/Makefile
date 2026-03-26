.PHONY: install update test


install:
	uv sync --all-extras


update:
	uv sync --all-extras --upgrade


test:
	uv run pytest tests/
