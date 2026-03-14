.PHONY: install update test


install:
	uv sync --all-extras
	uv pip install -e .
	uv run playwright install chromium --with-deps


update:
	uv sync --all-extras --upgrade
	uv run playwright install chromium --with-deps


test:
	uv run pytest tests/
