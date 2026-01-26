.PHONY: fmt fmt-check lint autoformat

default: autoformat

# Code Style
fmt:
	uv run lint ruff format .

fmt-check:
	uv run ruff format --check .

lint:
	uv run ruff check .

autoformat:
	uv run ruff format . && uv run ruff check --fix .