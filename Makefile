.PHONY: check
check:
	uv run ruff format --check
	uv run ruff check
	uv run pyright
	uv run pytest -m "not network"
