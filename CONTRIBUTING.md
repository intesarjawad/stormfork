# Contributing

Read the [engineering standards](docs/engineering-standards.md) before changing code.
Use Python 3.12 and uv 0.13.7:

```sh
uv sync --frozen
uv run pre-commit install
make check
```

For the site, install Node.js 22 or newer and pnpm, then run:

```sh
(cd site && pnpm install --frozen-lockfile && pnpm check && pnpm build)
```

The browser suite runs with `cd site && pnpm test:visual` and requires Docker.
Hooks run the tracked configuration before commits; `make check` runs the full
local Python checks.

Keep network access outside the pure parsers. Tests use recorded public data and must pass without credentials. Include fixture source URLs and retrieval times when updating data. Do not upload user data or add paid services.

Use conventional commit subjects and describe the behavior changed and checks run. The project is named Stormfork.
