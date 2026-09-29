# mecabite-back

FastAPI backend for Mecabite.

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
cp .env.example .env
```

## Run

```bash
uv run fastapi dev app/main.py
```

- API: http://localhost:8000/api/v1/health
- Docs: http://localhost:8000/docs

## Tests and linting

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

## Docker

```bash
docker build -t mecabite-back .
docker run --env-file .env -p 8000:8000 mecabite-back
```

## Structure

```
app/
  main.py              # App factory, middleware, routers
  core/config.py       # Settings loaded from env / .env
  api/v1/router.py     # Aggregates v1 routes
  api/v1/routes/       # One module per resource
tests/
```
