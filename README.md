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

## MediaPipe y entrenamiento rápido

El módulo de visión procesa la cámara localmente, guarda únicamente landmarks y
permite entrenar un clasificador inicial para probar una seña antes de integrar
el guante:

```bash
source .venv/bin/activate
python -m app.vision collect --label A --participant p01
python -m app.vision collect --label reposo --participant p01
python -m app.vision train
python -m app.vision live --target A
```

En la ventana de recolección, `Espacio` inicia o termina una grabación y `Q`
sale. La primera ejecución descarga el modelo de MediaPipe en
`models/vision/hand_landmarker.task`. Los datos se guardan en
`data/vision/` y están excluidos de Git.

Para comprobar una imagen sin abrir la cámara:

```bash
python -m app.vision extract --image ruta/a/imagen.jpg
```

El contrato de integración con la aplicación móvil está en
[`docs/mobile-api-contract.md`](docs/mobile-api-contract.md).

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
