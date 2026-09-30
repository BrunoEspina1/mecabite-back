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

Los ajustes del reconocimiento (`VISION_HOLD_MS`, `VISION_CONFIDENCE_THRESHOLD`,
`VISION_VOTE_MS`, `VISION_GRACE_MS`, `VISION_WRONG_SIGN_MS`) están explicados en [`.env.example`](.env.example). Se aplican al
reiniciar la API o la CLI; una variable de entorno tiene prioridad sobre el `.env`.

## Run

```bash
uv run fastapi dev app/main.py
```

- API: http://localhost:8000/api/v1/health
- Catálogo: http://localhost:8000/api/v1/catalog/signs
- Docs: http://localhost:8000/docs

Para conectar un iPhone en la misma red Wi-Fi, `fastapi dev` debe escuchar en todas las
interfaces (por defecto solo acepta `127.0.0.1`):

```bash
uv run fastapi dev app/main.py --host 0.0.0.0
ipconfig getifaddr en0     # IP de la laptop para la app: http://<ip>:8000/api/v1
```

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
vision collect --label A --participant p01
vision collect --label reposo --participant p01
vision train               # --only static|dynamic, --model mlp|rf|et|svm
vision demo               # cámara en vivo mostrando la seña reconocida
vision live --target A    # práctica: indica si haces la seña objetivo
```

`vision` queda instalado en el `.venv`; sin activarlo usa `uv run vision demo`.

En `demo` y `live` la seña aparece en el recuadro de la esquina superior derecha:

- **Estáticas**: al sostenerla `VISION_HOLD_MS` (1 s por defecto, como pide RF-06; la
  barra debajo muestra el avance).
- **Con movimiento** (J, K, Ñ, Q, X, Z): al terminar el movimiento y dejar la mano quieta un
  instante; hay hasta 2.5 s para terminarla (2 s para palabras de nivel 3) y se compara la ventana
  de trayectoria de los últimos 2 s. El clip/datos de las palabras nivel 3 conserva 4 s.

`vision train` entrena dos modelos (MLP) y los evalúa dejando fuera personas completas:
`models/vision/static.joblib` y `models/vision/dynamic.joblib`. Las señas vienen de
[`app/catalog/signs.json`](app/catalog/signs.json) (el mismo catálogo que sirve
`GET /catalog/signs`; `data_label` es la carpeta de datos de cada seña) más el resto del
abecedario en [`app/vision/training_signs.json`](app/vision/training_signs.json), que solo
usa visión (la API no lo anuncia). Cualquier otra carpeta de `data/vision/` (por ejemplo
`reposo`) se aprende como `otra` para evitar falsos positivos. Para agregar una seña:
añádela a uno de esos dos archivos, pon sus grabaciones en `data/vision/<seña>/` y reentrena.

Las coordenadas de la mano se corrigen por la proporción de la imagen (los datasets son
cuadrados y una webcam es 16:9; sin esto la mano se ve 44 % más angosta).

En la ventana de recolección, `Espacio` inicia o termina una grabación y `Q`
sale. La primera ejecución descarga el modelo de MediaPipe en
`models/vision/hand_landmarker.task`. Los datos se guardan en
`data/vision/` y están excluidos de Git.

### Práctica guiada (datos propios)

`vision practice` pide seña por seña. Cuando el modelo reconoce la pedida, muestra la
captura (foto con la mano dibujada, o el clip en bucle si la seña es con movimiento) para
aprobarla y pasa a la siguiente:

```bash
vision practice --participant omar                  # todas las señas que conoce el modelo
vision practice --participant omar --signs N,U,X --reps 3
vision practica_guante --participant omar --signs A,B,C --reps 3
vision train                                        # aprende de lo guardado
vision clear                                        # borra todo lo de práctica (-y sin preguntar)
```

| Tecla | En vivo | Al revisar la captura |
|---|---|---|
| `Espacio` | guardar aunque el modelo no la reconozca | guardar |
| `Enter` | | guardar |
| `R` | | descartar y repetir la seña |
| `N` | saltar la seña | saltar la seña |
| `Q` | salir | salir |

Se guarda en `data/practice/` (fuera de Git):

- `landmarks/<seña>/*.csv`: puntos de la mano y del cuerpo por cuadro (lo que usa `train`,
  evaluado como otra persona).
- `captures/<seña>/*.jpg|.mp4`: la foto (JPEG calidad 95) o el clip, **sin dibujos
  encima**, para poder volver a extraer puntos con otros modelos sin volver a grabar.
- `captures/<seña>/*.json`: datos del momento (persona, resultado, predicción y confianza,
  encuadre, zona de la mano, resolución, versión de los modelos).
- `sesiones.csv`: cada intento (reconocida, forzada, descartada o saltada) y cuánto tardó.

Las forzadas son las más valiosas: son justo los casos en que el modelo falla contigo.
`vision clear` solo borra `data/practice/`; los datasets de `data/vision/` no se tocan.

`vision practica_guante` requiere el receptor BLE encendido. Guarda los landmarks en
`data/practice/landmarks/<seña>/` y los paquetes crudos compatibles con `manitas` en
`data/practice/glove/<seña>/`. La configuración BLE está en `GLOVE_DEVICE_NAME` y
`GLOVE_CHARACTERISTIC_UUID` del `.env`. En señas estáticas, al mantener la seña reconocida un
segundo aparece una barra de progreso y se toma la captura automáticamente. La revisión muestra
una tabla con los paquetes BLE recibidos durante ese segundo; las señas con movimiento se capturan
al completar la trayectoria. Sus videos y datos BLE guardan 2000 ms para letras con movimiento y
4000 ms para palabras del nivel 3; la tabla presenta 10 filas por página, una muestra cada 200 ms,
con timestamps completos y marca `OK` para cada paquete validado. En nivel 3 se puede avanzar de
página para revisar los 4 segundos. El CSV conserva todos los paquetes.
`--no-review` guarda sin mostrar la captura.

**Mano principal y registro de entradas.** `demo`, `live`, `practice` y `practica_guante` aceptan:

- `--hand derecha|izquierda`: con las dos manos a la vista, la principal (la que se reconoce y se
  compara con los modelos) es esa; con una sola se usa la que se ve. Sin la opción, la principal es
  la primera mano que aparece.
- `--log`: imprime dos veces por segundo las dos manos y los dos guantes que llegan. En `demo` y
  `live` agrega `--glove` para conectar los guantes (en `practica_guante` ya están conectados).

```bash
vision live --target gracias --hand izquierda --log --glove
vision practica_guante --participant omar --signs A,B --hand derecha --log
```

```text
[12:03:04.512] manos: principal=derecha (A 0.93) · otra=izquierda | guante der E1: dedos 1-3-2-1-1 roll -4 pitch -43 | guante izq E2: sin datos
```

En la API (app), `INPUT_LOG=true` imprime la misma línea por cada sesión, más el estado y el mensaje. Las capturas incluyen cara y cuerpo: son datos
personales, solo para desarrollo, con consentimiento de quien graba (en la app el video no se
guarda, RNF-10).

### Correcciones con guante y cámara (RF-13)

La práctica por API dice qué corregir (`corrections` en el `feedback`): "Estira más el dedo
anular", "Gira la palma hacia la cámara", "Inclina la mano hacia abajo", "Sube más el brazo".
Cómo debe verse cada seña está en `expected` de [`app/catalog/signs.json`](app/catalog/signs.json)
(dedos del guante, dedos y orientación con la cámara). Para usar el guante en la API:

```bash
GLOVE_LIVE=true GLOVE_REQUIRED=true uv run fastapi dev app/main.py --host 0.0.0.0
vision glove-reference                  # inclinación por seña con las grabaciones de practica_guante
vision glove-reference --live --sign a  # o midiendo en vivo con la seña bien hecha
```

`glove-reference` también avisa si los dedos grabados no coinciden con `expected.glove_fingers`.

### Cuerpo: encuadre y localización (RF-03)

`demo`, `live`, `practice` y `collect` detectan también el cuerpo (MediaPipe Pose Landmarker
lite, 33 puntos: cara, hombros, brazos) y muestran una línea de encuadre:

- **Encuadre**: de frente, con cara y ambos hombros a la vista, el pecho dentro de la imagen,
  centrado y a buena distancia; si no, dice qué corregir. En `practice` una seña solo se
  guarda bien encuadrada y en `live --target` solo cuenta como intento correcto así.
- **Localización**: zona donde está la palma medida en anchos de hombro (frente, cara,
  boca/barbilla, lado de la cara, hombro, pecho, lejos del cuerpo). Las letras (niveles 1 y
  2) no la evalúan; queda lista para las palabras del nivel 3.

Los umbrales están en [`app/vision/body.py`](app/vision/body.py). `--no-body` lo desactiva.

### Dataset de videos MSL-dynamic-signs

Los videos de señas dinámicas (J, K, Q, X, Z, Ñ) viven en `data/raw/` (excluido de Git):

- `data/raw/MSL-dynamic-signs/{train,test}/` — vista frontal (622 videos)
- `data/raw/MSL-dynamic-signs-profile/{J,K,Q,X,Z,Ñ}/` — vista de perfil (620 videos)

`import-videos` pasa cada video por MediaPipe y guarda los landmarks en
`data/vision/<letra>/` con el mismo formato que `collect`, así que `train` los
usa directamente junto con lo que grabes con la webcam:

```bash
vision import-videos --source data/raw/MSL-dynamic-signs
vision import-videos --source data/raw/MSL-dynamic-signs-profile
vision train
```

Para videos propios con nombres distintos al formato MSL, indica la etiqueta y la persona;
por ejemplo: `vision import-videos --source data/raw/mama --label mama --participant mama01`.

Las letras estáticas (A–Y) vienen de MSL-ABC, copiadas como landmarks
(`msl-abc__S<persona>.csv`) desde el proyecto Manitas.

`train` agrega cada cuadro (y cada trayectoria) también en espejo, así el modelo
reconoce la mano derecha y la izquierda sin depender de la etiqueta de mano de
MediaPipe (que falla seguido). La evaluación separa por persona (`S1`, `S2`, ...).

Los videos ya importados se saltan (usa `--overwrite` para regenerarlos) y
`--stride N` procesa 1 de cada N cuadros (por defecto 1).

Para comprobar una imagen sin abrir la cámara:

```bash
vision extract --image ruta/a/imagen.jpg
```

El contrato de integración con la aplicación móvil está en
[`docs/mobile-api-contract.md`](docs/mobile-api-contract.md).

El estado de cada requerimiento, las actividades pendientes y la bitácora están en
[`docs/estado-requerimientos.md`](docs/estado-requerimientos.md). Actualízalo al terminar
cualquier requerimiento.

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
  catalog/signs.json   # Catálogo de señas: ids, niveles, componentes (RF-09, RNF-11)
  vision/              # MediaPipe, entrenamiento y prueba en vivo
  vision/recognizer.py # Reconocimiento sin cámara (lo usan la CLI y, pronto, la API)
tests/
```
