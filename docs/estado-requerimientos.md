# Estado De Los Requerimientos

Control global de [`Requirements.md`](../Requirements.md): qué está hecho, qué falta y qué
sigue. Es la referencia común para todas las personas y sesiones que trabajan en el repo.

**Última actualización:** 2026-09-29

## Cómo Mantenerlo

Al terminar algo, en el mismo cambio:

1. Actualiza la fila del requerimiento: estado, qué quedó hecho (con el archivo o test que
   lo demuestra) y qué falta.
2. Mueve la actividad de [Pendientes](#pendientes) a la [Bitácora](#bitácora) con la fecha.
3. Si se tomó una decisión, anótala en [Decisiones](#decisiones).
4. Cambia la fecha de "Última actualización".

Estados:

| Estado | Significado |
| --- | --- |
| ✅ Hecho | Cumple el requerimiento en el MVP y hay evidencia (código y test) |
| 🟡 Parcial | Existe una parte; la columna "Falta" dice qué |
| ⬜ Pendiente | No se ha empezado |
| ⛔ Bloqueado | Depende de algo externo (guante, intérprete, móvil) |
| ➖ Fuera del MVP | Aplica a la versión final |

"Webcam" significa que funciona en la CLI `vision` con la cámara de la laptop. "API" significa
que funciona con datos del iPhone vía REST o WebSocket, que es lo que pide el MVP.

## Contexto Del MVP

- El iPhone corre MediaPipe y envía landmarks; el backend hace todo el reconocimiento y
  responde por WebSocket. Se prueba en red local.
  Contrato: [`mobile-api-contract.md`](mobile-api-contract.md).
- El guante se está terminando de construir: todo lo que dependa de él se prueba solo con
  datos simulados.
- Catálogo de la app (15 señas): [`app/catalog/signs.json`](../app/catalog/signs.json).
  Letras extra solo para entrenar: [`app/vision/training_signs.json`](../app/vision/training_signs.json).

## Requerimientos Funcionales

| ID | Requerimiento | Estado | Hecho | Falta |
| --- | --- | --- | --- | --- |
| RF-01 | Recibir datos del guante en tiempo real | ⛔ Bloqueado | Formato preliminar de 11 valores en el contrato | Specs reales del guante (electrónica); lectura y validación en el backend |
| RF-02 | Calibración guiada y recalibrar | 🟡 Parcial | Encuadre de cámara: dice qué corregir (`body.framing_issue`, `tests/test_body.py`) | Encuadre vía API; calibración del guante (bloqueada); flujo "antes de la primera lección" |
| RF-03 | MediaPipe de mano y cuerpo para localizar la seña | 🟡 Parcial | Webcam: Hand + Pose Landmarker lite (33 puntos) y zona de la palma (frente, cara, boca/barbilla, lado de la cara, hombro, pecho) en `app/vision/body.py`, `tracker.py`; tests en `tests/test_body.py` | Recibir `pose_landmarks` del iPhone por la API; validar zonas con intérprete |
| RF-04 | Combinar guante y cámara | ⛔ Bloqueado | — | Guante; definir fusión por componente |
| RF-05 | Detectar y evaluar movimiento | 🟡 Parcial | Webcam: `DynamicDetector` detecta inicio y fin por velocidad y clasifica la trayectoria (`app/vision/sequence.py`) | Por API; reportar el componente `movement` |
| RF-06 | Estática válida al sostenerla ≥ 1 s | 🟡 Parcial | `SignStabilizer` en `app/vision/recognizer.py`, sin cámara y con el tiempo de cada cuadro (`tests/test_recognizer.py`). Tiempo configurable con `VISION_HOLD_MS`: 1000 por defecto; la API lo reporta como `hold_time_ms` | **El `.env` local usa 200 ms**; el requerimiento pide 1 s: decisión pendiente. Con 0.2 s se confirman formas intermedias de las dinámicas (I al inicio de la J, L al inicio de Z/Q/X, N al inicio de la Ñ). Por API |
| RF-07 | Dinámica: trayectoria completa en ≤ 3 s | 🟡 Parcial | Modelo dinámico sobre ventana de 2 s; `Recognizer` lo combina con el estático sin cámara. Con videos grabados reconoce J, Z, Q, X y Ñ | Por API; decidir si un clasificador cuenta como "patrón de referencia" o se agregan plantillas al catálogo |
| RF-08 | Evaluar solo los componentes del nivel | 🟡 Parcial | Niveles y componentes definidos en el catálogo y en `GET /catalog/signs` | Que el motor evalúe por nivel. El texto del nivel 3 es ambiguo ("se agregan localización y movimiento") |
| RF-09 | Catálogo exclusivo de 15 señas en 3 niveles | 🟡 Parcial | `signs.json` + `GET /catalog/signs` (`tests/test_catalog.py`); letras extra fuera de la API | **No hay datos de las 5 palabras**: el nivel 3 no se reconoce |
| RF-10 | Aprobada con 3 correctas consecutivas | 🟡 Parcial | Webcam: `live --target` cuenta intentos (el conteo aún vive en `teacher.py`) | Por API, con estado `approved` |
| RF-11 | Video de referencia y descripción de componentes | 🟡 Parcial | Campos en el catálogo; borradores de 7 señas (`validated: false`) | Descripciones de Ñ, Q, X y palabras; grabar los 15 videos; validación del intérprete |
| RF-12 | Modo práctica en tiempo real | 🟡 Parcial | Webcam: `vision live --target`, `vision practice` | `POST /sessions` + WebSocket (modo `practice`) |
| RF-13 | Indicar qué componente falló | 🟡 Parcial | Webcam: `live --target` y `practice` rechazan una seña correcta sin buen encuadre y dicen qué corregir (`body.framing_issue`) | Evaluar por separado configuración, orientación (normal de la palma), movimiento y localización |
| RF-14 | Modo demostración | 🟡 Parcial | Webcam: `vision demo` | Modo `demo` por API |
| RF-15 | Entrada de datos simulados por consola (array) | ⬜ Pendiente | Formato en el contrato (`POST /simulations/predict`) | `vision simulate` + endpoint + evaluador por reglas del catálogo |

## Requerimientos No Funcionales

| ID | Requerimiento | Estado | Hecho | Falta |
| --- | --- | --- | --- | --- |
| RNF-01 | Resultado en < 500 ms | 🟡 Parcial | `processing_time_ms` definido en el contrato. `Recognizer` con los modelos reales: 0.2–0.4 ms por cuadro en promedio, 6.4 ms el peor caso (sin contar MediaPipe ni red). Una estática se confirma ~0.6 s después de empezar a hacerla | Medir ida y vuelta real iPhone ↔ backend |
| RNF-02 | MediaPipe ≥ 20 fps en teléfono de gama media | ⬜ Pendiente | Referencia en la Mac: mano ~4 ms y cuerpo ~7 ms por cuadro a 1920×1080 | Lo mide móvil; ahora corre Hand + Pose a la vez |
| RNF-03 | Exactitud 90 / 80 / 75 % por nivel con usuarios nuevos | 🟡 Parcial | `vision train` evalúa dejando fuera personas completas (`evaluate_by_person`) | Medir por intento (no por cuadro) y por nivel; nivel 3 sin datos |
| RNF-04 | Falsos positivos < 10 % | 🟡 Parcial | `evaluate_by_person` reporta falsos positivos; clase `otra` y abecedario extra como negativos | Medir por intento, incluyendo reposo y transiciones |
| RNF-05 | Patrones validados por intérprete de LSM | ⛔ Bloqueado | — | Contactar a la persona intérprete |
| RNF-06 | iOS 18 o superior | ⬜ Pendiente | — | Lado móvil |
| RNF-07 | Reconocimiento sin internet | ➖ Fuera del MVP | — | Versión final |
| RNF-08 | Calibración + primera seña en < 5 min sin ayuda | ⬜ Pendiente | — | Prueba de usabilidad con la app |
| RNF-09 | Interfaz en español con apoyo visual | 🟡 Parcial | `message` en español y `feedback_code` definidos en el contrato | Implementar en la API; UI en móvil |
| RNF-10 | Video procesado localmente, sin guardar ni enviar | 🟡 Parcial | Arquitectura: solo viajan landmarks | Nota: `vision practice` guarda fotos y clips en `data/practice/captures/` (fuera de Git, solo desarrollo, con consentimiento). La app no debe guardar video |
| RNF-11 | Catálogo y patrones separados del código | ✅ Hecho | `app/catalog/signs.json` y `app/vision/training_signs.json`; agregar una seña no toca el clasificador (`tests/test_catalog.py`) | Patrones de trayectoria o zonas, si se agregan al catálogo |

## Pendientes

En orden de prioridad para el MVP.

| # | Actividad | Área | Requerimientos | Depende de |
| --- | --- | --- | --- | --- |
| 1 | Hacer commit del trabajo actual (hay más de 15 archivos sin commit) | Repo | — | — |
| 4 | Convertir las observaciones del iPhone al formato de entrenamiento (proporción, espejo, mano) | Backend | RF-03 | 3; JSON real de móvil |
| 5 | Sesión de práctica y demo: estados, 3 consecutivas, `feedback_code`, ids del catálogo | Backend | RF-10, RF-12, RF-14 | 3 |
| 6 | `POST /sessions` + WebSocket | Backend | RF-12, RF-14, RNF-01 | 4, 5 |
| 7 | Script que simula al iPhone reproduciendo grabaciones por WebSocket | Backend | RNF-01 | 6 |
| 8 | `vision simulate` + `POST /simulations/predict` con evaluador por reglas | Backend | RF-15, RF-01 | 5 |
| 9 | Orientación explícita (normal de la palma) para reportar el componente | Visión | RF-13 | — |
| 10 | Evaluación por intento y por nivel | Visión | RNF-03, RNF-04 | — |
| 11 | Datos de las 5 palabras (buscar dataset o grabar con mano + cuerpo) | Visión | RF-09, RF-03 | Intérprete |
| 12 | Encuadre y localización por API (`pose_landmarks`) | Backend | RF-02, RF-03 | 6 |
| 13 | Contrato: `pose_landmarks` en **todos** los niveles (hoy dice `null` en 1 y 2); móvil ya lo implementa | Contrato + Móvil | RF-02, RF-03 | — |

### Bloqueos Externos Y Decisiones Abiertas

| Tema | Quién | Bloquea |
| --- | --- | --- |
| Estáticas: confirmar a 0.2 s (`.env` local) o a 1 s (valor por defecto, RF-06). Con 0.2 s aparecen formas intermedias de las dinámicas como señas confirmadas | Equipo | RF-06 |
| Specs del guante: unidades, rangos, frecuencia, conexión | Electrónica | RF-01, RF-02, RF-04, RF-15 |
| Persona intérprete de LSM | Equipo | RNF-05, RF-11, nivel 3 |
| Confirmar contrato v0.2.0 y enviar un JSON de observación real | Móvil | Pendiente 4 |
| Aclarar el texto de RF-08 para el nivel 3 | Equipo | RF-08 |

## Decisiones

| Fecha | Decisión |
| --- | --- |
| 2026-09-29 | En el MVP el iPhone corre MediaPipe y el backend procesa todo por API + WebSocket en red local. RNF-07 aplica solo a la versión final |
| 2026-09-29 | Catálogo único en `app/catalog/signs.json` (lo usan la API y visión) |
| 2026-09-29 | El resto del abecedario (16 estáticas + K) se entrena como extra en `app/vision/training_signs.json`; la API solo anuncia las 15 señas |
| 2026-09-29 | Contrato móvil v0.2.0: `hands` como lista, `image_width`/`image_height`/`mirrored`, `glove: null` en el MVP |
| 2026-09-29 | El iPhone enviará `pose_landmarks` en todos los niveles: el backend lo usa para validar el encuadre (de frente, cara y hombros visibles) |

## Bitácora

| Fecha | Qué se hizo | Requerimientos |
| --- | --- | --- |
| 2026-09-29 | Ajustes del reconocimiento en `.env` (`VISION_HOLD_MS`, `VISION_CONFIDENCE_THRESHOLD`, `VISION_VOTE_MS`, documentados en `.env.example`). `hold_time_ms` del catálogo sale del mismo valor | RF-06, RNF-11 |
| 2026-09-29 | `Recognizer` y `SignStabilizer` pasan a `app/vision/recognizer.py`: sin cámara, todo con el `t_ms` de cada cuadro y votos por tiempo (400 ms) en vez de 12 cuadros, así se comporta igual a 30 fps (webcam) que a 15 Hz (iPhone). `teacher.py` solo dibuja. 43 tests pasan | RF-06, RF-07, RNF-01 |
| 2026-09-29 | Grabaciones propias pesan ×5 al entrenar (`OWN_WEIGHT`) y `vision train` reporta el acierto con ellas sin haberlas visto (hoy: estático 91.0 %, dinámico 74.4 %, con 1 repetición por letra) | RNF-03 |
| 2026-09-29 | `vision train` guarda los modelos de forma atómica y Ctrl+C cancela sin dejar un modelo a medias; reentrenado con los datos de `practice` (estático 93.8 %, dinámico 93.4 %) | RNF-03 |
| 2026-09-29 | `test_load_static_dataset_adds_mirrored_copy` aislado de `data/practice/` (33 tests pasan) | — |
| 2026-09-29 | Cuerpo con Pose Landmarker: encuadre y zona de la mano (`app/vision/body.py`) | RF-03, RF-02 |
| 2026-09-29 | `vision practice` y `vision clear`: práctica guiada que guarda datos para reentrenar | RF-12 |
| 2026-09-29 | Modelos estático y dinámico separados, evaluación dejando fuera personas completas, 27 letras | RF-05, RF-07, RNF-03, RNF-04 |
| 2026-09-29 | Catálogo con niveles y componentes, y `GET /catalog/signs` | RF-08, RF-09, RF-11, RNF-11 |
| 2026-09-29 | Contrato móvil v0.2.0 con guía de conexión en red local | RF-12, RNF-10 |
