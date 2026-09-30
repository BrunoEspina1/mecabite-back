# Contrato De Integración Con La Aplicación Móvil

## Estado Del Documento

- **Versión del protocolo:** 0.2.0 (MVP).
- **Alcance:** qué envía la app, qué responde el backend y cómo conectarse en red local.
- **Idea central:** la app corre MediaPipe y envía landmarks; el backend hace todo el
  reconocimiento y devuelve la retroalimentación.

| Endpoint | Estado |
| --- | --- |
| `GET /health` | Implementado |
| `GET /catalog/signs` | Implementado |
| `POST /sessions` y WebSocket `/ws/sessions/{id}` | Implementado |
| `POST /simulations/predict` | Pendiente |

Lo pendiente ya tiene su formato definido aquí; la app puede construirse contra este
contrato con datos simulados mientras se implementa.

### Cambios Dentro De 0.2.x (compatibles)

- `POST /sessions` y el WebSocket ya están implementados (`app/api/v1/routes/sessions.py`).
- `pose_landmarks` se envía **en todos los niveles**: el backend lo usa para validar que la
  persona esté de frente, con cara y hombros a la vista. Una ejecución correcta sin buen
  encuadre no cuenta. `required_inputs` de `ready` es `["hand", "pose"]`.
- Nuevo `feedback_code`: `adjust_framing` (el `message` dice qué corregir).
- Nuevos errores de `POST /sessions`: `503 MODEL_NOT_AVAILABLE` y `503 SIGN_NOT_TRAINED`.

### Cambios Respecto A 0.1.0

- En el MVP **el backend procesa todo**. Se elimina el modo `local` / `remote_debug`; la
  versión final sin internet (RNF-07) queda fuera del MVP.
- `vision.hand_landmarks` se reemplaza por `vision.hands` (lista) y se agregan
  `image_width`, `image_height` y `mirrored`. Sin estos datos el backend no puede comparar
  los landmarks del teléfono con los de entrenamiento.
- `handedness` pasa a ser la etiqueta **cruda** de MediaPipe con su score.
- Se quita `detection_confidence`: MediaPipe Hand Landmarker no la entrega por mano.
- `glove` va en `null`: el guante aún no está terminado.
- Frecuencia: se envía cada cuadro que procese MediaPipe (15 Hz mínimo), no 15 Hz fijos.
- `POST /sessions`: se agregan `mode` (`practice` / `demo`) y `record`; se quitan `level` y
  `model_version` (el backend los obtiene del catálogo y del modelo cargado).
- `GET /catalog/signs` agrega `levels`, `components` (descripción de cada componente) y
  `validated`.
- `feedback` agrega `progress` y se define la lista de `feedback_code`.
- Dos manos: `num_hands = 2`, `vision.hands` lleva hasta 2 manos y el catálogo agrega
  `hands` (gracias y por favor se hacen con ambas). Nuevo `feedback_code`: `use_both_hands`.

## Arquitectura Del MVP

```text
iPhone                                              Laptop (misma red Wi-Fi)
┌──────────────────────────────┐   WebSocket JSON   ┌──────────────────────────────┐
│ Cámara frontal               │ ─ observaciones ─▶ │ mecabite-back                │
│ MediaPipe Hand Landmarker    │                    │ normaliza landmarks          │
│ MediaPipe Pose Landmarker    │ ◀── feedback ───── │ evalúa componentes de la seña│
│ UI de práctica / demo        │                    │ aprueba con 3 seguidas       │
└──────────────────────────────┘                    └──────────────────────────────┘
```

- **El video nunca sale del teléfono** (RNF-10). Solo viajan landmarks y metadatos.
- El backend guarda las observaciones **solo en memoria** durante la sesión. Únicamente
  guarda landmarks (nunca video) si la sesión se crea con `record: true`.
- REST para salud, catálogo, sesiones y simulación; WebSocket para el flujo en tiempo real.
  No se hace una petición HTTP por cuadro.

## Cómo Conectarse

### En la laptop (backend)

1. Arrancar el servidor escuchando en todas las interfaces. `fastapi dev` por defecto solo
   acepta conexiones de la propia laptop (`127.0.0.1`) y el teléfono no llegaría:

   ```bash
   uv run fastapi dev app/main.py --host 0.0.0.0
   ```

2. Obtener la IP de la laptop en el Wi-Fi: `ipconfig getifaddr en0`.
3. Si macOS pregunta si Python puede aceptar conexiones entrantes, permitirlo.
4. Desde Safari en el iPhone abrir `http://<ip>:8000/api/v1/health`. Si responde
   `{"status": "ok", ...}`, la red está bien y cualquier falla posterior es de la app.

### Red

- Laptop e iPhone en la **misma red Wi-Fi**. Las redes de escuela, oficina o de invitados
  suelen aislar a los dispositivos entre sí; si el paso 4 falla, usar el hotspot del
  teléfono o un router propio.
- La IP de la laptop cambia al cambiar de red: la URL base debe ser **configurable** en la
  app (pantalla de ajustes o configuración de build), nunca fija en el código.
- En el simulador de iOS funciona `http://127.0.0.1:8000`; en un iPhone físico hay que usar
  la IP de la laptop, no `localhost`.

URLs:

```text
REST:      http://<ip-de-la-laptop>:8000/api/v1
WebSocket: ws://<ip-de-la-laptop>:8000/api/v1/ws/sessions/{session_id}
```

### En la app iOS

- **Permiso de red local:** agregar `NSLocalNetworkUsageDescription` al `Info.plist`. iOS
  pregunta al usuario la primera vez que la app se conecta a un equipo de la red local; si
  lo niega, las conexiones fallan sin un error claro (se reactiva en Ajustes > Privacidad >
  Red local).
- **App Transport Security:** el MVP usa `http://` y `ws://` sin TLS. Si iOS bloquea la
  conexión (error `-1022`), agregar `NSAppTransportSecurity > NSAllowsLocalNetworking = YES`.
  Cualquier excepción adicional debe quedar solo en builds de desarrollo.
- `URLSessionWebSocketTask` sirve para el WebSocket.
- Cuando la app pasa a segundo plano iOS suspende el socket: al volver, reconectar a la
  **misma** sesión en lugar de crear otra.

## Datos Que Envía La App

### Mano (MediaPipe Hand Landmarker)

- Enviar todas las manos detectadas (máximo 2), en el orden en que las entrega MediaPipe. La
  app actual usa `num_hands = 1`: gracias y por favor (`hands: 2` en el catálogo) solo
  cuentan si la otra mano también participa, y el backend la detecta con las muñecas del
  cuerpo (`pose_landmarks`). Con `num_hands = 2` también se usa la segunda mano. El backend
  decide cuál es la mano principal: las letras se reconocen con una sola.
- `landmarks`: los 21 puntos **normalizados** (`x`, `y` de 0 a 1 respecto al ancho y alto
  de la imagen; `z` tal como lo entrega MediaPipe), en el orden oficial. No enviar los
  *world landmarks* ni escalar las coordenadas: el backend corrige la proporción de la
  imagen con `image_width` e `image_height`.
- Coordenadas de la **imagen vertical**, tal como la ve el usuario (cabeza arriba).
  `image_width` e `image_height` son de esa imagen ya orientada.
- `mirrored`: `true` si la imagen que recibió MediaPipe estaba volteada horizontalmente
  (vista tipo espejo o selfie). Ojo: en AVFoundation la vista previa de la cámara frontal se
  ve en espejo, pero los cuadros de `AVCaptureVideoDataOutput` no lo están salvo que se
  active `isVideoMirrored` en la conexión. Hay que reportar lo que recibió MediaPipe, no lo
  que muestra la vista previa.
- `handedness`: la etiqueta y el score **sin corregir** que da MediaPipe (`"Left"` o
  `"Right"`). El backend la interpreta según `mirrored`.
- Redondear a 5 decimales para reducir el tamaño del mensaje.

Pruebas rápidas para validar la orientación antes de integrar:

1. Mano abierta con los dedos hacia el techo: `y` de la punta del dedo medio (punto 12) debe
   ser **menor** que `y` de la muñeca (punto 0).
2. Mano derecha levantada a la altura del hombro derecho: con `mirrored: true` la muñeca debe
   quedar en `x > 0.5`; con `mirrored: false`, en `x < 0.5`.

### Cuerpo (MediaPipe Pose Landmarker)

- `pose_landmarks`: 33 puntos `[x, y, z, visibility]` normalizados, con las mismas reglas de
  orientación que la mano, del **mismo cuadro y timestamp** que las manos.
- Enviar en **todos los niveles**; `null` solo si no se detecta cuerpo. El backend lo usa para
  el encuadre (de frente, con cara y hombros visibles; si no, `adjust_framing`) y para saber
  si la otra mano participa. La localización del nivel 3 queda `not_available` hasta definir
  las zonas con la persona intérprete.
- Prueba rápida: de frente a la cámara, los puntos 11 y 12 (hombros) y 0 (nariz) deben tener
  `visibility > 0.5`.

### Guante — reservado

El guante se está terminando de construir. **En el MVP `glove` siempre va en `null`** y los
componentes de configuración y orientación se evalúan con la cámara.

Formato preliminar para cuando esté listo (11 valores con nombre; cambiará cuando
electrónica confirme unidades, rangos, frecuencia y tipo de conexión):

```json
{
  "connected": true,
  "sample_rate_hz": 15.0,
  "values": {
    "izq": 0.2, "der": 0.4, "arr": 0.8, "abj": 0.1, "giro_izq": 0.3, "giro_der": 0.2,
    "pulgar": 1.0, "indice": 3.0, "medio": 3.0, "anular": 3.0, "menique": 3.0
  }
}
```

### Tiempo, orden y frecuencia

- `sequence`: entero que empieza en 0 por sesión y aumenta en 1 por observación.
- `timestamp_ms`: milisegundos desde el inicio de la sesión, con reloj **monotónico**. Usar el
  tiempo de captura del cuadro (presentation timestamp del `CMSampleBuffer`), no la hora en
  que se envía: el backend mide trayectorias y velocidades con este valor. No usar la hora
  del calendario.
- Enviar **cada cuadro que procese MediaPipe**: lo ideal es 20–30 Hz y el mínimo 15 Hz. El
  backend trabaja por tiempo, así que no importa si la frecuencia varía.
- Enviar también los cuadros **sin mano** (`hands: []`); así el backend sabe que la mano
  salió de cuadro.
- Si se pierde un cuadro no se reenvía ni se inventa: `sequence` sigue contando.

## REST API

Base: `http://<ip-de-la-laptop>:8000/api/v1`

### `GET /health` — implementado

Respuesta `200`:

```json
{ "status": "ok", "environment": "local" }
```

### `GET /catalog/signs` — implementado

Catálogo de las 15 señas y los 3 niveles. La app lo usa para armar lecciones y mostrar
cada seña con su descripción (RF-11).

Respuesta `200` (recortada a dos señas):

```json
{
  "catalog_version": "0.1.0",
  "levels": [
    { "level": 1, "name": "Letras estáticas", "required_components": ["configuration", "orientation"] },
    { "level": 2, "name": "Letras con movimiento", "required_components": ["configuration", "orientation", "movement"] },
    { "level": 3, "name": "Palabras", "required_components": ["configuration", "orientation", "localization", "movement"] }
  ],
  "signs": [
    {
      "id": "a",
      "display_name": "A",
      "level": 1,
      "type": "static",
      "required_components": ["configuration", "orientation"],
      "hold_time_ms": 1000,
      "max_duration_ms": null,
      "reference_asset": "a.mp4",
      "hands": 1,
      "components": {
        "configuration": "Puño cerrado con el pulgar extendido junto al índice.",
        "orientation": "Palma hacia el frente.",
        "localization": null,
        "movement": null
      },
      "validated": false
    },
    {
      "id": "j",
      "display_name": "J",
      "level": 2,
      "type": "dynamic",
      "required_components": ["configuration", "orientation", "movement"],
      "hold_time_ms": null,
      "max_duration_ms": 3000,
      "reference_asset": "j.mp4",
      "hands": 1,
      "components": {
        "configuration": "Meñique extendido; demás dedos cerrados.",
        "orientation": null,
        "localization": null,
        "movement": "Trazar una J con la punta del meñique."
      },
      "validated": false
    }
  ]
}
```

| Campo | Significado |
| --- | --- |
| `id` | Identificador estable en ASCII; es el que se usa en todo el protocolo |
| `display_name` | Nombre para mostrar (puede llevar acentos) |
| `type` | `static`: se valida sosteniéndola `hold_time_ms` (RF-06). `dynamic`: se valida la trayectoria completa en máximo `max_duration_ms` (RF-07) |
| `hold_time_ms` | Lo configura el servidor y puede cambiar: la app debe leerlo del catálogo (para la barra de `progress`), no asumir 1000 |
| `required_components` | Componentes que se evalúan en ese nivel (RF-08) |
| `reference_asset` | Nombre del video de referencia **empaquetado en la app**; no se descarga del backend |
| `hands` | `2` si la seña se hace con las dos manos (hoy gracias y por favor); con una sola no se acepta. La app debe indicarlo en la lección |
| `components` | Descripción de cada componente; `null` si no aplica o aún no está redactada |
| `validated` | `false` mientras una persona intérprete de LSM no haya revisado la seña (RNF-05) |

Las descripciones actuales son **borradores** y varias están en `null`; los videos de
referencia todavía no existen. La app debe tolerar ambos casos.

IDs del catálogo:

| ID | Nombre | Nivel | Tipo |
| --- | --- | --- | --- |
| `a`, `b`, `c`, `l`, `y` | A, B, C, L, Y | 1 | static |
| `j`, `enie`, `q`, `x`, `z` | J, Ñ, Q, X, Z | 2 | dynamic |
| `hola`, `gracias`, `por_favor`, `ayuda`, `mama` | Hola, Gracias, Por favor, Ayuda, Mamá | 3 | dynamic |

`otra`, `reposo` y `transicion` son clases internas del reconocimiento y nunca aparecen en
el catálogo ni en `predicted_sign`.

### `POST /sessions` — implementado

Crea una sesión temporal. Una sesión evalúa una sola seña; para pasar a la siguiente se
cierra y se crea otra.

Solicitud:

```json
{
  "mode": "practice",
  "target_sign": "a",
  "participant_id": "p01",
  "device_id": "iphone-dev-1",
  "client_version": "0.2.0",
  "calibration_id": null,
  "record": false
}
```

| Campo | Significado |
| --- | --- |
| `mode` | `practice`: evalúa `target_sign` y cuenta ejecuciones (RF-12). `demo`: reconoce cualquier seña del catálogo y la reporta (RF-14); `target_sign` va en `null` |
| `target_sign` | `id` del catálogo |
| `calibration_id` | `null` en el MVP; se usará con la calibración del guante |
| `record` | `true` para guardar los landmarks de la sesión (nunca video) y usarlos para entrenar. Solo con consentimiento de la persona. Por defecto `false`. **Aún no se guarda nada** aunque venga en `true`: falta la política (ver Pendientes) |

Respuesta `201`:

```json
{
  "session_id": "sess_01JABC123",
  "mode": "practice",
  "target_sign": "a",
  "level": 1,
  "status": "created",
  "websocket_path": "/api/v1/ws/sessions/sess_01JABC123",
  "expires_in_seconds": 900,
  "protocol_version": "0.2.0",
  "catalog_version": "0.1.0",
  "model_version": "vision-0.1.0"
}
```

Reglas:

- `target_sign` debe existir en el catálogo (`404 SIGN_NOT_FOUND`).
- La sesión expira tras 15 minutos sin actividad.
- El backend no conserva la sesión después de cerrarla.

### `POST /simulations/predict` — pendiente

Para RF-15: probar el flujo del guante con un array de lecturas simuladas mientras el
dispositivo no existe. Lo usa sobre todo el equipo de backend; **la app no necesita
implementarlo**.

```json
{
  "target_sign": "a",
  "samples": [
    { "sequence": 0, "timestamp_ms": 0, "values": [0.2, 0.4, 0.8, 0.1, 0.3, 0.2, 1, 3, 3, 3, 3] },
    { "sequence": 1, "timestamp_ms": 67, "values": [0.2, 0.4, 0.8, 0.1, 0.3, 0.2, 1, 3, 3, 3, 3] }
  ]
}
```

El orden de `values` es `izq, der, arr, abj, giro_izq, giro_der, pulgar, indice, medio,
anular, menique`. Responde con el mismo formato que el mensaje `feedback`.

## WebSocket API

`ws://<ip-de-la-laptop>:8000/api/v1/ws/sessions/{session_id}`

### Flujo

1. La app crea la sesión con `POST /sessions`.
2. Abre el WebSocket de la sesión. Si la sesión no existe o expiró, el backend cierra con el
   código `4404`.
3. El backend envía `ready`.
4. La app envía un `observation` por cuadro.
5. El backend responde con `feedback`.
6. La app envía `end_session` al terminar.
7. El backend responde con `session_summary` y cierra el canal.

### `ready` (backend → app)

```json
{
  "type": "ready",
  "session_id": "sess_01JABC123",
  "protocol_version": "0.2.0",
  "model_version": "vision-0.1.0",
  "min_sample_rate_hz": 15,
  "required_inputs": ["hand", "pose"],
  "server_timestamp_ms": 0
}
```

`required_inputs` incluye `"glove"` cuando exista el guante.

### `observation` (app → backend)

```json
{
  "type": "observation",
  "sequence": 42,
  "timestamp_ms": 2800,
  "vision": {
    "image_width": 720,
    "image_height": 1280,
    "mirrored": true,
    "hands": [
      {
        "landmarks": [[0.52341, 0.71022, 0.0], [0.55012, 0.66431, -0.01234]],
        "handedness": { "label": "Right", "score": 0.97 }
      }
    ],
    "pose_landmarks": null
  },
  "glove": null
}
```

El ejemplo muestra dos landmarks para ahorrar espacio; una observación real lleva
exactamente 21 por mano.

| Campo | Regla |
| --- | --- |
| `vision.hands` | `[]` si no se detecta mano; máximo 2 elementos |
| `vision.hands[].landmarks` | Exactamente 21 elementos `[x, y, z]` |
| `vision.pose_landmarks` | Exactamente 33 elementos `[x, y, z, visibility]`; `null` si no se detecta cuerpo |
| `glove` | `null` en el MVP |

### `feedback` (backend → app)

```json
{
  "type": "feedback",
  "sequence": 42,
  "timestamp_ms": 2800,
  "state": "candidate",
  "target_sign": "a",
  "predicted_sign": "a",
  "confidence": 0.94,
  "progress": 0.6,
  "correct": true,
  "approved": false,
  "consecutive_correct": 2,
  "components": {
    "configuration": "correct",
    "orientation": "correct",
    "localization": "not_required",
    "movement": "not_required"
  },
  "feedback_code": "hold_position",
  "message": "Mantén la posición",
  "processing_time_ms": 18
}
```

- `sequence` indica a qué observación responde. Si el backend se atrasa puede saltarse
  observaciones viejas para no acumular retraso; con `sequence` la app correlaciona
  respuestas y mide el tiempo de ida y vuelta (RNF-01: menos de 500 ms).
- `progress` (0 a 1): avance para una barra en pantalla. En señas estáticas es el tiempo
  sostenido respecto a `hold_time_ms`; en dinámicas va en `null`.
- En modo `demo`: `target_sign`, `correct`, `approved` y `consecutive_correct` van en `null`
  y `predicted_sign` trae la seña reconocida.

Estados (`state`):

| Estado | Significado |
| --- | --- |
| `waiting` | Todavía no hay datos suficientes |
| `candidate` | Se está evaluando una posible seña |
| `confirmed` | Una ejecución se reconoció correctamente |
| `approved` | Se cumplieron 3 ejecuciones correctas consecutivas (RF-10) |
| `rejected` | La ejecución falló; `components` dice qué falló (RF-13) |
| `no_hand` | No se detecta la mano |
| `disconnected` | Falta una fuente requerida (por ejemplo, el guante cuando exista) |

Estados de cada componente: `correct`, `incorrect`, `not_required`, `not_available`
(la fuente no está disponible; **no** es un error del usuario) e `insufficient_data`.

Códigos de retroalimentación (`feedback_code`). La app muestra `message` y usa el código
para el apoyo visual (ícono, animación, color; RNF-09). No debe interpretar el texto de
`message`.

| `feedback_code` | Cuándo | Mensaje de ejemplo |
| --- | --- | --- |
| `show_hand` | No se ve la mano | Muestra tu mano a la cámara |
| `hold_position` | Seña estática correcta, sosteniéndose | Mantén la posición |
| `correct` | Ejecución confirmada | ¡Bien! 2 de 3 |
| `approved` | Tercera ejecución consecutiva | ¡Seña aprobada! |
| `wrong_configuration` | Falló la forma de la mano | Revisa la forma de tu mano |
| `wrong_orientation` | Falló la orientación | Revisa hacia dónde apunta tu palma |
| `wrong_movement` | Falló el movimiento (niveles 2 y 3) | Revisa el movimiento |
| `wrong_localization` | Falló la ubicación (nivel 3) | Revisa dónde colocas la mano |
| `too_slow` | Seña dinámica de más de `max_duration_ms` | Hazla en menos de 3 segundos |
| `use_both_hands` | Seña de dos manos hecha con una (`hands: 2`) | Esta seña se hace con las dos manos |
| `adjust_framing` | No se ve el cuerpo de frente con cara y hombros; o seña correcta sin buen encuadre (no cuenta) | Aléjate un poco: no se ven tus dos hombros |

`feedback_code` va en `null` mientras se espera la seña (por ejemplo, `Haz la seña A` o
`Baja la mano y vuelve a hacer la seña` tras una ejecución confirmada). Un resultado
(`confirmed`, `rejected`) se sigue reportando 1.5 s para que la app alcance a mostrarlo.

En `practice` solo cuentan como intento las señas del mismo tipo que la objetivo: con una
estática no se buscan movimientos y con una dinámica no cuentan las formas quietas del camino.
`wrong_configuration` mientras se sostiene otra forma aparece tras `VISION_WRONG_SIGN_MS`
(500 ms), y una pérdida de la seña menor a `VISION_GRACE_MS` (300 ms) no reinicia `progress`.

### `end_session` (app → backend)

```json
{ "type": "end_session", "reason": "user_finished" }
```

Respuesta:

```json
{
  "type": "session_summary",
  "session_id": "sess_01JABC123",
  "target_sign": "a",
  "attempts": 4,
  "correct_attempts": 3,
  "approved": true,
  "average_processing_time_ms": 19
}
```

## Errores

Respuesta REST:

```json
{
  "error": {
    "code": "INVALID_OBSERVATION",
    "message": "Se esperaban 21 landmarks de mano",
    "details": { "received": 20 }
  }
}
```

| HTTP / código | Significado |
| --- | --- |
| `400 INVALID_REQUEST` | JSON inválido o campo faltante |
| `404 SIGN_NOT_FOUND` | La seña no existe en el catálogo |
| `404 SESSION_NOT_FOUND` | La sesión no existe o expiró |
| `409 SESSION_NOT_READY` | La sesión no está lista para recibir datos |
| `422 INVALID_OBSERVATION` | Forma, rango o timestamp inválido |
| `426 PROTOCOL_VERSION_UNSUPPORTED` | Versión de protocolo incompatible |
| `500 INFERENCE_ERROR` | Error interno del modelo |
| `503 MODEL_NOT_AVAILABLE` | El servidor no tiene modelo entrenado (`vision train` y reiniciar) |
| `503 SIGN_NOT_TRAINED` | La seña está en el catálogo pero el modelo aún no tiene datos de ella |

En el WebSocket los errores llegan como mensaje y **no cierran la conexión**, salvo que sean
irrecuperables:

```json
{
  "type": "error",
  "sequence": 42,
  "code": "INVALID_OBSERVATION",
  "message": "Se esperaban 21 landmarks de mano"
}
```

## Checklist Para La App

- [ ] URL base configurable; nada fijo en el código.
- [ ] `NSLocalNetworkUsageDescription` en `Info.plist` y excepción de ATS para red local.
- [ ] MediaPipe Hand y Pose Landmarker en modo video en vivo, con el mismo cuadro y timestamp.
- [ ] Enviar landmarks normalizados sin escalar, más `image_width`, `image_height`,
      `mirrored` y `handedness` crudo. Pasar las dos pruebas de orientación.
- [ ] Enviar cada cuadro, incluidos los que no tienen mano (`hands: []`).
- [ ] `timestamp_ms` del momento de captura, monotónico y relativo al inicio de la sesión.
- [ ] `glove: null` por ahora; `pose_landmarks` en todos los niveles.
- [ ] Mostrar `message`, usar `feedback_code` para el apoyo visual y `progress` para la barra.
- [ ] Tratar `not_available` como espera, no como fallo.
- [ ] Reconectar a la misma sesión al volver de segundo plano.
- [ ] Videos de referencia empaquetados en la app (cuando existan).
- [ ] Nunca enviar video ni imágenes.

## Criterios De Integración Del MVP

1. El iPhone consulta `/health` en la misma red Wi-Fi.
2. La app descarga `/catalog/signs` y muestra las 15 señas por nivel.
3. Crea una sesión de práctica para `a`.
4. Abre el WebSocket y recibe `ready`.
5. Envía observaciones a 15 Hz o más, y pasa las pruebas de orientación.
6. Recibe `feedback` sin enviar video.
7. Una seña estática sostenida al menos 1000 ms se confirma.
8. Tres ejecuciones correctas consecutivas producen `approved: true`.
9. Un dato inválido produce un `error` descriptivo sin cerrar la sesión.
10. El tiempo de ida y vuelta medido en el teléfono se mantiene por debajo de 500 ms.

## Pendientes

- **Guante:** unidades, rangos, frecuencia y tipo de conexión; se integra cuando electrónica
  lo termine.
- **Intérprete de LSM:** validar descripciones y patrones, y confirmar qué palabras usan
  dos manos. En el dataset de glosas usan ambas gracias, por favor y ayuda; hoy el catálogo
  marca gracias y por favor.
- **Videos de referencia** de las 15 señas.
- **Nivel 3:** confirmar Pose Landmarker (33 puntos) y definir las zonas de localización.
- **Política para `record: true`:** consentimiento, dónde se guarda y cómo se borra.
- **Versión final (fuera del MVP):** reconocimiento sin internet (RNF-07), autenticación y
  TLS si el backend sale de la red local.
