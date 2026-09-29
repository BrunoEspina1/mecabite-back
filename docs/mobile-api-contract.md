# Contrato De Integración Con La Aplicación Móvil

## Estado Del Documento

- **Versión:** 0.1.0
- **Propósito:** integración del hackathon y pruebas de inferencia remota.
- **Estado del backend:** actualmente solo está implementado `GET /api/v1/health`.
- **Alcance:** contrato propuesto para que móvil y backend puedan trabajar en paralelo.

Este contrato no sustituye la inferencia local de producción. El modo remoto sirve
para validar el flujo mientras se entrena y exporta el modelo a iOS.

## Decisiones De Arquitectura

### 1. Dos modos de ejecución

La aplicación debe contemplar dos modos:

| Modo | Uso | Dónde se ejecuta la inferencia |
| --- | --- | --- |
| `remote_debug` | Desarrollo, hackathon y pruebas con el backend local | `mecabite-back` |
| `local` | Uso final, sin internet y con privacidad | Aplicación iOS |

El modo `remote_debug` puede conectarse a un backend ejecutándose en una laptop
dentro de la misma red local. El modo `local` no depende del backend.

### 2. MediaPipe se ejecuta en el dispositivo

La aplicación debe capturar el video y ejecutar MediaPipe localmente. Al backend
no se deben enviar frames ni video.

En el modo remoto se envían únicamente:

- Landmarks de mano.
- Landmarks de cuerpo cuando el nivel los requiera.
- Confianza de detección.
- Mano detectada.
- Datos del guante.

Esto reduce latencia, ancho de banda y exposición de información privada.

### 3. WebSocket para tiempo real

No se debe hacer una petición HTTP por cada lectura del guante o frame de cámara.
El canal de observaciones será WebSocket.

REST se usará para:

- Salud del servicio.
- Catálogo de señas.
- Creación de sesiones.
- Pruebas con arrays simulados.

### 4. El backend no almacena observaciones por defecto

El backend mantendrá las observaciones en memoria durante la sesión y devolverá
la retroalimentación. No guardará video ni landmarks salvo que se active
explícitamente un modo de recolección para entrenamiento.

### 5. Los identificadores son estables y ASCII

Los nombres que se muestran al usuario pueden tener acentos, pero los IDs del
protocolo no los tendrán:

| ID | Nombre mostrado | Nivel |
| --- | --- | --- |
| `a` | A | 1 |
| `b` | B | 1 |
| `c` | C | 1 |
| `l` | L | 1 |
| `y` | Y | 1 |
| `j` | J | 2 |
| `enie` | Ñ | 2 |
| `q` | Q | 2 |
| `x` | X | 2 |
| `z` | Z | 2 |
| `hola` | Hola | 3 |
| `gracias` | Gracias | 3 |
| `por_favor` | Por favor | 3 |
| `ayuda` | Ayuda | 3 |
| `mama` | Mamá | 3 |

`reposo` y `transicion` son clases internas para el reconocimiento y no deben
mostrarse como señas del catálogo.

## Requisitos Para Móvil

### Datos Del Guante

El contrato inicial conserva el formato de `Manitas`: 11 valores por lectura,
con una frecuencia objetivo aproximada de 15 Hz.

Orden canónico si se usa un array:

```text
[
  izq,
  der,
  arr,
  abj,
  giro_izq,
  giro_der,
  pulgar,
  indice,
  medio,
  anular,
  menique
]
```

La forma recomendada para WebSocket es un objeto con nombres, para evitar errores
si algún día cambia el orden:

```json
{
  "izq": 0.2,
  "der": 0.4,
  "arr": 0.8,
  "abj": 0.1,
  "giro_izq": 0.3,
  "giro_der": 0.2,
  "pulgar": 1.0,
  "indice": 3.0,
  "medio": 3.0,
  "anular": 3.0,
  "menique": 3.0
}
```

La electrónica debe confirmar antes de integrar:

- Unidades de orientación.
- Rango real de cada sensor.
- Frecuencia real.
- Significado de positivo y negativo.
- Comportamiento durante desconexiones.
- Si los sensores flex usan la escala 1 a 3.

### MediaPipe

Para cada mano, móvil debe enviar exactamente 21 landmarks en el orden oficial
de MediaPipe Hand Landmarker. Cada landmark tiene `[x, y, z]`.

- `x` y `y` son coordenadas normalizadas.
- `z` conserva la convención entregada por MediaPipe.
- La lista debe tener exactamente 21 elementos.
- Si no se detecta una mano, `hand_landmarks` debe ser `null`.
- `handedness` puede ser `left`, `right` o `unknown`.

Para el cuerpo se reservará una lista de landmarks compatible con MediaPipe Pose.
Se enviará como `null` hasta que el nivel de la seña requiera localización.

### Tiempo Y Orden

Cada observación debe incluir:

- `sequence`: entero creciente desde cero por sesión.
- `timestamp_ms`: tiempo monotónico relativo al inicio de la sesión.
- `source`: `mobile`.

No se debe usar la hora calendario del teléfono para sincronizar sensores y cámara.
La combinación debe usar el reloj monotónico de la sesión.

Si se pierde una lectura, móvil debe conservar el número de secuencia y reportar
el salto. No debe inventar valores sin indicarlo.

### Calibración

La calibración se realiza en móvil y produce un `calibration_id`. El backend no
debe asumir que todos los usuarios tienen los mismos rangos.

El paquete de sesión debe incluir:

- `calibration_id`.
- `handedness` configurada por el usuario, si existe.
- Versión del algoritmo de calibración.
- Fecha local de calibración.

Los valores calibrados deben poder usarse sin conexión.

## REST API

### Base URL

En desarrollo:

```text
http://<ip-de-la-laptop>:8000/api/v1
```

En el simulador iOS, normalmente se puede usar `127.0.0.1`. En un teléfono
físico se debe usar la IP local de la laptop, no `localhost`.

WebSocket:

```text
ws://<ip-de-la-laptop>:8000/api/v1/ws/sessions/{session_id}
```

### `GET /health`

Endpoint existente para comprobar conectividad.

Respuesta `200`:

```json
{
  "status": "ok",
  "environment": "local"
}
```

### `GET /catalog/signs`

Devuelve el catálogo que la app debe mostrar.

Respuesta `200`:

```json
{
  "catalog_version": "0.1.0",
  "signs": [
    {
      "id": "a",
      "display_name": "A",
      "level": 1,
      "type": "static",
      "required_components": ["configuration", "orientation"],
      "hold_time_ms": 1000,
      "max_duration_ms": null,
      "reference_asset": "a.mp4"
    },
    {
      "id": "j",
      "display_name": "J",
      "level": 2,
      "type": "dynamic",
      "required_components": ["configuration", "orientation", "movement"],
      "hold_time_ms": null,
      "max_duration_ms": 3000,
      "reference_asset": "j.mp4"
    }
  ]
}
```

`reference_asset` debe ser una referencia a un recurso local o empaquetado en la
app. No se debe asumir que estará disponible desde internet.

### `POST /sessions`

Crea una sesión temporal para el modo `remote_debug`.

Solicitud:

```json
{
  "mode": "remote_debug",
  "target_sign": "a",
  "level": 1,
  "participant_id": "p01",
  "device_id": "iphone-local",
  "calibration_id": "cal-001",
  "model_version": "sensor-0.1.0",
  "client_version": "0.1.0"
}
```

Respuesta `201`:

```json
{
  "session_id": "sess_01JABC123",
  "mode": "remote_debug",
  "target_sign": "a",
  "level": 1,
  "status": "created",
  "websocket_path": "/api/v1/ws/sessions/sess_01JABC123",
  "expires_in_seconds": 900
}
```

Reglas:

- `target_sign` debe existir en el catálogo.
- `level` debe coincidir con el nivel del catálogo.
- La sesión expira después de 15 minutos sin actividad.
- El backend no debe persistir la sesión después de cerrarla.

### `POST /simulations/predict`

Endpoint exclusivo para RF-15 y pruebas sin dispositivo. Recibe un array de
lecturas del guante. No simula la cámara si no se envían landmarks.

Solicitud:

```json
{
  "target_sign": "a",
  "level": 1,
  "samples": [
    {
      "sequence": 0,
      "timestamp_ms": 0,
      "values": [0.2, 0.4, 0.8, 0.1, 0.3, 0.2, 1, 3, 3, 3, 3]
    },
    {
      "sequence": 1,
      "timestamp_ms": 67,
      "values": [0.2, 0.4, 0.8, 0.1, 0.3, 0.2, 1, 3, 3, 3, 3]
    }
  ]
}
```

Respuesta `200`:

```json
{
  "target_sign": "a",
  "predicted_sign": "a",
  "confidence": 0.94,
  "correct": true,
  "approved": false,
  "consecutive_correct": 1,
  "components": {
    "configuration": "correct",
    "orientation": "correct",
    "localization": "not_required",
    "movement": "not_required"
  },
  "feedback_code": "hold_position",
  "message": "Mantén la posición"
}
```

## WebSocket API

### Flujo De Conexión

1. Móvil crea una sesión con `POST /sessions`.
2. Móvil abre el WebSocket de la sesión.
3. Backend responde con `ready`.
4. Móvil envía observaciones.
5. Backend responde con un mensaje `feedback` por observación procesada.
6. Móvil envía `end_session` al terminar.
7. Backend responde con `session_summary` y cierra el canal.

### Mensaje `ready`

```json
{
  "type": "ready",
  "session_id": "sess_01JABC123",
  "protocol_version": "0.1.0",
  "model_version": "sensor-0.1.0",
  "expected_sample_rate_hz": 15,
  "required_inputs": ["glove"],
  "server_timestamp_ms": 0
}
```

### Mensaje `observation`

Este es el mensaje principal de móvil a backend.

```json
{
  "type": "observation",
  "sequence": 42,
  "timestamp_ms": 2800,
  "glove": {
    "connected": true,
    "sample_rate_hz": 15.0,
    "values": {
      "izq": 0.2,
      "der": 0.4,
      "arr": 0.8,
      "abj": 0.1,
      "giro_izq": 0.3,
      "giro_der": 0.2,
      "pulgar": 1.0,
      "indice": 3.0,
      "medio": 3.0,
      "anular": 3.0,
      "menique": 3.0
    }
  },
  "vision": {
    "hand_landmarks": [[0.5, 0.5, 0.0]],
    "pose_landmarks": null,
    "handedness": "right",
    "detection_confidence": 0.95
  }
}
```

En el ejemplo se muestra un solo landmark para reducir espacio, pero una
observación real debe enviar exactamente 21 landmarks de mano.

Campos opcionales:

- `glove` puede ser `null` durante una prueba de visión.
- `vision` puede ser `null` para señas que solo requieran sensores.
- `pose_landmarks` puede ser `null` en niveles 1 y 2.

### Mensaje `feedback`

```json
{
  "type": "feedback",
  "sequence": 42,
  "timestamp_ms": 2800,
  "state": "candidate",
  "target_sign": "a",
  "predicted_sign": "a",
  "confidence": 0.94,
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

Estados posibles:

- `waiting`: todavía no hay suficientes datos.
- `candidate`: se está evaluando una posible seña.
- `confirmed`: una ejecución fue reconocida correctamente.
- `approved`: se cumplieron 3 ejecuciones consecutivas.
- `rejected`: la ejecución falló.
- `no_hand`: no se detectó la mano.
- `disconnected`: falta una fuente requerida.

Estados de componentes:

- `correct`.
- `incorrect`.
- `not_required`.
- `not_available`.
- `insufficient_data`.

### Mensaje `end_session`

Solicitud de móvil:

```json
{
  "type": "end_session",
  "reason": "user_finished"
}
```

Respuesta del backend:

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

Respuesta REST estándar:

```json
{
  "error": {
    "code": "INVALID_OBSERVATION",
    "message": "glove.values debe contener los 11 sensores requeridos",
    "details": {
      "missing_fields": ["menique"]
    }
  }
}
```

Códigos iniciales:

| HTTP / código | Significado |
| --- | --- |
| `400 INVALID_REQUEST` | JSON inválido o campo faltante |
| `404 SIGN_NOT_FOUND` | La seña no existe en el catálogo |
| `409 SESSION_NOT_READY` | La sesión no está lista para recibir datos |
| `422 INVALID_OBSERVATION` | Lectura con forma, rango o timestamp inválido |
| `426 PROTOCOL_VERSION_UNSUPPORTED` | Versión de protocolo incompatible |
| `429 RATE_LIMITED` | Se enviaron demasiados mensajes |
| `500 INFERENCE_ERROR` | Error interno del modelo |

En WebSocket, los errores deben enviarse como mensajes y no cerrar la conexión
salvo que el error sea irrecuperable:

```json
{
  "type": "error",
  "sequence": 42,
  "code": "INVALID_OBSERVATION",
  "message": "Se esperaban 21 landmarks de mano"
}
```

## Necesidades De Implementación En Móvil

- Implementar un selector de modo `local` / `remote_debug`.
- No enviar video al backend.
- Ejecutar MediaPipe en la app y enviar landmarks.
- Mantener un timestamp monotónico por sesión.
- Mantener `sequence` aunque existan paquetes perdidos.
- Enviar datos del guante con nombres de sensores o respetar estrictamente el orden canónico.
- Guardar la calibración localmente.
- Mostrar `message` y `feedback_code` como retroalimentación visual.
- Tratar `not_available` como un estado de espera, no como un fallo del usuario.
- Reconectar el WebSocket sin crear una segunda sesión automáticamente.
- Desactivar el modo remoto si no existe una red local disponible.
- Mantener los assets de referencia disponibles offline.
- Versionar junto con la app el `protocol_version`, `catalog_version` y `model_version`.

## Criterios De Integración

La integración inicial se considera funcional cuando:

1. El teléfono puede consultar `/health` desde la misma red local.
2. Puede crear una sesión para la seña `a`.
3. Puede abrir el WebSocket y recibir `ready`.
4. Puede enviar observaciones a aproximadamente 15 Hz.
5. Recibe `feedback` sin enviar video.
6. Una seña estática se evalúa durante al menos 1000 ms.
7. Tres ejecuciones correctas consecutivas producen `approved: true`.
8. Un dato faltante produce un error descriptivo sin cerrar necesariamente la sesión.
9. El tiempo de procesamiento del backend se mantiene por debajo de 500 ms.
10. El modo local puede funcionar sin que el backend esté disponible.

## Pendientes Antes De Congelar El Contrato

- Confirmar las unidades y rangos reales del guante.
- Confirmar la frecuencia real de lectura.
- Definir si los landmarks de pose serán necesarios para las palabras del Nivel 3.
- Definir el formato final de exportación a Core ML.
- Implementar autenticación si el backend sale de la red local.
- Definir la política de almacenamiento explícito para sesiones de entrenamiento.
- Confirmar las señas y patrones con una persona intérprete de LSM.
