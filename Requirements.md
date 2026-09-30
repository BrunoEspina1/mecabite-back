# Requerimientos del sistema

## Requerimientos funcionales

### Conexión y usuario

- **RF-01:** El sistema debe recibir e interpretar en tiempo real los datos de configuración y orientación enviados por el dispositivo.
- **RF-02:** El sistema debe guiar la calibración antes de la primera lección y permitir recalibrar en cualquier momento.

### Reconocimiento

- **RF-03:** El sistema debe capturar video y obtener los puntos de referencia de mano y cuerpo con MediaPipe para determinar la localización de la seña.
- **RF-04:** El sistema debe combinar los datos del dispositivo y de la cámara para clasificar la seña ejecutada.
- **RF-05:** El sistema debe detectar y evaluar el movimiento en las señas dinámicas.
- **RF-06:** El sistema debe validar una seña estática cuando la configuración y la orientación correctas se mantengan estables durante al menos 1 segundo.
- **RF-07:** El sistema debe validar una seña dinámica comparando su trayectoria completa contra el patrón de referencia, dentro de una ventana máxima de 3 segundos.
- **RF-08:** El sistema debe evaluar únicamente los componentes que correspondan a cada nivel:
  - **Nivel 1:** configuración y orientación.
  - **Nivel 2:** se agrega movimiento.
  - **Nivel 3:** se agregan localización y movimiento.

### Contenido y aprendizaje

- **RF-09:** El sistema debe ofrecer exclusivamente el siguiente catálogo de señas, organizado en tres niveles:
  1. **Letras estáticas:** A, B, C, L, Y.
  2. **Letras con movimiento:** J, Ñ, Q, X, Z.
  3. **Palabras:** Hola, Gracias, Por favor, Ayuda, Mamá.
- **RF-10:** El sistema debe considerar aprobada una seña cuando el usuario la ejecute correctamente 3 veces consecutivas.
- **RF-11:** El sistema debe mostrar, para cada seña, un video o animación de referencia y una descripción de sus componentes.
- **RF-12:** El sistema debe ofrecer un modo práctica con evaluación en tiempo real.
- **RF-13:** Cuando un intento falle, el sistema debe indicar qué componente falló: configuración, orientación, localización o movimiento.
- **RF-14:** El sistema debe incluir un modo demostración que reconozca una seña y la muestre en pantalla.
- **RF-15:** El sistema debe permitir una entrada mediante consola de datos simulados para poder trabajar antes de que el dispositivo esté listo. La entrada debe basarse en un array que contenga la información.

## Requerimientos no funcionales

### Rendimiento

- **RNF-01:** El sistema debe mostrar el resultado de una seña en menos de 500 ms después de ejecutarla.
- **RNF-02:** El procesamiento con MediaPipe debe mantener al menos 20 fps en un teléfono de gama media.

### Precisión

- **RNF-03:** El sistema debe alcanzar una exactitud mínima de reconocimiento de 90 % en nivel 1, 80 % en nivel 2 y 75 % en nivel 3, medida con usuarios distintos a quienes grabaron los patrones.
- **RNF-04:** La tasa de falsos positivos debe ser menor al 10 %.
- **RNF-05:** Los patrones de referencia deben estar validados por al menos una persona usuaria o intérprete de LSM.

### Compatibilidad y disponibilidad

- **RNF-06:** El sistema debe funcionar en iOS 18 o superior.
- **RNF-07:** El reconocimiento debe funcionar sin conexión a internet.

### Usabilidad

- **RNF-08:** Un usuario nuevo debe completar la calibración y su primera seña en menos de 5 minutos, sin ayuda.
- **RNF-09:** La interfaz debe estar en español y ofrecer apoyo visual en cada instrucción.

### Seguridad y privacidad

- **RNF-10:** El video debe procesarse localmente, sin almacenarse ni enviarse a servidores.

### Mantenibilidad

- **RNF-11:** El catálogo de señas y sus patrones deben almacenarse separados del código, para poder agregar señas sin modificar el clasificador.
