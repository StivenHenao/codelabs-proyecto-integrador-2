
Codelabs - Proyecto Integrador II
================================

Este repositorio contiene diversos codelabs y experimentos desarrollados durante el curso Proyecto Integrador II.
Cada carpeta incluye código, modelos y ejemplos relacionados con técnicas de machine learning, visión por computadora y procesamiento de datos.

------------------------------------------------------------
Estructura del repositorio
------------------------------------------------------------

### Keras-XOR/
    Ejemplo básico usando Keras para resolver el problema XOR. Ideal para entender redes neuronales simples.

### MFCC Con microfono/
    Extracción de coeficientes MFCC en tiempo real a partir del micrófono. Útil para reconocimiento de voz.

### MTCNNEnimagenes/
    Uso de MTCNN para detección de rostros en imágenes.

### Reconocimiento de voz/
    Scripts para convertir audio en texto utilizando diferentes librerías de STT (Speech-To-Text).

### Yolo Train Roboflow Universe/
    Proyecto de entrenamiento de YOLO usando datasets de Roboflow Universe.

### clasificador comentarios negocio/
    Clasificador de comentarios para detectar sentimiento o tipo de mensaje en un negocio.

### codelab-face-dlib-video/
    Detección de rostros en video usando dlib.

### deteccion SSD vs YOLO/
    Comparativa entre SSD y YOLO para detección de objetos. Incluye scripts de evaluación y entrenamiento.

### deteccion-yolo-lite/
    Implementación ligera de YOLO para dispositivos con menos recursos.

### estafa spam/
    Clasificador para detectar mensajes de estafa/spam.

------------------------------------------------------------
Requisitos
------------------------------------------------------------

1. Tener Python 3.9+ instalado.
2. Crear un entorno virtual (opcional pero recomendado):
```
    python -m venv venv
    source venv/bin/activate  # Linux / Mac
    venv\Scripts\activate     # Windows
```
3. Instalar dependencias:

``
    pip install -r requirements.txt
``

------------------------------------------------------------
Ejecución de los ejemplos
------------------------------------------------------------

Cada carpeta puede ejecutarse de forma independiente.
Dentro de cada carpeta revisa el código o notebooks para ver instrucciones específicas.
Ejemplo para ejecutar un script de YOLO:
```
    cd "deteccion SSD vs YOLO"
    python train.py
```

Notas
------------------------------------------------------------

- La carpeta .history/ está ignorada en Git y solo es usada por VS Code para mantener historial de cambios.
- Algunos datasets grandes no están incluidos en el repositorio.
  Revisa cada carpeta y su README interno para instrucciones de descarga.
