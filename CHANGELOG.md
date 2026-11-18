# Changelog

Todos los cambios notables en el proyecto `lector-billetes` serán documentados en este archivo.

## [Unreleased] - Plan de Mejora (v0.2.0)

### Por Corregir (Bugs Críticos Identificados)

- **`train_cop.py`**: Eliminar la doble normalización en `MobileNetV3Small` (ajustar `include_preprocessing` o retirar `preprocess_input` manual) para que la red reciba los tensores en el rango correcto.
- **`procesar_videos.py`**: Evitar _Data Leakage_ separando por videos/bloques temporales o asegurando diversidad real entre `train`, `val` y `test`.

### Por Añadir (Mejoras de Calidad y Precisión)

- **`procesar_videos.py` (Filtro de Calidad)**:
  - Detección de desenfoque (_blur_) mediante la varianza del Laplaciano (`cv2.Laplacian`) para descartar fotogramas movidos.
  - Filtro de exposición/vacío para descartar fotogramas en blanco, negros o sin textura suficiente.
  - Preservación de relación de aspecto o recorte centrado para no deformar el billete al pasar a `640x640`.
- **`train_cop.py` (Entrenamiento y Evaluación)**:
  - Fase de _Fine-Tuning_ descongelando las últimas capas de `MobileNetV3Small` con un _learning rate_ bajo (`1e-5`).
  - Incorporar una clase `fondo` / `ninguno` para que la app no clasifique superficies vacías como billetes.
  - Evaluación sobre el split `test/` al finalizar el entrenamiento (Matriz de confusión y F1-score por denominación).
  - Pesos de clase (`class_weight`) en caso de desbalance entre denominaciones.

---

## [0.1.0] - Línea Base Actual

### Añadido

- **`procesar_videos.py`**:
  - Extracción automática de fotogramas cada 5 cuadros (`FRAME_INTERVAL = 5`) usando OpenCV.
  - Redimensionamiento estandarizado a `640x640` px con interpolación `INTER_AREA`.
  - Partición aleatoria reproducible (`RANDOM_SEED = 42`) en `train` (80%), `val` (10%) y `test` (10%).
- **`train_cop.py`**:
  - Pipeline `tf.data` con `cache()` y `prefetch(AUTOTUNE)`.
  - Generación automática de `labels.json` con las clases detectadas.
  - Bloque `data_augmentation` integrado en el modelo (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomContrast`, `RandomBrightness`).
  - Clasificador basado en `MobileNetV3Small` (`224x224x3`) con `Adam (1e-3)` y `EarlyStopping` / `ReduceLROnPlateau`.
  - Exportación directa a `cop_model.keras` y conversión cuantizada a `cop_model.tflite`.
- **`export_onnx.py`**:
  - Conversión desde `cop_model.keras` a ONNX (`opset=13`, entrada `[1, 224, 224, 3]`) con salida directa hacia `cop-app/public/cop_model.onnx`.
