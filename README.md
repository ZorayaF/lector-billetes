# Lector de Billetes COP (Accesibilidad Visual)

Sistema de visión por computadora e inferencia móvil diseñado para identificar denominaciones de billetes colombianos (COP) en tiempo real, asistiendo a personas con discapacidad visual.

## Estructura del Proyecto

```text
lector-billetes/
├── cop-app/                 # Aplicación frontend/móvil (consume public/cop_model.onnx)
├── videos_crudos/           # Videos originales organizados en subcarpetas por clase (ej. 2k, 5k, 10k...)
├── cop_dataset/             # Dataset generado dividido en train/ (80%), val/ (10%) y test/ (10%)
├── procesar_videos.py       # Script ETL: extracción de fotogramas cada N cuadros y partición del dataset
├── train_cop.py             # Entrenamiento con Transfer Learning (MobileNetV3Small) y exportación a TFLite
├── export_onnx.py           # Conversión del modelo .keras a formato ONNX (opset 13) para la app
├── labels.json              # Lista ordenada de clases generada durante el entrenamiento
├── cop_model.keras          # Modelo entrenado en formato nativo TensorFlow/Keras
├── cop_model.tflite         # Modelo optimizado (Quantization DEFAULT) para TensorFlow Lite
└── venv/                    # Entorno virtual de Python
```

## Especificaciones Técnicas Actuales (v0.1.0)

### Extracción de Datos (procesar_videos.py):

- Formatos soportados: .mp4, .avi, .mov, .mkv, .m4v.

- Muestreo: 1 fotograma cada 5 cuadros (FRAME_INTERVAL = 5).

- Redimensionamiento intermedio: 640x640 px (cv2.INTER_AREA).

- División de datos: 80% Entrenamiento, 10% Validación, 10% Prueba (Semilla fija: 42).

### Arquitectura del Modelo (train_cop.py):

- Backbone: MobileNetV3Small preentrenado en ImageNet (include_top=False, pesos base congelados).

- Entrada: Tensor [1, 224, 224, 3] en float32.

- Data Augmentation (en grafo): RandomFlip (H/V), RandomRotation(0.2), RandomZoom(0.15), RandomContrast(0.2), RandomBrightness(0.2).

- Cabezal de clasificación: GlobalAveragePooling2D -> Dropout(0.2) -> Dense(num_classes, softmax).

- Hiperparámetros: Optimizador Adam (lr=1e-3), Batch Size 32, hasta 15 épocas con EarlyStopping (paciencia 4) y ReduceLROnPlateau (factor 0.5, paciencia 2).

### Formatos de Salida:

- .keras: Modelo completo con capas de preprocesamiento.

- .tflite: Exportado con tf.lite.Optimize.DEFAULT.

- .onnx: Exportado con tf2onnx (opset=13) directamente en cop-app/public/cop_model.onnx.

### Flujo de Ejecución (Pipeline)

1. Preparar videos: Colocar los videos grabados en videos_crudos/<denominacion>/.

2. Generar dataset:

```bash
python procesar_videos.py
```

3. Entrenar modelo

```bash
python train_cop.py
```

4. Exportar a la App (ONNX):

```bash
python export_onnx.py
```

## Diagnóstico Técnico y Problemas Detectados (Base para v0.2.0)

1. Doble preprocesamiento en MobileNetV3Small (train_cop.py): Keras incluye por defecto el reescalado interno en MobileNetV3Small (include_preprocessing=True), que espera valores en [0, 255]. El script actual aplica mobilenet_v3.preprocess_input manualmente antes de la base, comprimiendo los píxeles dos veces y degradando severamente el aprendizaje.

2. Fotogramas vacíos o borrosos (procesar_videos.py): La extracción ciega cada 5 fotogramas incluye cuadros desenfocados por movimiento o momentos donde el billete aún no entra en escena.

3. Data Leakage (Fuga de datos): Se mezclan todos los fotogramas de un mismo video aleatoriamente antes de separar train/val/test, provocando que fotogramas separados por apenas 0.16 segundos estén tanto en entrenamiento como en validación.

4. Distorsión geométrica: Redimensionar directamente de formato panorámico (16:9 / 9:16) a cuadrado (640x640 y luego 224x224) deforma las proporciones de los billetes.
