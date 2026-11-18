import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# --- CONFIGURACIÓN ---
DATASET_DIR = "cop_dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3

def main():
    train_dir = Path(DATASET_DIR) / "train"
    val_dir = Path(DATASET_DIR) / "val"

    if not train_dir.exists():
        print(f"Error: No existe {train_dir}. Ejecuta primero procesar_videos.py")
        return

    # 1. Cargar datasets desde carpetas
    print("--- Cargando Datasets ---")
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"\nClases detectadas ({num_classes}): {class_names}")

    # Guardar mapa de etiquetas en un JSON para la app móvil
    with open("labels.json", "w") as f:
        json.dump(class_names, f, indent=2)
    print("-> Etiquetas guardadas en 'labels.json'")

    # Optimización de pipeline de datos en memoria/GPU
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 2. Capas de Data Augmentation integradas en el modelo
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ], name="data_augmentation")

    # 3. Construir arquitectura con MobileNetV3Small
    print("\n--- Construyendo Modelo MobileNetV3 ---")
    base_model = keras.applications.MobileNetV3Small(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        minimalistic=False
    )
    # Congelamos los pesos base para el transfer learning inicial
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = data_augmentation(inputs)
    # Preprocesamiento interno de MobileNetV3 (escala [-1, 1])
    x = keras.applications.mobilenet_v3.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="COP_Classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # 4. Callbacks para evitar overfitting y guardar el mejor resultado
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", 
            patience=4, 
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.5, 
            patience=2, 
            verbose=1
        )
    ]

    # 5. Entrenamiento
    print("\n--- Iniciando Entrenamiento ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # 6. Guardar modelo nativo Keras
    model_path = "cop_model.keras"
    model.save(model_path)
    print(f"\n-> Modelo guardado en formato Keras: '{model_path}'")

    # 7. Convertir y Exportar a TensorFlow Lite (.tflite)
    print("\n--- Convirtiendo a TensorFlow Lite (.tflite) ---")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Optimización por defecto para reducir tamaño y acelerar inferencia
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = "cop_model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    file_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"-> ¡Modelo TFLite generado con éxito!: '{tflite_path}' ({file_size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
