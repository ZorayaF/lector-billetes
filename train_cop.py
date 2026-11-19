import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# --- CONFIGURACIÓN ---
DATASET_DIR = "cop_dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Reducido de 32 a 16 para mejorar actualización de gradientes en dataset compacto
EPOCHS_HEAD = 15
EPOCHS_FINETUNE = 15
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
SEED = 42


def calcular_pesos_clases(train_dir, class_names):
    """
    Calcula class_weight inversamente proporcional a la cantidad de imágenes
    para compensar clases con menos muestras (como 100k o 10k).
    """
    conteos = []
    for cls in class_names:
        cls_path = Path(train_dir) / cls
        n_imgs = len(list(cls_path.glob("*.jpg"))) + len(list(cls_path.glob("*.png")))
        conteos.append(max(1, n_imgs))

    total = sum(conteos)
    n_clases = len(class_names)
    class_weights = {}

    print("\n--- Balance y Pesos por Clase (Train) ---")
    for idx, (cls, count) in enumerate(zip(class_names, conteos)):
        # Fórmula balanceada estándar: total / (n_clases * count)
        peso = total / (n_clases * count)
        class_weights[idx] = peso
        print(f"  [{idx}] {cls:<12}: {count:>3} imgs -> peso: {peso:.2f}")

    return class_weights


def evaluar_en_test(model, test_ds, class_names):
    """Evalúa el modelo final en el conjunto de Test e imprime matriz de confusión por consola."""
    print("\n================ EVALUACIÓN EN TEST ================")
    y_true = []
    y_pred = []

    for imgs, labels in test_ds:
        preds = model.predict(imgs, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc_global = np.mean(y_true == y_pred) * 100
    print(f"Precisión Global en Test: {acc_global:.2f}% ({np.sum(y_true == y_pred)}/{len(y_true)} aciertos)\n")

    # Matriz de confusión en texto
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    header = "Real \\ Pred | " + " ".join([f"{c[:6]:>6}" for c in class_names]) + " | Recall"
    print(header)
    print("-" * len(header))
    for i, cls in enumerate(class_names):
        fila_str = " ".join([f"{cm[i, j]:>6}" for j in range(n)])
        total_fila = np.sum(cm[i, :])
        recall = (cm[i, i] / total_fila * 100) if total_fila > 0 else 0.0
        print(f"{cls:<11} | {fila_str} | {recall:>5.1f}%")


def main():
    train_dir = Path(DATASET_DIR) / "train"
    val_dir = Path(DATASET_DIR) / "val"
    test_dir = Path(DATASET_DIR) / "test"

    if not train_dir.exists():
        print(f"Error: No existe {train_dir}. Ejecuta primero procesar_videos.py")
        return

    # 1. Cargar datasets desde carpetas
    print("--- Cargando Datasets ---")
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
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

    # Calcular pesos para compensar el desbalance de imágenes entre billetes
    class_weights = calcular_pesos_clases(train_dir, class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 2. Capas de Data Augmentation (solo se activan durante model.fit)
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ], name="data_augmentation")

    # 3. Construir arquitectura con MobileNetV3Small
    # NOTA CRÍTICA: include_preprocessing=True ya viene activo por defecto en MobileNetV3Small
    # y escala automáticamente de [0, 255] a [-1, 1]. NO se debe llamar preprocess_input afuera.
    print("\n--- Construyendo Modelo MobileNetV3Small ---")
    base_model = keras.applications.MobileNetV3Small(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        minimalistic=False,
        include_preprocessing=True
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_image")
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="COP_Classifier")

    # --- FASE 1: Entrenar solo el cabezal clasificador ---
    print("\n--- FASE 1: Entrenando Cabezal Clasificador ---")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_HEAD),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_fase1 = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
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

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        class_weight=class_weights,
        callbacks=callbacks_fase1
    )

    # --- FASE 2: Fine-Tuning (Descongelar últimas capas del backbone) ---
    print("\n--- FASE 2: Fine-Tuning de MobileNetV3Small ---")
    base_model.trainable = True

    # Congelamos todas menos las últimas 35 capas, y mantenemos BatchNormalization congelado
    for layer in base_model.layers[:-35]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_FINETUNE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_fase2 = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINETUNE,
        class_weight=class_weights,
        callbacks=callbacks_fase2
    )

    # 4. Evaluar en conjunto Test
    evaluar_en_test(model, test_ds, class_names)

    # 5. Construir modelo limpio de inferencia (sin capas de Data Augmentation) para exportar a ONNX/TFLite
    print("\n--- Empaquetando Modelo de Inferencia (sin capas Random*) ---")
    infer_inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_image")
    infer_x = base_model(infer_inputs, training=False)
    infer_x = model.layers[-3](infer_x)  # GlobalAveragePooling2D
    infer_x = model.layers[-2](infer_x)  # BatchNormalization
    # Omitimos Dropout en el grafo de inferencia o lo pasamos en training=False
    infer_outputs = model.layers[-1](infer_x)  # Dense softmax
    inference_model = keras.Model(infer_inputs, infer_outputs, name="COP_Classifier_Inference")

    model_path = "cop_model.keras"
    inference_model.save(model_path)
    print(f"-> Modelo limpio guardado en formato Keras: '{model_path}'")

    # 6. Convertir y Exportar a TensorFlow Lite (.tflite)
    print("\n--- Convirtiendo a TensorFlow Lite (.tflite) ---")
    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = "cop_model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    file_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"-> ¡Modelo TFLite generado con éxito!: '{tflite_path}' ({file_size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
