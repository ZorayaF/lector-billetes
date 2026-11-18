import os
import cv2
import random
from pathlib import Path

# --- CONFIGURACIÓN ---
RAW_VIDEOS_DIR = "videos_crudos"
OUTPUT_DIR = "cop_dataset"

# Proporciones de división (deben sumar 1.0)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Extraer 1 fotograma cada FRAME_INTERVAL cuadros
# (A 30 FPS, un valor de 5 extrae 6 imágenes por segundo)
FRAME_INTERVAL = 5

# Semilla fija para reproducibilidad
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Dimensiones estándar para redimensionar (opcional, útil para estandarizar)
TARGET_SIZE = (640, 640)


def crear_estructura_carpetas(base_dir, clases):
    """Crea la jerarquía train/val/test para cada clase."""
    splits = ['train', 'val', 'test']
    for split in splits:
        for clase in clases:
            ruta = Path(base_dir) / split / clase
            ruta.mkdir(parents=True, exist_ok=True)


def procesar_video(ruta_video, intervalo=5):
    """Abre un video y devuelve una lista de fotogramas extraídos."""
    frames = []
    cap = cv2.VideoCapture(str(ruta_video))

    if not cap.isOpened():
        print(f"  [!] Error al abrir el video: {ruta_video}")
        return frames

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % intervalo == 0:
            if TARGET_SIZE is not None:
                frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            frames.append(frame)

        frame_count += 1

    cap.release()
    return frames


def main():
    ruta_crudos = Path(RAW_VIDEOS_DIR)
    
    if not ruta_crudos.exists():
        print(f"Error: No existe el directorio de origen '{RAW_VIDEOS_DIR}'.")
        return

    # Obtener todas las carpetas de clases (2k, 5k, 10k, etc.)
    clases = [d.name for d in ruta_crudos.iterdir() if d.is_dir()]
    
    if not clases:
        print(f"Error: No se encontraron carpetas de clases dentro de '{RAW_VIDEOS_DIR}'.")
        return

    print(f"Clases detectadas ({len(clases)}): {clases}")
    crear_estructura_carpetas(OUTPUT_DIR, clases)

    extensiones_validas = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}
    totales = {'train': 0, 'val': 0, 'test': 0}

    for clase in clases:
        carpeta_clase = ruta_crudos / clase
        videos = [v for v in carpeta_clase.iterdir() if v.suffix.lower() in extensiones_validas]

        if not videos:
            print(f"\n[Aviso] No hay videos en la carpeta '{clase}'. Saltando...")
            continue

        print(f"\nProcesando clase '{clase}' ({len(videos)} videos)...")
        todos_los_frames = []

        # 1. Extraer fotogramas de todos los videos de esta clase
        for video in videos:
            print(f" -> Leyendo {video.name}...")
            frames_extraidos = procesar_video(video, intervalo=FRAME_INTERVAL)
            todos_los_frames.extend(frames_extraidos)

        print(f" -> Total fotogramas extraídos para '{clase}': {len(todos_los_frames)}")

        # 2. Mezclar aleatoriamente los fotogramas
        random.shuffle(todos_los_frames)

        # 3. Calcular puntos de corte para train / val / test
        n_total = len(todos_los_frames)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)

        train_frames = todos_los_frames[:n_train]
        val_frames = todos_los_frames[n_train:n_train + n_val]
        test_frames = todos_los_frames[n_train + n_val:]

        splits_data = {
            'train': train_frames,
            'val': val_frames,
            'test': test_frames
        }

        # 4. Guardar las imágenes en disco
        for split_name, frames_split in splits_data.items():
            carpeta_destino = Path(OUTPUT_DIR) / split_name / clase
            for idx, frame in enumerate(frames_split):
                nombre_archivo = f"{clase}_{split_name}_{idx:05d}.jpg"
                ruta_guardado = carpeta_destino / nombre_archivo
                cv2.imwrite(str(ruta_guardado), frame)
            
            totales[split_name] += len(frames_split)
            print(f"    - {split_name.upper()}: {len(frames_split)} guardadas")

    print("\n--- RESUMEN FINAL ---")
    print(f"Imágenes en Train: {totales['train']}")
    print(f"Imágenes en Val:   {totales['val']}")
    print(f"Imágenes en Test:  {totales['test']}")
    print(f"Total global:      {sum(totales.values())}")
    print(f"Dataset generado con éxito en '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
