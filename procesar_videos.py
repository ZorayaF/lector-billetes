import shutil
import random
import cv2
import numpy as np
from pathlib import Path

# --- CONFIGURACIÓN GENERAL ---
RAW_VIDEOS_DIR = "videos_crudos"
OUTPUT_DIR = "cop_dataset"

# Proporciones (70% Train / 15% Val / 15% Test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Extraer 1 fotograma cada 3 cuadros
FRAME_INTERVAL = 2
TARGET_SIZE = (640, 640)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Carpetas que se extraen directo SIN filtros de nitidez/bordes
CLASES_SIN_FILTRO = {"backround", "background", "fondo", "ninguno"}

# --- UMBRALES DE CALIDAD (Solo aplican a carpetas de billetes) ---
MIN_LAPLACIAN_VAR = 25.0
MIN_STD_DEV = 14.0
MIN_BRIGHTNESS = 20.0
MAX_BRIGHTNESS = 245.0
MIN_EDGE_DENSITY = 0.008
MIN_DIFF_PREV = 2.0

# Tamaño de cada micro-bloque temporal para repartir todo el video en train/val/test
CHUNK_SIZE = 6


def limpiar_y_crear_carpetas(base_dir, clases):
    """Limpia el dataset anterior y recrea la jerarquía train/val/test."""
    base_path = Path(base_dir)
    if base_path.exists():
        print(f"[Limpieza] Eliminando dataset anterior en '{base_dir}'...")
        shutil.rmtree(base_path)

    for split in ['train', 'val', 'test']:
        for clase in clases:
            ruta = base_path / split / clase
            ruta.mkdir(parents=True, exist_ok=True)


def redimensionar_sin_deformar(frame, target_size=(640, 640)):
    """
    Aplica Letterbox: conserva el 100% del fotograma original (ej. 1280x720)
    sin recortar el billete ni deformarlo, rellenando los bordes faltantes en negro.
    """
    h, w = frame.shape[:2]
    target_w, target_h = target_size

    # Calcular escala para que el lado más largo quepa completo
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Crear lienzo negro cuadrado y centrar la imagen completa
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def evaluar_calidad_frame(frame_bgr, ultimo_gray=None):
    """Filtra frames vacíos, quemados o borrosos (solo para billetes)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    pad_h, pad_w = int(h * 0.1), int(w * 0.1)
    roi = gray[pad_h:h - pad_h, pad_w:w - pad_w]

    mean_val, std_val = cv2.meanStdDev(roi)
    brillo = float(mean_val[0][0])
    contraste = float(std_val[0][0])

    if brillo < MIN_BRIGHTNESS or brillo > MAX_BRIGHTNESS:
        return False, "exposicion", gray

    if contraste < MIN_STD_DEV:
        return False, "vacio_o_plano", gray

    nitidez = cv2.Laplacian(roi, cv2.CV_64F).var()
    if nitidez < MIN_LAPLACIAN_VAR:
        return False, "borroso", gray

    edges = cv2.Canny(roi, 50, 150)
    densidad_bordes = np.count_nonzero(edges) / edges.size
    if densidad_bordes < MIN_EDGE_DENSITY:
        return False, "sin_detalle", gray

    if ultimo_gray is not None:
        diff = cv2.absdiff(gray, ultimo_gray)
        if np.mean(diff) < MIN_DIFF_PREV:
            return False, "duplicado", gray

    return True, "ok", gray


def procesar_video(ruta_video, intervalo=3, aplicar_filtros=True):
    """Extrae los fotogramas de un video (con o sin filtros según la clase)."""
    frames_validos = []
    stats = {
        "leidos": 0,
        "exposicion": 0,
        "vacio_o_plano": 0,
        "borroso": 0,
        "sin_detalle": 0,
        "duplicado": 0
    }

    cap = cv2.VideoCapture(str(ruta_video))
    if not cap.isOpened():
        print(f"  [!] Error al abrir el video: {ruta_video}")
        return frames_validos, stats

    frame_count = 0
    ultimo_gray_guardado = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % intervalo == 0:
            stats["leidos"] += 1
            frame_proc = redimensionar_sin_deformar(frame, TARGET_SIZE)

            if not aplicar_filtros:
                # Extracción directa sin filtros para backround
                frames_validos.append(frame_proc)
            else:
                es_valido, motivo, gray_actual = evaluar_calidad_frame(frame_proc, ultimo_gray_guardado)
                if es_valido:
                    frames_validos.append(frame_proc)
                    ultimo_gray_guardado = gray_actual
                else:
                    stats[motivo] += 1

        frame_count += 1

    cap.release()
    return frames_validos, stats


def dividir_por_segmentos(frames, train_r, val_r, chunk_size=6):
    """Divide los fotogramas en segmentos a lo largo del video y los reparte en train/val/test."""
    if len(frames) < chunk_size * 3:
        chunk_size = max(1, len(frames) // 6)

    chunks = [frames[i:i + chunk_size] for i in range(0, len(frames), chunk_size)]
    random.shuffle(chunks)

    n_chunks = len(chunks)
    n_val = max(1, int(round(n_chunks * val_r)))
    n_test = max(1, int(round(n_chunks * (1.0 - train_r - val_r))))
    n_train = max(1, n_chunks - n_val - n_test)

    train_chunks = chunks[:n_train]
    val_chunks = chunks[n_train:n_train + n_val]
    test_chunks = chunks[n_train + n_val:]

    train_frames = [f for chunk in train_chunks for f in chunk]
    val_frames = [f for chunk in val_chunks for f in chunk]
    test_frames = [f for chunk in test_chunks for f in chunk]

    return train_frames, val_frames, test_frames


def main():
    ruta_crudos = Path(RAW_VIDEOS_DIR)
    if not ruta_crudos.exists():
        print(f"Error: No existe el directorio de origen '{RAW_VIDEOS_DIR}'.")
        return

    clases = sorted([d.name for d in ruta_crudos.iterdir() if d.is_dir()])
    if not clases:
        print(f"Error: No se encontraron carpetas de clases dentro de '{RAW_VIDEOS_DIR}'.")
        return

    print(f"Clases detectadas ({len(clases)}): {clases}")
    limpiar_y_crear_carpetas(OUTPUT_DIR, clases)

    extensiones_validas = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}
    totales = {'train': 0, 'val': 0, 'test': 0}

    for clase in clases:
        carpeta_clase = ruta_crudos / clase
        videos = sorted([v for v in carpeta_clase.iterdir() if v.suffix.lower() in extensiones_validas])

        if not videos:
            print(f"\n[Aviso] No hay videos en la carpeta '{clase}'. Saltando...")
            continue

        usar_filtros = clase.lower() not in CLASES_SIN_FILTRO
        modo_str = "CON FILTROS" if usar_filtros else "SIN FILTROS (EXTRACCIÓN DIRECTA)"
        print(f"\n=== Procesando clase '{clase}' ({len(videos)} videos) [{modo_str}] ===")

        todos_frames_clase = []
        rechazos_clase = {"exposicion": 0, "vacio_o_plano": 0, "borroso": 0, "sin_detalle": 0, "duplicado": 0}
        candidatos_totales = 0

        for video in videos:
            frames_video, stats = procesar_video(
                video, intervalo=FRAME_INTERVAL, aplicar_filtros=usar_filtros
            )
            candidatos_totales += stats["leidos"]
            for k in rechazos_clase:
                rechazos_clase[k] += stats[k]

            print(f" -> {video.name}: {len(frames_video)} extraídos de {stats['leidos']} candidatos")
            todos_frames_clase.extend(frames_video)

        if not todos_frames_clase:
            print(f" [!] Ningún fotograma extraído en la clase '{clase}'.")
            continue

        tr, va, te = dividir_por_segmentos(
            todos_frames_clase, TRAIN_RATIO, VAL_RATIO, chunk_size=CHUNK_SIZE
        )
        splits_clase = {'train': tr, 'val': va, 'test': te}

        if usar_filtros:
            print(f" [Resumen '{clase}'] Aceptados: {len(todos_frames_clase)}/{candidatos_totales} | "
                  f"Descartados -> Borrosos: {rechazos_clase['borroso']}, "
                  f"Vacíos/Planos: {rechazos_clase['vacio_o_plano']}, "
                  f"Exposición: {rechazos_clase['exposicion']}, "
                  f"Sin detalle: {rechazos_clase['sin_detalle']}, "
                  f"Duplicados: {rechazos_clase['duplicado']}")

        for split_name, frames_split in splits_clase.items():
            carpeta_destino = Path(OUTPUT_DIR) / split_name / clase
            for idx, frame in enumerate(frames_split):
                nombre_archivo = f"{clase}_{split_name}_{idx:05d}.jpg"
                ruta_guardado = carpeta_destino / nombre_archivo
                cv2.imwrite(str(ruta_guardado), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            totales[split_name] += len(frames_split)
            print(f"    - {split_name.upper()}: {len(frames_split)} guardadas")

    print("\n================ RESUMEN FINAL ================")
    print(f"Imágenes en Train: {totales['train']}")
    print(f"Imágenes en Val:   {totales['val']}")
    print(f"Imágenes en Test:  {totales['test']}")
    print(f"Total global:      {sum(totales.values())}")
    print(f"Dataset generado con éxito en '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
