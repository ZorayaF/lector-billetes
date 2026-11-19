import shutil
import cv2
import numpy as np
from pathlib import Path

# --- CONFIGURACIÓN ---
RAW_VIDEOS_DIR = "videos_crudos"
OUTPUT_DIR = "fotos_para_etiquetar"

FOTOS_POR_VIDEO = 15      # Ajusta según cuántos videos grabes (ej. 3-4 videos x 15 = 45-60 fotos)
MAX_DIMENSION = 1024      # Lado más largo en px (rápido para MakeSense.ai y nítido para números)
EXTENSIONES_VALIDAS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}


def redimensionar_proporcional(frame, max_dim=1024):
    """Redimensiona conservando la proporción original (sin recortes ni bordes negros)."""
    h, w = frame.shape[:2]
    mayor = max(h, w)
    if mayor <= max_dim:
        return frame
    escala = max_dim / mayor
    return cv2.resize(frame, (int(w * escala), int(h * escala)), interpolation=cv2.INTER_AREA)


def calcular_nitidez(frame_bgr):
    """Calcula la nitidez mediante la varianza del Laplaciano."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extraer_mejores_de_video(ruta_video, cuota):
    """Divide el video en segmentos iguales y extrae el fotograma más nítido de cada uno."""
    cap = cv2.VideoCapture(str(ruta_video))
    if not cap.isOpened():
        print(f"  [!] No se pudo abrir: {ruta_video.name}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    cuota_real = min(cuota, total_frames)
    limites = np.linspace(0, total_frames, cuota_real + 1, dtype=int)
    seleccionados = []

    for i in range(cuota_real):
        inicio, fin = limites[i], limites[i + 1]
        if inicio >= fin:
            continue

        paso = max(1, (fin - inicio) // 5)
        mejor_frame = None
        mejor_nitidez = -1.0

        for idx_frame in range(inicio, fin, paso):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx_frame)
            ret, frame = cap.read()
            if not ret:
                break
            nitidez = calcular_nitidez(frame)
            if nitidez > mejor_nitidez:
                mejor_nitidez = nitidez
                mejor_frame = frame

        if mejor_frame is not None:
            seleccionados.append(redimensionar_proporcional(mejor_frame, MAX_DIMENSION))

    cap.release()
    return seleccionados


def main():
    ruta_crudos = Path(RAW_VIDEOS_DIR)
    ruta_salida = Path(OUTPUT_DIR)

    if not ruta_crudos.exists():
        print(f"Error: No existe '{RAW_VIDEOS_DIR}'.")
        return

    if ruta_salida.exists():
        shutil.rmtree(ruta_salida)
    ruta_salida.mkdir(parents=True, exist_ok=True)

    # Busca todos los videos en videos_crudos (sueltos o en subcarpetas)
    videos = sorted([
        v for v in ruta_crudos.rglob("*")
        if v.is_file() and v.suffix.lower() in EXTENSIONES_VALIDAS
    ])

    if not videos:
        print(f"No se encontraron videos en '{RAW_VIDEOS_DIR}'.")
        return

    print(f"Videos encontrados: {len(videos)} | Extrayendo hasta {FOTOS_POR_VIDEO} fotos por video...\n")
    total_guardadas = 0

    for idx_vid, video in enumerate(videos, start=1):
        frames = extraer_mejores_de_video(video, FOTOS_POR_VIDEO)
        for idx_img, frame in enumerate(frames):
            nombre = f"vid{idx_vid:02d}_{video.stem}_{idx_img:03d}.jpg"
            cv2.imwrite(str(ruta_salida / nombre), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            total_guardadas += 1

        print(f" -> {video.name}: {len(frames)} fotos extraídas")

    print("\n================ LISTO PARA MAKESENSE.AI ================")
    print(f"Total de imágenes guardadas: {total_guardadas}")
    print(f"Carpeta lista para subir:    '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
