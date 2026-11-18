import { useState, useEffect, useRef } from "react";
import { loadOnnxModel, runInference } from "../services/onnxClassifier";
import { BILL_CONFIG } from "../config/bills";

export function useBillScanner(videoRef, speak) {
  const [detectedClass, setDetectedClass] = useState("background");
  const [confidence, setConfidence] = useState(0);
  const [allScores, setAllScores] = useState({});
  const [modelReady, setModelReady] = useState(false);
  const backgroundTimeRef = useRef(Date.now());
  const guideAnnouncedRef = useRef(false);

  // Canvas aislado 100% en memoria (fuera del DOM)
  const memoryCanvasRef = useRef(null);

  useEffect(() => {
    if (!memoryCanvasRef.current) {
      const c = document.createElement("canvas");
      c.width = 224;
      c.height = 224;
      memoryCanvasRef.current = c;
    }

    loadOnnxModel()
      .then(() => setModelReady(true))
      .catch((err) => console.error("Error cargando ONNX:", err));
  }, []);

  useEffect(() => {
    let isMounted = true;
    let timerId = null;

    async function processLoop() {
      if (!isMounted) return;

      if (
        modelReady &&
        videoRef.current &&
        videoRef.current.readyState >= 2 &&
        memoryCanvasRef.current
      ) {
        try {
          const result = await runInference(
            videoRef.current,
            memoryCanvasRef.current,
          );

          if (isMounted) {
            setAllScores(result.allScores);

            if (result.confidence > 0.65 && result.className !== "background") {
              setDetectedClass(result.className);
              setConfidence(result.confidence);
              backgroundTimeRef.current = Date.now();
              guideAnnouncedRef.current = false;

              const info = BILL_CONFIG[result.className];
              if (info && speak) {
                speak(info.name, false, 2500, 70);
              }
            } else {
              setDetectedClass("background");
              setConfidence(
                result.allScores["background"] || result.confidence,
              );

              const elapsed = Date.now() - backgroundTimeRef.current;
              if (elapsed > 6000 && !guideAnnouncedRef.current) {
                guideAnnouncedRef.current = true;
                if (speak) {
                  speak(
                    "Buscando billete. Mueva el billete frente a la cámara.",
                    false,
                    8000,
                    0,
                  );
                }
              }
            }
          }
        } catch (err) {
          console.error("Error en inferencia:", err);
        }
      }

      // 180 ms de descanso entre inferencias completadas (evita colapso de GPU y batería)
      if (isMounted) {
        timerId = setTimeout(processLoop, 180);
      }
    }

    if (modelReady) {
      processLoop();
    }

    return () => {
      isMounted = false;
      if (timerId) clearTimeout(timerId);
    };
  }, [modelReady, videoRef, speak]);

  return { detectedClass, confidence, allScores, modelReady };
}
