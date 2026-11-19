import { useState, useEffect, useRef } from "react";
import { loadOnnxModel, runInference } from "../services/onnxClassifier";
import { BILL_CONFIG } from "../config/bills";

export function useBillScanner(videoRef, speak) {
  const [detectedClass, setDetectedClass] = useState("background");
  const [confidence, setConfidence] = useState(0);
  const [allScores, setAllScores] = useState({});
  const [modelReady, setModelReady] = useState(false);

  // Referencias para estabilidad (histéresis)
  const historyRef = useRef([]);
  const lastSpokenBillRef = useRef(null);
  const backgroundCounterRef = useRef(0);
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

            // 1. Filtrar por alta confianza (> 80%) y descartar background preliminar
            const conf = result.confidence;
            const rawClass = result.className;

            // Agregamos al historial de las últimas 4 lecturas para evitar titubeos
            historyRef.current.push({ className: rawClass, confidence: conf });
            if (historyRef.current.length > 4) {
              historyRef.current.shift();
            }

            // Votación: verificar si las últimas lecturas coinciden en la misma denominación
            const validDeltas = historyRef.current.filter(
              (h) => h.className !== "background" && h.confidence > 0.8,
            );

            // Buscamos si hay un ganador claro en el historial reciente
            const counts = {};
            validDeltas.forEach((item) => {
              counts[item.className] = (counts[item.className] || 0) + 1;
            });

            let stableClass = "background";
            let maxCount = 0;
            for (const [cls, cnt] of Object.entries(counts)) {
              if (cnt >= 3) {
                // Al menos 3 de 4 lecturas deben coincidir
                stableClass = cls;
                maxCount = cnt;
              }
            }

            if (stableClass !== "background") {
              backgroundCounterRef.current = 0;
              setDetectedClass(stableClass);
              setConfidence(conf);

              // Hablar SOLAMENTE si es un billete nuevo o si pasó un buen rato,
              // evitando que hable constantemente encima de lo mismo
              if (lastSpokenBillRef.current !== stableClass) {
                lastSpokenBillRef.current = stableClass;
                const info = BILL_CONFIG[stableClass];
                if (info && speak) {
                  speak(info.name, true, 4000, 100); // Fuerza lectura clara
                }
              }
            } else {
              // Si no hay certeza, aumentamos contador de fondo
              backgroundCounterRef.current += 1;

              // Solo cambiamos a background si lleva varios ciclos sin ver nada (evita parpadeo)
              if (backgroundCounterRef.current >= 3) {
                setDetectedClass("background");
                setConfidence(result.allScores["background"] || conf);

                // Permitimos que si vuelve a aparecer el mismo billete lo vuelva a anunciar después
                if (backgroundCounterRef.current > 10) {
                  lastSpokenBillRef.current = null;
                }
              }
            }
          }
        } catch (err) {
          console.error("Error en inferencia:", err);
        }
      }

      // Ciclo de muestreo cada 200 ms
      if (isMounted) {
        timerId = setTimeout(processLoop, 200);
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
