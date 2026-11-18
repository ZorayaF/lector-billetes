import * as ort from "onnxruntime-web";

// Configuración de rutas estáticas compatibles con Vite y Capacitor
const isBrowser = typeof window !== "undefined";
const basePath = isBrowser ? `${window.location.origin}/wasm/` : "/wasm/";

ort.env.wasm.wasmPaths = basePath;
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

let session = null;
let labelsMap = ["100k", "10k", "20k", "2k", "50k", "5k", "background"];

export async function loadOnnxModel() {
  if (session) return session;

  try {
    const labelRes = await fetch("/labels.json");
    if (labelRes.ok) {
      labelsMap = await labelRes.json();
    }

    session = await ort.InferenceSession.create("/cop_model.onnx", {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });

    console.log("-> Sesión ONNX lista. Mapeo:", labelsMap);
    return session;
  } catch (err) {
    console.error("Error al cargar sesión ONNX:", err);
    throw err;
  }
}

export async function runInference(videoElement, canvasElement) {
  if (!session || !videoElement || videoElement.readyState < 2) {
    return { className: "background", confidence: 0, allScores: {} };
  }

  const ctx = canvasElement.getContext("2d", { willReadFrequently: true });
  const vw = videoElement.videoWidth || 640;
  const vh = videoElement.videoHeight || 480;

  const size = Math.min(vw, vh);
  const sx = (vw - size) / 2;
  const sy = (vh - size) / 2;

  ctx.drawImage(videoElement, sx, sy, size, size, 0, 0, 224, 224);
  const { data } = ctx.getImageData(0, 0, 224, 224);

  const float32Data = new Float32Array(1 * 224 * 224 * 3);
  let j = 0;
  for (let i = 0; i < data.length; i += 4) {
    float32Data[j] = data[i];
    float32Data[j + 1] = data[i + 1];
    float32Data[j + 2] = data[i + 2];
    j += 3;
  }

  try {
    const inputTensor = new ort.Tensor(
      "float32",
      float32Data,
      [1, 224, 224, 3],
    );
    const inputName = session.inputNames[0];
    const feeds = { [inputName]: inputTensor };

    const results = await session.run(feeds);
    const outputName = session.outputNames[0];
    const probabilities = results[outputName].data;

    const allScores = {};
    let maxProb = -Infinity;
    let maxIndex = 0;

    for (let i = 0; i < labelsMap.length; i++) {
      const score = probabilities[i] || 0;
      const label = labelsMap[i] || `Clase_${i}`;
      allScores[label] = score;
      if (score > maxProb) {
        maxProb = score;
        maxIndex = i;
      }
    }

    const predictedClass = labelsMap[maxIndex] || "background";

    return {
      className: predictedClass,
      confidence: maxProb,
      allScores,
    };
  } catch (err) {
    console.error("Error en inferencia ONNX:", err);
    return { className: "background", confidence: 0, allScores: {} };
  }
}
