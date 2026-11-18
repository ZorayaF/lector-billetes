import React, { useRef } from "react";
import { BILL_CONFIG } from "./config/bills";
import { useSpeech } from "./hooks/useSpeech";
import { useCameraTorch } from "./hooks/useCameraTorch";
import { useBillScanner } from "./hooks/useBillScanner";
import { CameraBackground } from "./components/CameraBackground";

export default function App() {
  const videoRef = useRef(null);
  const touchStartYRef = useRef(0);
  const touchStartTimeRef = useRef(0);
  const longPressTimerRef = useRef(null);
  const isLongPressRef = useRef(false);

  // 1. Voz y Linterna
  const { ttsEnabled, speak, toggleTts, vibrate } = useSpeech();
  const { torchActive, hasTorch, toggleTorch, registerTorch } =
    useCameraTorch();

  // 2. IA e Inferencia
  const { detectedClass, allScores } = useBillScanner(videoRef, speak);
  const currentInfo = BILL_CONFIG[detectedClass] || BILL_CONFIG.background;
  const isDetecting = detectedClass !== "background";

  // 3. Gestos globales
  const handleTouchStart = (e) => {
    touchStartYRef.current = e.touches[0].clientY;
    touchStartTimeRef.current = Date.now();
    isLongPressRef.current = false;

    longPressTimerRef.current = setTimeout(async () => {
      isLongPressRef.current = true;
      vibrate([100, 60, 100]);
      const newState = await toggleTorch();
      speak(newState ? "Linterna encendida" : "Linterna apagada", true);
    }, 800);
  };

  const handleTouchEnd = (e) => {
    if (longPressTimerRef.current) {
      clearTimeout(longPressTimerRef.current);
    }

    if (isLongPressRef.current) return;

    const deltaY = e.changedTouches[0].clientY - touchStartYRef.current;
    const duration = Date.now() - touchStartTimeRef.current;

    if (Math.abs(deltaY) > 80) {
      toggleTts();
      return;
    }

    if (duration < 350) {
      vibrate(40);
      if (isDetecting) {
        speak(currentInfo.name, true, 0, 50);
      } else {
        speak("Buscando billete...", true, 0, 30);
      }
    }
  };

  return (
    <div
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
      role="status"
      aria-live="polite"
      style={{
        position: "relative",
        width: "100vw",
        height: "100vh",
        backgroundColor: "#000",
        overflow: "hidden",
      }}
    >
      {/* 1. Cámara Aislada (Nunca se re-renderiza con la IA) */}
      <CameraBackground
        videoRef={videoRef}
        onTorchReady={registerTorch}
        onReady={() =>
          speak("Lector listo. Apunte hacia el billete.", true, 0, 80)
        }
      />

      {/* 2. Película oscura protectora */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundColor: "rgba(0, 0, 0, 0.65)",
          zIndex: 1,
          pointerEvents: "none",
        }}
      />

      {/* 3. Capa de Texto y Detección */}
      <div
        style={{
          position: "relative",
          zIndex: 10,
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
        }}
      >
        <header
          style={{
            padding: "20px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <span
            style={{
              color: "#34d399",
              fontWeight: "900",
              letterSpacing: "2px",
              fontSize: "1.2rem",
            }}
          >
            COP LECTOR
          </span>
          <div
            style={{
              display: "flex",
              gap: "15px",
              fontSize: "0.85rem",
              fontFamily: "monospace",
              opacity: 0.8,
            }}
          >
            <span>{hasTorch && torchActive ? "LUZ: ON" : "LUZ: OFF"}</span>
            <span>{ttsEnabled ? "VOZ: ON" : "VOZ: OFF"}</span>
          </div>
        </header>

        <main
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            padding: "20px",
            textAlign: "center",
          }}
        >
          <div
            style={{
              width: "100%",
              maxWidth: "480px",
              padding: "32px 20px",
              backgroundColor: "rgba(10, 10, 10, 0.85)",
              border: "2px solid #3f3f46",
              borderRadius: "24px",
              boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.5)",
            }}
          >
            <span
              style={{
                fontSize: "0.75rem",
                fontWeight: "900",
                letterSpacing: "2px",
                textTransform: "uppercase",
                color: "#a1a1aa",
                marginBottom: "12px",
                display: "block",
              }}
            >
              {isDetecting ? "BILLETE DETECTADO" : "BUSCANDO BILLETE"}
            </span>

            <h1
              style={{
                fontSize: "2.75rem",
                fontWeight: "900",
                lineHeight: "1.1",
                color: "#ffffff",
                margin: "10px 0",
              }}
            >
              {currentInfo.name}
            </h1>

            <div
              style={{
                marginTop: "24px",
                display: "flex",
                flexWrap: "wrap",
                justifyContent: "center",
                gap: "8px",
              }}
            >
              {Object.entries(allScores).map(([label, score]) => (
                <span
                  key={label}
                  style={{
                    padding: "6px 12px",
                    borderRadius: "9999px",
                    fontSize: "0.75rem",
                    fontFamily: "monospace",
                    fontWeight: "bold",
                    backgroundColor:
                      label === detectedClass && isDetecting
                        ? "#34d399"
                        : "#18181b",
                    color:
                      label === detectedClass && isDetecting
                        ? "#000000"
                        : "#a1a1aa",
                    border:
                      label === detectedClass && isDetecting
                        ? "none"
                        : "1px solid #27272a",
                  }}
                >
                  {label}: {(score * 100).toFixed(0)}%
                </span>
              ))}
            </div>
          </div>
        </main>

        <footer style={{ padding: "20px", textAlign: "center" }}>
          <div
            style={{
              backgroundColor: "rgba(24, 24, 27, 0.9)",
              padding: "12px 18px",
              borderRadius: "16px",
              border: "1px solid #27272a",
              display: "inline-block",
              fontSize: "0.8rem",
              color: "#d4d4d8",
            }}
          >
            Toca para repetir • Mantén presionado para linterna • Desliza para
            silenciar
          </div>
        </footer>
      </div>
    </div>
  );
}
