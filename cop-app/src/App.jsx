import React, { useRef } from "react";
import { BILL_CONFIG } from "./config/bills";
import { useSpeech } from "./hooks/useSpeech";
import { useCameraTorch } from "./hooks/useCameraTorch";
import { useBillScanner } from "./hooks/useBillScanner";
import { CameraBackground } from "./components/CameraBackground";

export default function App() {
  const videoRef = useRef(null);
  const lastTapTimeRef = useRef(0);
  const touchStartYRef = useRef(0);

  // 1. Voz y Linterna
  const { ttsEnabled, speak, toggleTts, vibrate } = useSpeech();
  const { torchActive, hasTorch, toggleTorch, registerTorch } =
    useCameraTorch();

  // 2. IA e Inferencia
  const { detectedClass, allScores } = useBillScanner(videoRef, speak);
  const currentInfo = BILL_CONFIG[detectedClass] || BILL_CONFIG.background;
  const isDetecting = detectedClass !== "background";

  const themeColors = {
    "2k": { border: "#22d3ee", bg: "rgba(8, 145, 178, 0.25)", text: "#67e8f9" },
    "5k": { border: "#d97706", bg: "rgba(180, 83, 9, 0.25)", text: "#fcd34d" },
    "10k": {
      border: "#e11d48",
      bg: "rgba(190, 18, 60, 0.25)",
      text: "#fda4af",
    },
    "20k": {
      border: "#f97316",
      bg: "rgba(194, 65, 12, 0.25)",
      text: "#fdba74",
    },
    "50k": {
      border: "#9333ea",
      bg: "rgba(126, 34, 206, 0.25)",
      text: "#e9d5ff",
    },
    "100k": {
      border: "#059669",
      bg: "rgba(4, 120, 87, 0.25)",
      text: "#6ee7b7",
    },
    background: {
      border: "#3f3f46",
      bg: "rgba(10, 10, 10, 0.85)",
      text: "#a1a1aa",
    },
  };

  const activeTheme = themeColors[detectedClass] || themeColors.background;

  // 3. Sistema de Gestos Simplificado y Robusto
  const handleTouchStart = (e) => {
    touchStartYRef.current = e.touches[0].clientY;
  };

  const handleTouchEnd = (e) => {
    const deltaY = e.changedTouches[0].clientY - touchStartYRef.current;
    const now = Date.now();
    const timeSinceLastTap = now - lastTapTimeRef.current;

    // A. Gesto de Deslizar (Swipe vertical > 70px) -> Alternar TTS
    if (Math.abs(deltaY) > 70) {
      toggleTts();
      return;
    }

    // B. Gesto de Doble Toque (< 300ms entre toques) -> Alternar Linterna
    if (timeSinceLastTap < 300) {
      lastTapTimeRef.current = 0; // Reset
      vibrate([80, 40, 80]);
      toggleTorch().then((newState) => {
        speak(newState ? "Linterna encendida" : "Linterna apagada", true, 0, 0);
      });
      return;
    }

    // C. Toque Simple -> Repetir / Anunciar estado actual de inmediato (forzado)
    lastTapTimeRef.current = now;
    vibrate(40);

    if (isDetecting) {
      speak(currentInfo.name, true, 0, 40); // force = true para que repita sin restricciones de tiempo
    } else {
      speak("Buscando billete...", true, 0, 30);
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
        touchAction: "manipulation",
      }}
    >
      {/* 1. Cámara Aislada */}
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
          backgroundColor: "rgba(0, 0, 0, 0.60)",
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
              color: activeTheme.border,
              fontWeight: "900",
              letterSpacing: "2px",
              fontSize: "1.2rem",
              textShadow: "0 2px 4px rgba(0,0,0,0.8)",
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
              color: "#fff",
              backgroundColor: "rgba(0,0,0,0.6)",
              padding: "6px 12px",
              borderRadius: "12px",
              border: "1px solid rgba(255,255,255,0.1)",
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
              padding: "36px 20px",
              backgroundColor: activeTheme.bg,
              border: `3px solid ${activeTheme.border}`,
              borderRadius: "28px",
              boxShadow: `0 0 30px ${activeTheme.border}40`,
              backdropFilter: "blur(12px)",
              transition: "all 0.3s ease-in-out",
            }}
          >
            <span
              style={{
                fontSize: "0.8rem",
                fontWeight: "900",
                letterSpacing: "2px",
                textTransform: "uppercase",
                color: activeTheme.text,
                marginBottom: "12px",
                display: "block",
              }}
            >
              {isDetecting ? "BILLETE DETECTADO" : "BUSCANDO BILLETE"}
            </span>

            <h1
              style={{
                fontSize: "2.85rem",
                fontWeight: "900",
                lineHeight: "1.1",
                color: "#ffffff",
                margin: "10px 0",
                textShadow: "0 2px 8px rgba(0,0,0,0.6)",
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
                        ? activeTheme.border
                        : "rgba(24, 24, 27, 0.8)",
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
              backgroundColor: "rgba(24, 24, 27, 0.85)",
              padding: "12px 18px",
              borderRadius: "16px",
              border: "1px solid rgba(255,255,255,0.1)",
              display: "inline-block",
              fontSize: "0.8rem",
              color: "#d4d4d8",
              backdropFilter: "blur(8px)",
            }}
          >
            Toca para repetir • Doble toque para linterna • Desliza para voz
          </div>
        </footer>
      </div>
    </div>
  );
}
