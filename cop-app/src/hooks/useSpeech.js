import { useState, useCallback, useRef } from "react";
import { TextToSpeech } from "@capacitor-community/text-to-speech";

export function useSpeech() {
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const lastSpokenTextRef = useRef("");
  const lastSpokenTimeRef = useRef(0);

  const vibrate = useCallback((pattern = 60) => {
    if (navigator.vibrate) {
      navigator.vibrate(pattern);
    }
  }, []);

  const speak = useCallback(
    async (text, force = false, debounceMs = 2500, vibrationPattern = 60) => {
      if (!ttsEnabled || !text) return;

      const now = Date.now();
      const isSameText = text === lastSpokenTextRef.current;
      const isCoolingDown = now - lastSpokenTimeRef.current < debounceMs;

      if (!force && isSameText && isCoolingDown) {
        return;
      }

      lastSpokenTextRef.current = text;
      lastSpokenTimeRef.current = now;

      // Vibración de confirmación háptica
      vibrate(vibrationPattern);

      try {
        // Usar el plugin nativo de Capacitor (funciona al 100% en dispositivos reales)
        await TextToSpeech.speak({
          text: text,
          lang: "es-CO",
          rate: 1.0,
          pitch: 1.0,
          volume: 1.0,
          category: "ambient",
        });
      } catch (err) {
        // Fallback por si se prueba en navegador web de escritorio común
        if ("speechSynthesis" in window) {
          window.speechSynthesis.cancel();
          const utterance = new SpeechSynthesisUtterance(text);
          utterance.lang = "es-CO";
          window.speechSynthesis.speak(utterance);
        } else {
          console.error("Error en TTS nativo y web:", err);
        }
      }
    },
    [ttsEnabled, vibrate],
  );

  const toggleTts = useCallback(() => {
    setTtsEnabled((prev) => {
      const next = !prev;
      vibrate(150);
      if (next) {
        speak("Voz activada", true, 0, 150);
      }
      return next;
    });
  }, [speak, vibrate]);

  return { ttsEnabled, speak, toggleTts, vibrate };
}
