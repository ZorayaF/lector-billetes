import { useState, useCallback, useRef } from "react";

export function useSpeech() {
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const lastSpokenTextRef = useRef("");
  const lastSpokenTimeRef = useRef(0);

  // Vibración háptica diferenciada
  const vibrate = useCallback((pattern = 60) => {
    if (navigator.vibrate) {
      navigator.vibrate(pattern);
    }
  }, []);

  const speak = useCallback(
    (text, force = false, debounceMs = 2500, vibrationPattern = 60) => {
      if (!ttsEnabled || !("speechSynthesis" in window) || !text) return;

      const now = Date.now();
      const isSameText = text === lastSpokenTextRef.current;
      const isCoolingDown = now - lastSpokenTimeRef.current < debounceMs;

      if (!force && isSameText && isCoolingDown) {
        return;
      }

      lastSpokenTextRef.current = text;
      lastSpokenTimeRef.current = now;

      // Vibración de confirmación
      vibrate(vibrationPattern);

      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "es-CO";
      utterance.rate = 1.05;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
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
