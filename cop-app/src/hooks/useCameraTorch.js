import { useState, useRef, useCallback } from "react";

export function useCameraTorch() {
  const [torchActive, setTorchActive] = useState(false);
  const [hasTorch, setHasTorch] = useState(false);
  const torchControllerRef = useRef(null);

  const registerTorch = useCallback(({ hasTorch: supported, setTorch }) => {
    setHasTorch(supported);
    torchControllerRef.current = setTorch;
  }, []);

  const toggleTorch = useCallback(async () => {
    if (!torchControllerRef.current || !hasTorch) return false;
    const nextState = !torchActive;
    const success = await torchControllerRef.current(nextState);
    if (success) {
      setTorchActive(nextState);
    }
    return nextState;
  }, [torchActive, hasTorch]);

  return { torchActive, hasTorch, toggleTorch, registerTorch };
}
