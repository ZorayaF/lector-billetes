import React, { useEffect, useRef, memo } from "react";
import { App as CapApp } from "@capacitor/app";

export const CameraBackground = memo(function CameraBackground({
  videoRef,
  onTorchReady,
  onReady,
}) {
  const localVideoRef = useRef(null);
  const trackRef = useRef(null);

  useEffect(() => {
    let stream = null;

    async function initCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "environment",
            width: { ideal: 640 },
            height: { ideal: 480 },
          },
          audio: false,
        });

        const target = videoRef ? videoRef.current : localVideoRef.current;
        if (target) {
          target.srcObject = stream;
        }

        const videoTrack = stream.getVideoTracks()[0];
        trackRef.current = videoTrack;

        const capabilities = videoTrack.getCapabilities?.() || {};
        const hasTorch = Boolean(capabilities.torch);

        if (onTorchReady) {
          onTorchReady({
            hasTorch,
            setTorch: async (state) => {
              if (trackRef.current && hasTorch) {
                await trackRef.current.applyConstraints({
                  advanced: [{ torch: state }],
                });
                return true;
              }
              return false;
            },
          });
        }

        if (onReady) onReady();
      } catch (err) {
        console.error("Error iniciando cámara:", err);
      }
    }

    initCamera();

    const stateListener = CapApp.addListener(
      "appStateChange",
      ({ isActive }) => {
        if (!isActive && trackRef.current) {
          try {
            trackRef.current.applyConstraints({ advanced: [{ torch: false }] });
          } catch (_) {}
        }
      },
    );

    return () => {
      stateListener.then((sub) => sub.remove());
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
      }
    };
  }, []); // [] = Se monta 1 sola vez en toda la vida de la app

  return (
    <video
      ref={videoRef || localVideoRef}
      autoPlay
      playsInline
      muted
      webkit-playsinline="true"
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100vw",
        height: "100vh",
        objectFit: "cover",
        zIndex: 0,
        pointerEvents: "none",
        backgroundColor: "#000",
      }}
    />
  );
});
