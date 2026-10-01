"use client";

import Image from "next/image";
import { ChangeEvent, useCallback, useEffect, useRef, useState } from "react";

import { predictWasteImage, wasteCategories } from "@/lib/prediction";
import type { PredictionResponse, WasteCategory } from "@/lib/prediction";

type InputMode = "camera" | "upload";
type CameraStatus = "idle" | "starting" | "ready" | "error";

const allowedImageTypes = new Set(["image/jpeg", "image/png", "image/webp"]);

const categoryDetails: Record<
  WasteCategory,
  { name: string; guidance: string; color: string }
> = {
  general: {
    name: "General Waste",
    guidance:
      "Use the general waste stream when the item does not belong in a supported recyclable or organic category.",
    color: "bg-stone-500",
  },
  metal: {
    name: "Metal",
    guidance: "Place in the metal or recycling stream when accepted locally.",
    color: "bg-slate-500",
  },
  organic: {
    name: "Organic",
    guidance: "Place in an organic or food-waste stream where available.",
    color: "bg-lime-600",
  },
  paper: {
    name: "Paper / Cardboard",
    guidance: "Keep paper and cardboard clean and dry before recycling.",
    color: "bg-amber-600",
  },
  plastic: {
    name: "Plastic",
    guidance:
      "Place in the plastic or recycling stream when accepted locally and reasonably clean.",
    color: "bg-sky-600",
  },
};

export default function ScanInterface() {
  const [inputMode, setInputMode] = useState<InputMode>("camera");
  const [cameraStatus, setCameraStatus] = useState<CameraStatus>("idle");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const previewUrlRef = useRef<string | null>(null);
  const cameraRequestIdRef = useRef(0);
  const predictionRequestRef = useRef<AbortController | null>(null);

  const stopCamera = useCallback((updateStatus = true) => {
    cameraRequestIdRef.current += 1;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (updateStatus) {
      setCameraStatus("idle");
    }
  }, []);

  const clearPreview = useCallback(() => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    setPreviewUrl(null);
  }, []);

  const showPreview = useCallback(
    (file: File) => {
      clearPreview();
      const objectUrl = URL.createObjectURL(file);
      previewUrlRef.current = objectUrl;
      setPreviewUrl(objectUrl);
      setSelectedFile(file);
    },
    [clearPreview],
  );

  const clearResult = useCallback(() => {
    predictionRequestRef.current?.abort();
    predictionRequestRef.current = null;
    setIsPredicting(false);
    setPrediction(null);
    setErrorMessage(null);
  }, []);

  const startCamera = useCallback(async () => {
    clearResult();
    clearPreview();
    setSelectedFile(null);
    stopCamera(false);

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraStatus("error");
      setErrorMessage("This browser does not support camera access. Upload an image instead.");
      return;
    }

    setCameraStatus("starting");
    const requestId = ++cameraRequestIdRef.current;

    try {
      const stream = await requestCameraStream();
      if (requestId !== cameraRequestIdRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      streamRef.current = stream;
      if (!videoRef.current) {
        throw new Error("Camera preview is unavailable.");
      }
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      if (requestId !== cameraRequestIdRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      setCameraStatus("ready");
    } catch (error) {
      if (requestId !== cameraRequestIdRef.current) {
        return;
      }
      stopCamera(false);
      setCameraStatus("error");
      setErrorMessage(getCameraErrorMessage(error));
    }
  }, [clearPreview, clearResult, stopCamera]);

  const requestPrediction = useCallback(async (file: File) => {
    predictionRequestRef.current?.abort();
    const controller = new AbortController();
    predictionRequestRef.current = controller;
    setPrediction(null);
    setErrorMessage(null);
    setIsPredicting(true);

    try {
      const result = await predictWasteImage(file, controller.signal);
      if (!controller.signal.aborted) {
        setPrediction(result);
      }
    } catch (error) {
      if (!(error instanceof DOMException && error.name === "AbortError")) {
        setErrorMessage(error instanceof Error ? error.message : "Something went wrong.");
      }
    } finally {
      if (predictionRequestRef.current === controller) {
        predictionRequestRef.current = null;
        setIsPredicting(false);
      }
    }
  }, []);

  const captureCameraFrame = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || cameraStatus !== "ready") {
      setErrorMessage("The camera is not ready yet. Please wait and try again.");
      return;
    }

    if (video.videoWidth === 0 || video.videoHeight === 0) {
      setErrorMessage("The camera has not produced a frame yet. Please try again.");
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) {
      setErrorMessage("VANTA could not capture this camera frame.");
      return;
    }

    const activeCameraRequestId = cameraRequestIdRef.current;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageBlob = await canvasToBlob(canvas);
    if (activeCameraRequestId !== cameraRequestIdRef.current) {
      return;
    }
    if (!imageBlob) {
      setErrorMessage("VANTA could not prepare this image. Please try again.");
      return;
    }

    const capturedFile = new File([imageBlob], `vanta-capture-${Date.now()}.jpg`, {
      type: "image/jpeg",
    });
    showPreview(capturedFile);
    stopCamera();
    await requestPrediction(capturedFile);
  }, [cameraStatus, requestPrediction, showPreview, stopCamera]);

  const handleFileSelection = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    clearResult();

    if (!file) {
      return;
    }
    if (!allowedImageTypes.has(file.type)) {
      clearPreview();
      setSelectedFile(null);
      setErrorMessage("Choose a JPEG, PNG, or WebP image.");
      event.target.value = "";
      return;
    }

    showPreview(file);
  };

  const switchInputMode = (mode: InputMode) => {
    if (mode === inputMode) {
      return;
    }
    stopCamera();
    clearResult();
    clearPreview();
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    setInputMode(mode);
  };

  const resetScanner = () => {
    stopCamera();
    clearResult();
    clearPreview();
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    if (inputMode === "camera") {
      void startCamera();
    }
  };

  useEffect(() => {
    return () => {
      cameraRequestIdRef.current += 1;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      predictionRequestRef.current?.abort();
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
      }
    };
  }, []);

  return (
    <main className="relative overflow-hidden px-4 py-10 sm:px-8 sm:py-14 lg:py-16">
      <div className="pointer-events-none absolute -left-32 top-24 h-72 w-72 rounded-full bg-lime-300/20 blur-3xl" />
      <div className="pointer-events-none absolute -right-36 bottom-20 h-80 w-80 rounded-full bg-emerald-300/20 blur-3xl" />

      <div className="relative mx-auto max-w-6xl">
        <div className="max-w-3xl">
          <p className="text-sm font-bold uppercase tracking-[0.2em] text-emerald-700">
            VANTA Scanner
          </p>
          <h1 className="mt-3 text-4xl font-black tracking-tight text-emerald-950 sm:text-5xl">
            Show us what you&apos;re sorting.
          </h1>
          <p className="mt-4 max-w-2xl text-base leading-7 text-emerald-950/65 sm:text-lg">
            Take one clear photo or choose an image. VANTA will identify the category and suggest
            a responsible next step.
          </p>
        </div>

        <div className="mt-9 grid items-start gap-6 lg:grid-cols-[minmax(0,1.25fr)_minmax(320px,0.75fr)]">
          <section className="overflow-hidden rounded-[1.75rem] border border-emerald-950/10 bg-white shadow-xl shadow-emerald-950/5">
            <div className="flex gap-1 border-b border-emerald-950/10 bg-emerald-50/70 p-2">
              <ModeButton
                active={inputMode === "camera"}
                label="Use camera"
                onClick={() => switchInputMode("camera")}
              />
              <ModeButton
                active={inputMode === "upload"}
                label="Upload image"
                onClick={() => switchInputMode("upload")}
              />
            </div>

            <div className="p-3 sm:p-5">
              <div className="relative aspect-[4/3] overflow-hidden rounded-[1.35rem] bg-emerald-950">
                {previewUrl ? (
                  <Image
                    src={previewUrl}
                    alt="Selected item preview"
                    fill
                    unoptimized
                    className="object-contain"
                  />
                ) : inputMode === "camera" ? (
                  <CameraPreview
                    cameraStatus={cameraStatus}
                    errorMessage={errorMessage}
                    videoRef={videoRef}
                    onStartCamera={() => void startCamera()}
                  />
                ) : (
                  <UploadPrompt fileInputRef={fileInputRef} />
                )}

                {isPredicting && (
                  <div
                    className="absolute inset-0 flex flex-col items-center justify-center bg-emerald-950/75 text-white backdrop-blur-sm"
                    role="status"
                  >
                    <span className="h-10 w-10 animate-spin rounded-full border-4 border-white/25 border-t-lime-300" />
                    <p className="mt-4 font-bold">Identifying your item…</p>
                  </div>
                )}
              </div>

              <canvas ref={canvasRef} className="hidden" aria-hidden="true" />
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={handleFileSelection}
                className="sr-only"
              />

              <div className="mt-4">
                {inputMode === "camera" && !previewUrl && cameraStatus === "ready" && (
                  <PrimaryButton disabled={isPredicting} onClick={() => void captureCameraFrame()}>
                    Scan item
                  </PrimaryButton>
                )}

                {inputMode === "upload" &&
                  previewUrl &&
                  selectedFile &&
                  !prediction &&
                  !errorMessage && (
                  <PrimaryButton
                    disabled={isPredicting}
                    onClick={() => void requestPrediction(selectedFile)}
                  >
                    Identify item
                  </PrimaryButton>
                )}

                {previewUrl && selectedFile && !isPredicting && errorMessage && (
                  <PrimaryButton onClick={() => void requestPrediction(selectedFile)}>
                    Try prediction again
                  </PrimaryButton>
                )}

                {previewUrl && !isPredicting && prediction && (
                  <PrimaryButton onClick={resetScanner}>Scan another item</PrimaryButton>
                )}

                {inputMode === "upload" && previewUrl && !isPredicting && !prediction && (
                  <button
                    type="button"
                    className="mt-3 w-full rounded-full px-5 py-2.5 text-sm font-bold text-emerald-800 transition hover:bg-emerald-50"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    Choose a different image
                  </button>
                )}
              </div>

              {errorMessage && (inputMode === "upload" || previewUrl) && (
                <ErrorNotice message={errorMessage} />
              )}
            </div>
          </section>

          <PredictionPanel prediction={prediction} isPredicting={isPredicting} />
        </div>

        <p className="mx-auto mt-7 max-w-3xl text-center text-xs leading-5 text-emerald-950/50">
          Disposal rules vary by location. Check local guidance when an item&apos;s material,
          condition, or recyclability is unclear.
        </p>
      </div>
    </main>
  );
}

function ModeButton({
  active,
  label,
  onClick,
}: {
  active: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onClick}
      className={`flex-1 rounded-full px-4 py-2.5 text-sm font-bold transition ${
        active
          ? "bg-emerald-800 text-white shadow-sm"
          : "text-emerald-950/60 hover:bg-white hover:text-emerald-900"
      }`}
    >
      {label}
    </button>
  );
}

function CameraPreview({
  cameraStatus,
  errorMessage,
  videoRef,
  onStartCamera,
}: {
  cameraStatus: CameraStatus;
  errorMessage: string | null;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  onStartCamera: () => void;
}) {
  return (
    <>
      <video
        ref={videoRef}
        muted
        playsInline
        className={`h-full w-full object-cover transition-opacity ${
          cameraStatus === "ready" ? "opacity-100" : "opacity-0"
        }`}
      />
      {cameraStatus === "ready" && (
        <div className="pointer-events-none absolute inset-[12%] rounded-3xl border-2 border-white/70 shadow-[0_0_0_999px_rgba(2,44,30,0.18)]" />
      )}
      {cameraStatus !== "ready" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center px-7 text-center text-white">
          <CameraIcon />
          {cameraStatus === "starting" ? (
            <p className="mt-5 font-bold" role="status">
              Starting camera…
            </p>
          ) : (
            <>
              <p className="mt-5 text-xl font-black">
                {cameraStatus === "error" ? "Camera unavailable" : "Ready to scan?"}
              </p>
              <p className="mt-2 max-w-sm text-sm leading-6 text-emerald-100/70">
                {cameraStatus === "error"
                  ? errorMessage
                  : "Allow camera access, then place one item clearly inside the frame."}
              </p>
              <button
                type="button"
                onClick={onStartCamera}
                className="mt-6 rounded-full bg-lime-300 px-6 py-3 text-sm font-black text-emerald-950 transition hover:bg-lime-200"
              >
                {cameraStatus === "error" ? "Try camera again" : "Enable camera"}
              </button>
            </>
          )}
        </div>
      )}
    </>
  );
}

function UploadPrompt({
  fileInputRef,
}: {
  fileInputRef: React.RefObject<HTMLInputElement | null>;
}) {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center px-7 text-center text-white">
      <UploadIcon />
      <p className="mt-5 text-xl font-black">Choose a clear image</p>
      <p className="mt-2 text-sm leading-6 text-emerald-100/70">
        JPEG, PNG, or WebP works best when one item fills most of the frame.
      </p>
      <button
        type="button"
        onClick={() => fileInputRef.current?.click()}
        className="mt-6 rounded-full bg-lime-300 px-6 py-3 text-sm font-black text-emerald-950 transition hover:bg-lime-200"
      >
        Select an image
      </button>
    </div>
  );
}

function PredictionPanel({
  prediction,
  isPredicting,
}: {
  prediction: PredictionResponse | null;
  isPredicting: boolean;
}) {
  if (!prediction) {
    return (
      <aside className="rounded-[1.75rem] border border-emerald-950/10 bg-white p-6 shadow-xl shadow-emerald-950/5 sm:p-8">
        <p className="text-xs font-bold uppercase tracking-[0.2em] text-emerald-700">
          {isPredicting ? "Analysing" : "Your result"}
        </p>
        <h2 className="mt-4 text-2xl font-black text-emerald-950">
          {isPredicting ? "VANTA is checking the image." : "Your guidance will appear here."}
        </h2>
        <p className="mt-3 leading-7 text-emerald-950/60">
          {isPredicting
            ? "This usually takes only a moment."
            : "Use an evenly lit image with a single item and a simple background for a clearer result."}
        </p>
        <div className="mt-8 space-y-3" aria-hidden="true">
          {[82, 64, 46, 30, 20].map((width) => (
            <div
              key={width}
              className="h-3 rounded-full bg-emerald-950/[0.06]"
              style={{ width: `${width}%` }}
            />
          ))}
        </div>
      </aside>
    );
  }

  const details = categoryDetails[prediction.label];
  return (
    <aside
      className="rounded-[1.75rem] border border-emerald-950/10 bg-white p-6 shadow-xl shadow-emerald-950/5 sm:p-8"
      aria-live="polite"
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.2em] text-emerald-700">
            Predicted category
          </p>
          <h2 className="mt-3 text-3xl font-black text-emerald-950">{details.name}</h2>
        </div>
        <div className="rounded-2xl bg-emerald-50 px-3 py-2 text-right">
          <p className="text-lg font-black text-emerald-900">
            {formatPercentage(prediction.confidence)}
          </p>
          <p className="text-[0.65rem] font-bold uppercase tracking-wide text-emerald-800/60">
            confidence
          </p>
        </div>
      </div>

      {prediction.is_uncertain && (
        <div className="mt-5 rounded-2xl border border-amber-300 bg-amber-50 p-4 text-sm leading-6 text-amber-950">
          <strong>Not fully confident.</strong> Try scanning again with better lighting or a clearer
          view.
        </div>
      )}

      <div className="mt-6 rounded-2xl bg-[#f5f7f2] p-5">
        <p className="text-xs font-bold uppercase tracking-[0.16em] text-emerald-700">
          Disposal guidance
        </p>
        <p className="mt-2 leading-7 text-emerald-950/75">{details.guidance}</p>
      </div>

      <div className="mt-7">
        <p className="text-sm font-black text-emerald-950">All category scores</p>
        <div className="mt-4 space-y-3">
          {wasteCategories.map((category) => {
            const categoryScore = prediction.scores[category];
            const categoryDetail = categoryDetails[category];
            return (
              <div key={category}>
                <div className="mb-1.5 flex justify-between gap-4 text-sm">
                  <span className="font-semibold text-emerald-950/70">
                    {categoryDetail.name}
                  </span>
                  <span className="tabular-nums text-emerald-950/55">
                    {formatPercentage(categoryScore)}
                  </span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-emerald-950/[0.07]">
                  <div
                    className={`h-full rounded-full ${categoryDetail.color}`}
                    style={{ width: `${Math.max(categoryScore * 100, 0.5)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </aside>
  );
}

function PrimaryButton({
  children,
  disabled = false,
  onClick,
}: {
  children: React.ReactNode;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className="w-full rounded-full bg-emerald-800 px-6 py-3.5 font-black text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-60"
    >
      {children}
    </button>
  );
}

function ErrorNotice({ message }: { message: string }) {
  return (
    <div
      className="mt-4 rounded-2xl border border-red-200 bg-red-50 p-4 text-sm leading-6 text-red-900"
      role="alert"
    >
      {message}
    </div>
  );
}

function CameraIcon() {
  return (
    <svg
      className="h-12 w-12 text-lime-300"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M8.5 6 10 4h4l1.5 2H19a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h3.5Z"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinejoin="round"
      />
      <circle cx="12" cy="12.5" r="3.5" stroke="currentColor" strokeWidth="1.7" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg
      className="h-12 w-12 text-lime-300"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

async function requestCameraStream(): Promise<MediaStream> {
  try {
    return await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: "environment" } },
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "OverconstrainedError") {
      return navigator.mediaDevices.getUserMedia({ audio: false, video: true });
    }
    throw error;
  }
}

function getCameraErrorMessage(error: unknown): string {
  if (error instanceof DOMException) {
    if (error.name === "NotAllowedError" || error.name === "SecurityError") {
      return "Camera permission was denied. Allow access in your browser settings or upload an image instead.";
    }
    if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
      return "No camera was found on this device. Upload an image instead.";
    }
    if (error.name === "NotReadableError" || error.name === "TrackStartError") {
      return "The camera is already in use or could not be started. Close other camera apps and try again.";
    }
  }
  return "VANTA could not start the camera. Try again or upload an image instead.";
}

function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob | null> {
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
}

function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}
