export const wasteCategories = ["general", "metal", "organic", "paper", "plastic"] as const;

export type WasteCategory = (typeof wasteCategories)[number];
export type PredictionSource = "camera" | "upload";

export interface PredictionResponse {
  label: WasteCategory;
  confidence: number;
  is_uncertain: boolean;
  scores: Record<WasteCategory, number>;
}

const defaultApiBaseUrl = "http://127.0.0.1:8000";

export async function predictWasteImage(
  imageFile: File,
  sourceType: PredictionSource,
  signal?: AbortSignal,
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", imageFile);
  formData.append("source_type", sourceType);

  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}/api/v1/predict`, {
      method: "POST",
      body: formData,
      signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw new Error(
      "VANTA could not reach the prediction service. Check that the backend is running and try again.",
    );
  }

  if (!response.ok) {
    throw new Error(await readApiError(response));
  }

  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new Error("The prediction service returned an unreadable response. Please try again.");
  }

  return parsePredictionResponse(payload);
}

export function parsePredictionResponse(payload: unknown): PredictionResponse {
  if (!isRecord(payload)) {
    throw new Error("The prediction service returned an invalid response.");
  }

  const { label, confidence, is_uncertain: isUncertain, scores } = payload;
  if (
    !isWasteCategory(label) ||
    !isProbability(confidence) ||
    typeof isUncertain !== "boolean" ||
    !isRecord(scores)
  ) {
    throw new Error("The prediction service returned an invalid response.");
  }

  const validatedScores = {} as Record<WasteCategory, number>;
  for (const category of wasteCategories) {
    const score = scores[category];
    if (!isProbability(score)) {
      throw new Error("The prediction service returned incomplete category scores.");
    }
    validatedScores[category] = score;
  }

  return {
    label,
    confidence,
    is_uncertain: isUncertain,
    scores: validatedScores,
  };
}

export function getApiBaseUrl(): string {
  return (process.env.NEXT_PUBLIC_API_BASE_URL || defaultApiBaseUrl).replace(/\/+$/, "");
}

async function readApiError(response: Response): Promise<string> {
  try {
    const payload: unknown = await response.json();
    if (isRecord(payload) && typeof payload.detail === "string") {
      return payload.detail;
    }
  } catch {
    // A status-based message is safer when the backend does not return JSON.
  }

  if (response.status === 413) {
    return "This image is too large. Choose a smaller image and try again.";
  }
  if (response.status === 415) {
    return "This file type is not supported. Choose a JPEG, PNG, or WebP image.";
  }
  if (response.status === 422) {
    return "VANTA could not read this image. Choose a different image and try again.";
  }
  return "The prediction service could not process this image. Please try again.";
}

function isWasteCategory(value: unknown): value is WasteCategory {
  return typeof value === "string" && wasteCategories.some((category) => category === value);
}

function isProbability(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
