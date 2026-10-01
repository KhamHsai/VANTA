import { getApiBaseUrl, wasteCategories } from "@/lib/prediction";
import type { PredictionSource, WasteCategory } from "@/lib/prediction";

export interface RecentPrediction {
  id: number;
  predicted_label: WasteCategory;
  confidence: number;
  is_uncertain: boolean;
  source_type: PredictionSource;
  created_at: string;
}

export interface DashboardSummary {
  total_scans: number;
  average_confidence: number;
  category_counts: Record<WasteCategory, number>;
  uncertain_scans: number;
  recent_predictions: RecentPrediction[];
}

export async function fetchDashboardSummary(signal?: AbortSignal): Promise<DashboardSummary> {
  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}/api/v1/dashboard/summary`, {
      signal,
      cache: "no-store",
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw new Error(
      "VANTA could not reach the dashboard service. Check that the backend is running and try again.",
    );
  }

  if (!response.ok) {
    throw new Error("Dashboard data is temporarily unavailable. Please try again.");
  }

  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new Error("The dashboard service returned an unreadable response.");
  }
  return parseDashboardSummary(payload);
}

export function parseDashboardSummary(payload: unknown): DashboardSummary {
  if (!isRecord(payload)) {
    throw new Error("The dashboard service returned an invalid response.");
  }

  const {
    total_scans: totalScans,
    average_confidence: averageConfidence,
    category_counts: categoryCounts,
    uncertain_scans: uncertainScans,
    recent_predictions: recentPredictions,
  } = payload;

  if (
    !isNonNegativeInteger(totalScans) ||
    !isProbability(averageConfidence) ||
    !isNonNegativeInteger(uncertainScans) ||
    !isRecord(categoryCounts) ||
    !Array.isArray(recentPredictions)
  ) {
    throw new Error("The dashboard service returned an invalid response.");
  }

  const validatedCounts = {} as Record<WasteCategory, number>;
  for (const category of wasteCategories) {
    const count = categoryCounts[category];
    if (!isNonNegativeInteger(count)) {
      throw new Error("The dashboard service returned invalid category totals.");
    }
    validatedCounts[category] = count;
  }

  return {
    total_scans: totalScans,
    average_confidence: averageConfidence,
    category_counts: validatedCounts,
    uncertain_scans: uncertainScans,
    recent_predictions: recentPredictions.map(parseRecentPrediction),
  };
}

function parseRecentPrediction(value: unknown): RecentPrediction {
  if (!isRecord(value)) {
    throw new Error("The dashboard service returned invalid recent scan data.");
  }

  const {
    id,
    predicted_label: predictedLabel,
    confidence,
    is_uncertain: isUncertain,
    source_type: sourceType,
    created_at: createdAt,
  } = value;
  if (
    !Number.isInteger(id) ||
    typeof id !== "number" ||
    !isWasteCategory(predictedLabel) ||
    !isProbability(confidence) ||
    typeof isUncertain !== "boolean" ||
    (sourceType !== "camera" && sourceType !== "upload") ||
    typeof createdAt !== "string" ||
    Number.isNaN(Date.parse(createdAt))
  ) {
    throw new Error("The dashboard service returned invalid recent scan data.");
  }

  return {
    id,
    predicted_label: predictedLabel,
    confidence,
    is_uncertain: isUncertain,
    source_type: sourceType,
    created_at: createdAt,
  };
}

function isWasteCategory(value: unknown): value is WasteCategory {
  return typeof value === "string" && wasteCategories.some((category) => category === value);
}

function isProbability(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
