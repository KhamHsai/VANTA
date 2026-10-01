"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { fetchDashboardSummary } from "@/lib/dashboard";
import type { DashboardSummary, RecentPrediction } from "@/lib/dashboard";
import { wasteCategories } from "@/lib/prediction";
import type { WasteCategory } from "@/lib/prediction";


const categoryDetails: Record<WasteCategory, { name: string; color: string }> = {
  general: { name: "General Waste", color: "bg-stone-500" },
  metal: { name: "Metal", color: "bg-slate-500" },
  organic: { name: "Organic", color: "bg-lime-600" },
  paper: { name: "Paper / Cardboard", color: "bg-amber-600" },
  plastic: { name: "Plastic", color: "bg-sky-600" },
};

export default function DashboardInterface() {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const activeRequestRef = useRef<AbortController | null>(null);

  const loadDashboard = useCallback(async () => {
    activeRequestRef.current?.abort();
    const controller = new AbortController();
    activeRequestRef.current = controller;
    setIsLoading(true);
    setErrorMessage(null);

    try {
      const dashboardSummary = await fetchDashboardSummary(controller.signal);
      if (!controller.signal.aborted) {
        setSummary(dashboardSummary);
      }
    } catch (error) {
      if (!(error instanceof DOMException && error.name === "AbortError")) {
        setErrorMessage(error instanceof Error ? error.message : "Unable to load the dashboard.");
      }
    } finally {
      if (activeRequestRef.current === controller) {
        activeRequestRef.current = null;
        setIsLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    activeRequestRef.current = controller;
    void fetchDashboardSummary(controller.signal)
      .then((dashboardSummary) => {
        if (!controller.signal.aborted) {
          setSummary(dashboardSummary);
        }
      })
      .catch((error: unknown) => {
        if (!(error instanceof DOMException && error.name === "AbortError")) {
          setErrorMessage(
            error instanceof Error ? error.message : "Unable to load the dashboard.",
          );
        }
      })
      .finally(() => {
        if (activeRequestRef.current === controller) {
          activeRequestRef.current = null;
          setIsLoading(false);
        }
      });

    return () => controller.abort();
  }, []);

  return (
    <main className="relative min-h-[70vh] overflow-hidden px-4 py-10 sm:px-8 sm:py-14 lg:py-16">
      <div className="pointer-events-none absolute -right-28 top-12 h-72 w-72 rounded-full bg-lime-300/20 blur-3xl" />
      <div className="relative mx-auto max-w-6xl">
        <div className="flex flex-col justify-between gap-6 sm:flex-row sm:items-end">
          <div>
            <p className="text-sm font-bold uppercase tracking-[0.2em] text-emerald-700">
              Dashboard
            </p>
            <h1 className="mt-3 text-4xl font-black tracking-tight text-emerald-950 sm:text-5xl">
              Your sorting activity.
            </h1>
            <p className="mt-4 max-w-2xl leading-7 text-emerald-950/60">
              A clear view of VANTA scans, confidence, and the waste categories you encounter.
            </p>
          </div>
          <Link
            href="/scan"
            className="inline-flex w-fit rounded-full bg-emerald-800 px-6 py-3 font-bold text-white transition hover:bg-emerald-700"
          >
            Scan an item
          </Link>
        </div>

        {isLoading && !summary ? (
          <DashboardLoading />
        ) : errorMessage && !summary ? (
          <DashboardError message={errorMessage} onRetry={() => void loadDashboard()} />
        ) : summary?.total_scans === 0 ? (
          <DashboardEmpty />
        ) : summary ? (
          <DashboardContent summary={summary} />
        ) : null}

        {errorMessage && summary && (
          <div className="mt-6 rounded-2xl border border-amber-300 bg-amber-50 p-4 text-sm text-amber-950" role="alert">
            {errorMessage} Showing the most recently loaded data.
          </div>
        )}
      </div>
    </main>
  );
}

function DashboardContent({ summary }: { summary: DashboardSummary }) {
  return (
    <div className="mt-10 space-y-6">
      <section className="grid gap-4 sm:grid-cols-3" aria-label="Scan summary">
        <MetricCard label="Total scans" value={summary.total_scans.toLocaleString()} />
        <MetricCard label="Average confidence" value={formatPercentage(summary.average_confidence)} />
        <MetricCard label="Uncertain scans" value={summary.uncertain_scans.toLocaleString()} />
      </section>

      <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
        <CategoryDistribution summary={summary} />
        <RecentScans predictions={summary.recent_predictions} />
      </div>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <article className="rounded-3xl border border-emerald-950/10 bg-white p-6 shadow-sm sm:p-7">
      <p className="text-sm font-semibold text-emerald-950/55">{label}</p>
      <p className="mt-5 text-4xl font-black tabular-nums text-emerald-950">{value}</p>
    </article>
  );
}

function CategoryDistribution({ summary }: { summary: DashboardSummary }) {
  return (
    <section className="rounded-3xl border border-emerald-950/10 bg-white p-6 shadow-sm sm:p-7">
      <p className="text-xs font-bold uppercase tracking-[0.18em] text-emerald-700">
        Category distribution
      </p>
      <h2 className="mt-3 text-2xl font-black text-emerald-950">What you&apos;ve scanned</h2>
      <div className="mt-7 space-y-5">
        {wasteCategories.map((category) => {
          const count = summary.category_counts[category];
          const percentage = count / summary.total_scans;
          return (
            <div key={category}>
              <div className="mb-2 flex items-center justify-between gap-4 text-sm">
                <span className="font-semibold text-emerald-950/70">
                  {categoryDetails[category].name}
                </span>
                <span className="tabular-nums text-emerald-950/55">
                  {count} · {formatPercentage(percentage)}
                </span>
              </div>
              <div className="h-2.5 overflow-hidden rounded-full bg-emerald-950/[0.07]">
                <div
                  className={`h-full rounded-full ${categoryDetails[category].color}`}
                  style={{ width: `${percentage * 100}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function RecentScans({ predictions }: { predictions: RecentPrediction[] }) {
  return (
    <section className="overflow-hidden rounded-3xl border border-emerald-950/10 bg-white shadow-sm">
      <div className="border-b border-emerald-950/10 p-6 sm:p-7">
        <p className="text-xs font-bold uppercase tracking-[0.18em] text-emerald-700">
          Recent scans
        </p>
        <h2 className="mt-3 text-2xl font-black text-emerald-950">Latest classifications</h2>
      </div>
      <div className="divide-y divide-emerald-950/10">
        {predictions.map((prediction) => (
          <RecentScanRow key={prediction.id} prediction={prediction} />
        ))}
      </div>
    </section>
  );
}

function RecentScanRow({ prediction }: { prediction: RecentPrediction }) {
  return (
    <article className="flex flex-col gap-3 p-5 sm:flex-row sm:items-center sm:justify-between sm:px-7">
      <div className="flex items-center gap-3">
        <span
          className={`h-3 w-3 shrink-0 rounded-full ${categoryDetails[prediction.predicted_label].color}`}
        />
        <div>
          <p className="font-bold text-emerald-950">
            {categoryDetails[prediction.predicted_label].name}
          </p>
          <p className="mt-1 text-xs text-emerald-950/50">
            {formatDate(prediction.created_at)} · {capitalize(prediction.source_type)}
          </p>
        </div>
      </div>
      <div className="flex items-center gap-3 pl-6 sm:pl-0">
        {prediction.is_uncertain && (
          <span className="rounded-full bg-amber-100 px-2.5 py-1 text-xs font-bold text-amber-900">
            Uncertain
          </span>
        )}
        <span className="font-black tabular-nums text-emerald-900">
          {formatPercentage(prediction.confidence)}
        </span>
      </div>
    </article>
  );
}

function DashboardEmpty() {
  return (
    <section className="mt-10 rounded-[2rem] border border-emerald-950/10 bg-white px-6 py-16 text-center shadow-sm sm:px-12">
      <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-emerald-100 text-2xl" aria-hidden="true">
        ◌
      </div>
      <h2 className="mt-6 text-2xl font-black text-emerald-950">No scans yet</h2>
      <p className="mx-auto mt-3 max-w-md leading-7 text-emerald-950/60">
        Your scan totals and category trends will appear here after your first successful prediction.
      </p>
      <Link
        href="/scan"
        className="mt-7 inline-flex rounded-full bg-emerald-800 px-6 py-3 font-bold text-white transition hover:bg-emerald-700"
      >
        Make the first scan
      </Link>
    </section>
  );
}

function DashboardLoading() {
  return (
    <div className="mt-10 grid animate-pulse gap-4 sm:grid-cols-3" role="status" aria-label="Loading dashboard">
      {[1, 2, 3].map((item) => (
        <div key={item} className="h-36 rounded-3xl bg-emerald-950/[0.07]" />
      ))}
    </div>
  );
}

function DashboardError({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <section className="mt-10 rounded-3xl border border-red-200 bg-red-50 p-7 text-red-950" role="alert">
      <h2 className="text-xl font-black">Dashboard unavailable</h2>
      <p className="mt-2 leading-7">{message}</p>
      <button
        type="button"
        onClick={onRetry}
        className="mt-5 rounded-full bg-red-900 px-5 py-2.5 text-sm font-bold text-white transition hover:bg-red-800"
      >
        Try again
      </button>
    </section>
  );
}

function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
