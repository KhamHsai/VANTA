export default function DashboardPage() {
  return (
    <main className="mx-auto min-h-[70vh] max-w-6xl px-5 py-20 sm:px-8">
      <p className="text-sm font-bold uppercase tracking-[0.2em] text-emerald-700">Dashboard</p>
      <h1 className="mt-4 text-4xl font-black text-emerald-950">Your impact, in one place.</h1>
      <p className="mt-4 max-w-2xl leading-7 text-emerald-950/60">
        Activity and impact reporting will appear here in a future phase.
      </p>
      <div className="mt-10 grid gap-5 sm:grid-cols-3">
        {["Items sorted", "Correct disposal", "Waste diverted"].map((label) => (
          <div key={label} className="rounded-3xl border border-emerald-950/10 bg-white p-7">
            <p className="text-sm font-semibold text-emerald-950/55">{label}</p>
            <p className="mt-6 text-4xl font-black text-emerald-950">—</p>
          </div>
        ))}
      </div>
    </main>
  );
}

