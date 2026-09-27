export default function ScanPage() {
  return (
    <main className="mx-auto flex min-h-[70vh] max-w-4xl items-center px-5 py-20 sm:px-8">
      <section className="w-full rounded-[2rem] border border-emerald-950/10 bg-white p-8 text-center shadow-sm sm:p-14">
        <p className="text-sm font-bold uppercase tracking-[0.2em] text-emerald-700">Scanner</p>
        <h1 className="mt-4 text-4xl font-black text-emerald-950">Scan experience coming next.</h1>
        <p className="mx-auto mt-5 max-w-xl leading-7 text-emerald-950/60">
          Camera capture and real-time classification will be added in a later phase. This page reserves the product flow without requesting device access.
        </p>
      </section>
    </main>
  );
}

