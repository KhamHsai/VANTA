import Link from "next/link";

const steps = [
  ["01", "Point", "Show VANTA an item you want to dispose of."],
  ["02", "Identify", "Receive a clear material classification in real time."],
  ["03", "Dispose", "Follow local, practical guidance for the right bin."],
];

export default function Home() {
  return (
    <main>
      <section className="relative overflow-hidden px-5 py-20 sm:px-8 sm:py-28">
        <div className="absolute -right-32 top-10 h-80 w-80 rounded-full bg-lime-300/30 blur-3xl" />
        <div className="relative mx-auto grid max-w-6xl items-center gap-14 lg:grid-cols-[1.2fr_0.8fr]">
          <div>
            <p className="mb-5 text-sm font-bold uppercase tracking-[0.2em] text-emerald-700">Waste, clarified</p>
            <h1 className="max-w-3xl text-5xl font-black leading-[1.02] tracking-tight text-emerald-950 sm:text-7xl">
              Know where it goes, before you let it go.
            </h1>
            <p className="mt-7 max-w-2xl text-lg leading-8 text-emerald-950/65">
              VANTA turns a quick scan into confident disposal guidance, helping people sort everyday waste with less guesswork.
            </p>
            <div className="mt-9 flex flex-wrap gap-4">
              <Link href="/scan" className="rounded-full bg-emerald-800 px-7 py-3.5 font-bold text-white transition hover:bg-emerald-700">
                Start a scan
              </Link>
              <a href="#how-it-works" className="rounded-full border border-emerald-900/20 px-7 py-3.5 font-bold text-emerald-950 transition hover:bg-white">
                How it works
              </a>
            </div>
          </div>
          <div className="rounded-[2rem] border border-white/60 bg-emerald-900 p-8 text-white shadow-2xl shadow-emerald-950/15 sm:p-10">
            <div className="mb-16 flex items-center justify-between">
              <span className="rounded-full bg-white/10 px-3 py-1 text-xs font-bold uppercase tracking-widest">Live guidance</span>
              <span className="h-3 w-3 rounded-full bg-lime-300" />
            </div>
            <div className="rounded-3xl bg-white/10 p-6">
              <div className="mb-6 h-36 rounded-2xl border border-dashed border-white/30 bg-emerald-950/25" />
              <p className="text-sm text-emerald-100/70">Designed for clear answers</p>
              <p className="mt-2 text-2xl font-bold">Scan. Sort. Move on.</p>
            </div>
          </div>
        </div>
      </section>

      <section id="how-it-works" className="bg-white px-5 py-20 sm:px-8">
        <div className="mx-auto max-w-6xl">
          <p className="text-sm font-bold uppercase tracking-[0.2em] text-emerald-700">How it works</p>
          <h2 className="mt-3 text-3xl font-black text-emerald-950 sm:text-4xl">A simpler path to the right bin.</h2>
          <div className="mt-12 grid gap-5 md:grid-cols-3">
            {steps.map(([number, title, description]) => (
              <article key={number} className="rounded-3xl border border-emerald-950/10 bg-[#f5f7f2] p-7">
                <span className="text-sm font-black text-emerald-700">{number}</span>
                <h3 className="mt-8 text-2xl font-bold text-emerald-950">{title}</h3>
                <p className="mt-3 leading-7 text-emerald-950/60">{description}</p>
              </article>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}

