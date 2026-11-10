import Link from "next/link";

export default function DashboardPage() {
  return (
    <main className="mx-auto min-h-[75vh] max-w-[90rem] px-6 py-12 sm:py-16 lg:px-10">
      <div className="flex flex-wrap items-end justify-between gap-6">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-lime-300">Your workspace</p>
          <h1 className="mt-3 text-4xl font-semibold tracking-tight text-white">Match dashboard</h1>
          <p className="mt-3 max-w-xl text-slate-400">Your video reviews and movement insights will live here.</p>
        </div>
        <Link href="/upload" className="rounded-xl bg-lime-300 px-4 py-3 text-sm font-semibold text-slate-950 transition hover:bg-lime-200 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-lime-300">+ Add a video</Link>
      </div>

      <section className="mt-10 overflow-hidden rounded-3xl border border-white/10 bg-[#101b1b]">
        <div className="grid lg:grid-cols-[1fr_0.8fr]">
          <div className="p-8 sm:p-12">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-lime-300/20 bg-lime-300/10 text-xl text-lime-300" aria-hidden="true">↗</div>
            <p className="mt-8 text-xs font-semibold uppercase tracking-[0.2em] text-lime-300">A fresh court</p>
            <h2 className="mt-3 text-2xl font-semibold text-white">No match reviews yet</h2>
            <p className="mt-3 max-w-md leading-7 text-slate-400">Upload a rally video to begin building your movement library. You’ll be able to review videos and analysis from this space.</p>
            <Link href="/upload" className="mt-7 inline-flex items-center gap-2 font-semibold text-lime-300 hover:text-lime-200">Upload your first video <span aria-hidden="true">→</span></Link>
          </div>
          <div className="relative hidden min-h-72 overflow-hidden border-l border-white/[0.06] bg-[#0c1716] p-8 lg:block">
            <div className="absolute inset-8 rounded-2xl border border-white/20" />
            <div className="absolute inset-x-8 top-1/2 border-t border-white/20" />
            <div className="absolute left-[22%] right-[22%] top-[27%] bottom-[27%] border border-white/15" />
            <div className="absolute top-1/2 bottom-[27%] left-1/2 border-l border-white/15" />
            <div className="absolute left-[36%] top-[66%] h-3 w-3 rounded-full bg-lime-300/60 shadow-[0_0_20px_rgba(190,242,100,0.35)]" />
            <div className="absolute left-[59%] top-[39%] h-2.5 w-2.5 rounded-full bg-white/50" />
            <p className="absolute bottom-3 right-8 font-mono text-[10px] tracking-[0.2em] text-slate-600">WAITING FOR YOUR FIRST RALLY</p>
          </div>
        </div>
      </section>
      <p className="mt-5 text-xs text-slate-500">Analysis features are being developed. Uploaded videos are not yet listed as completed match reviews.</p>
    </main>
  );
}
