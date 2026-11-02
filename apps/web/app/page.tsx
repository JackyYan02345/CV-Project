export default function Home() {
  return (
    <main className="min-h-screen bg-slate-950 px-6 py-16 text-white">
      <div className="mx-auto max-w-4xl">
        <p className="text-sm font-semibold uppercase tracking-widest text-lime-400">
          CourtVision AI
        </p>

        <h1 className="mt-5 text-4xl font-bold tracking-tight sm:text-6xl">
          Understand your game, one rally at a time.
        </h1>

        <p className="mt-6 max-w-2xl text-lg leading-8 text-slate-300">
          We are building tools to review tennis footage and explore
          player movement.
        </p>

        <p className="mt-10 text-sm text-slate-400">
          Video analysis features are in development.
        </p>
      </div>
    </main>
  );
}