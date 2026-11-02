import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "CourtVision AI",
  description: "Explore tennis footage and player movement.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-950 text-slate-100 antialiased">
        <header className="border-b border-slate-800">
          <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-6 py-4">
            <Link
              href="/"
              className="text-lg font-bold tracking-tight text-white focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-lime-400"
            >
              CourtVision <span className="text-lime-400">AI</span>
            </Link>

            <nav aria-label="Main navigation" className="flex gap-5 text-sm">
              <Link
                href="/"
                className="text-slate-300 hover:text-white focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-lime-400"
              >
                Home
              </Link>
              <Link
                href="/dashboard"
                className="text-slate-300 hover:text-white focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-lime-400"
              >
                Dashboard
              </Link>
              <Link
                href="/upload"
                className="text-slate-300 hover:text-white focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-lime-400"
              >
                Upload
              </Link>
            </nav>
          </div>
        </header>

        {children}

        <footer className="border-t border-slate-800 px-6 py-6 text-center text-sm text-slate-400">
          CourtVision AI · Tennis analysis in development
        </footer>
      </body>
    </html>
  );
}