import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "CourtVision AI — Performance, decoded.",
  description: "Explore tennis movement, timing, and rally telemetry with CourtVision AI.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <a className="skip-link" href="#main-content">Skip to content</a>
        <header className="floating-header">
          <Link href="/" className="brand" aria-label="CourtVision AI home"><svg viewBox="0 0 32 32" aria-hidden="true"><rect x="3" y="3" width="26" height="26" rx="8" fill="#c5fb65"/><path d="M10 9H22V23H10ZM10 16H22M16 9V23" fill="none" stroke="#141a10" strokeWidth="1.5"/></svg><span>CourtVision<span className="brand-ai">AI</span></span></Link>
          <nav aria-label="Main navigation"><Link href="/#features">Features</Link><Link href="/dashboard">Dashboard</Link><Link href="/#telemetry">Telemetry</Link><Link href="/#pricing">Pricing</Link></nav>
          <Link href="/upload" className="header-action">Upload video <span aria-hidden="true">↗</span></Link>
        </header>
        <div id="main-content">{children}</div>
        <footer className="micro-footer"><span>© 2026 CourtVision AI <span className="footer-dot">·</span> Performance, decoded.</span><Link href="/privacy">Privacy & data</Link><span className="footer-preview"><i /> Product preview</span></footer>
      </body>
    </html>
  );
}

