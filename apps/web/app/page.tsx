import Link from "next/link";
import RallyTelemetry from "./components/rally-telemetry";

const capabilities = [
  { number: "01", icon: "↥", title: "Raw Video Ingestion", label: "THE INPUT", text: "Bring your own footage. Preview your rally, choose your moment, and start with the video you already have.", detail: "MP4 · MOV · WEBM · Up to 500 MB", color: "lime" },
  { number: "02", icon: "⌖", title: "Spatial Tracking", label: "COMPUTER VISION", text: "See the court as coordinates. Explore player position, movement vectors, and the space covered between shots.", detail: "Player tracking · Court mapping", color: "cyan" },
  { number: "03", icon: "↗", title: "Tactical Insight Engine", label: "THE PERSPECTIVE", text: "Connect movement to moments. Review recovery patterns and build a clearer picture of how each rally unfolds.", detail: "Movement phases · Evidence-led review", color: "lime" },
];

export default function Home() {
  return (
    <main className="landing-shell">
      <section className="hero-grid">
        <div className="hero-copy">
          <span className="engine-badge"><i /> Match Analysis Engine <span>v2.4 / CONCEPT</span></span>
          <p className="hero-kicker">THE NEXT DIMENSION OF YOUR GAME</p>
          <h1>Performance.<br />Decoded.<br /><span>Point by point.</span></h1>
          <p className="hero-description">Beyond the scoreline lies a different game. Explore the movement, timing, and decisions that shape every rally with CourtVision AI.</p>
          <div className="hero-actions"><Link className="primary-action" href="/upload">Upload a match <span aria-hidden="true">↗</span></Link><a className="secondary-action" href="#telemetry"><span aria-hidden="true">▷</span> Explore telemetry</a></div>
          <div className="micro-stats"><span><b>2D</b> Court mapping</span><i /><span><b>Frame-level</b> Movement review</span></div>
          <p className="preview-note">Interactive product preview · Analysis integration in development</p>
        </div>
        <RallyTelemetry />
      </section>
      <section className="performance-strip" aria-label="Sample performance metrics">
        <div className="strip-label"><span className="eyebrow">PERFORMANCE PULSE</span><span className="muted">Illustrative rally data</span></div>
        <div className="strip-stat"><span>Rally duration</span><strong>NUMBER<small>s</small></strong><svg viewBox="0 0 100 30" aria-hidden="true"><path d="M0 24L12 21L22 26L35 12L46 17L58 7L70 14L84 3L100 6"/></svg></div>
        <div className="strip-stat"><span>Speed differential</span><strong>+2.4<small>km/h</small></strong><span className="stat-detail cyan">P1 vs. P2</span></div>
        <div className="strip-stat"><span>In–out accuracy</span><strong>98.6<small>%</small></strong><span className="stat-detail lime">Demo value</span></div>
      </section>
      <section id="features" className="capabilities-section">
        <div className="section-heading"><div><span className="eyebrow lime">FROM RAW FOOTAGE TO REVEALING DETAIL</span><h2>One rally.<br className="mobile-break" /> A whole new perspective.</h2></div><p>A connected workflow designed to make complex movement easier to see, understand, and review.</p></div>
        <div className="capabilities-grid">{capabilities.map((item) => <article className="capability-card" key={item.number}><div className="capability-top"><span className={`capability-icon ${item.color}`} aria-hidden="true">{item.icon}</span><span className="card-number">/{item.number}</span></div><span className="eyebrow muted">{item.label}</span><h3>{item.title}</h3><p>{item.text}</p><div className="capability-detail"><span className={item.color}>↗</span>{item.detail}</div></article>)}</div>
      </section>
      <section id="pricing" className="access-section"><div><span className="eyebrow cyan">BUILT FOR THE NEXT SESSION</span><h2>Your game has more to tell you.</h2><p>Explore the local prototype today. Paid plans and production access have not been announced.</p></div><Link className="primary-action" href="/upload">Start with your footage <span aria-hidden="true">↗</span></Link></section>
    </main>
  );
}

