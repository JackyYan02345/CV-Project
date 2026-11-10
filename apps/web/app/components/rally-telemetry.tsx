"use client";

import { useEffect, useState } from "react";

// Illustrative frames for the UI prototype, not model measurements.
const frames = [
  { x: 205, y: 310, bx: 300, by: 110, recovery: "0.82", split: "+42", coverage: "3.4" },
  { x: 260, y: 300, bx: 180, by: 205, recovery: "0.76", split: "+35", coverage: "4.1" },
  { x: 325, y: 315, bx: 330, by: 300, recovery: "0.71", split: "+28", coverage: "5.2" },
  { x: 275, y: 300, bx: 210, by: 95, recovery: "0.68", split: "+24", coverage: "5.8" },
  { x: 210, y: 315, bx: 160, by: 220, recovery: "0.74", split: "+31", coverage: "6.4" },
  { x: 155, y: 300, bx: 150, by: 305, recovery: "0.79", split: "+38", coverage: "7.2" },
];

export default function RallyTelemetry() {
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [trails, setTrails] = useState(true);
  const frame = frames[frameIndex];

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => setFrameIndex((index) => (index + 1) % frames.length), 1100);
    return () => window.clearInterval(timer);
  }, [playing]);

  return (
    <section id="telemetry" className="telemetry-card" aria-label="Interactive sample rally telemetry">
      <div className="hud-heading">
        <div><span className="eyebrow cyan">COURTVISION / TELEMETRY</span><h2>Every movement. In focus.</h2></div>
        <span className="demo-tag"><i /> Sample replay</span>
      </div>
      <div className="hud-toolbar"><span>RALLY 014 <span className="muted">/ SINGLES</span></span><button aria-pressed={trails} onClick={() => setTrails(!trails)}>Trajectories <span className={trails ? "lime" : "muted"}>{trails ? "ON" : "OFF"}</span></button></div>
      <div className="court-stage">
        <span className="court-coordinate">X / Y SPATIAL MAP</span>
        <svg viewBox="0 0 500 380" role="img" aria-label={`Sample player position at replay frame ${frameIndex + 1}`}>
          <defs><pattern id="hud-grid" width="25" height="25" patternUnits="userSpaceOnUse"><path d="M25 0H0V25" fill="none" stroke="#23303c" strokeWidth=".5" /></pattern><radialGradient id="player-glow"><stop stopColor="#c5fb65" stopOpacity=".28"/><stop offset="1" stopColor="#c5fb65" stopOpacity="0"/></radialGradient></defs>
          <rect width="500" height="380" fill="url(#hud-grid)" />
          <rect x="108" y="45" width="284" height="290" fill="#16302e" fillOpacity=".35" stroke="#778f98" />
          <path d="M139 45V335M361 45V335M108 190H392M139 111H361M139 269H361M250 111V269" fill="none" stroke="#778f98" strokeWidth="1.2" />
          <path d="M98 190H402" stroke="#d7e2e7" strokeWidth="2" strokeDasharray="3 3" />
          <text x="60" y="194" fill="#7a909e" fontSize="9">NET</text>
          {trails && <g fill="none"><path d="M205 310L260 300L325 315L275 300L210 315L155 300" stroke="#c5fb65" strokeWidth="2" strokeDasharray="5 5" opacity=".5"/><path d="M300 110Q205 170 330 300Q260 200 210 95Q120 220 150 305" stroke="#5ee2f1" strokeWidth="1.5" strokeDasharray="6 5" opacity=".7"/></g>}
          <g className="moving-marker" style={{ transform: `translate(${frame.x}px, ${frame.y}px)` }}><circle r="42" fill="url(#player-glow)"/><circle r="17" fill="none" stroke="#c5fb65" strokeOpacity=".4"/><circle r="6" fill="#c5fb65"/><path d="M0 -12V-30M-4 -25L0 -30L4 -25" stroke="#c5fb65" fill="none"/><text x="22" y="5" fill="#d8edb8" fontSize="10">P1</text></g>
          <g className="moving-marker" style={{ transform: `translate(${500 - frame.x}px, 70px)` }}><circle r="12" stroke="#5ee2f1" fill="#142c35"/><circle r="4" fill="#5ee2f1"/><text x="20" y="4" fill="#a2e7ed" fontSize="10">P2</text></g>
          <circle className="moving-marker" r="4" fill="white" style={{ transform: `translate(${frame.bx}px, ${frame.by}px)` }}/>
        </svg>
        <div className="court-legend"><span><i className="player-dot"/> Player vector</span><span><i className="shot-dot"/> Shot trajectory</span></div>
      </div>
      <div className="hud-metrics">
        <div><span>Recovery time</span><strong>{frame.recovery}<small>s</small></strong><em className="lime">Return to position</em></div>
        <div><span>Split-step timing</span><strong>{frame.split}<small>ms</small></strong><em className="cyan">Relative to contact</em></div>
        <div><span>Lateral coverage</span><strong>{frame.coverage}<small>m</small></strong><em className="lime">Across the rally</em></div>
      </div>
      <div className="replay-controls">
        <button className="play-button" onClick={() => setPlaying(!playing)} aria-label={playing ? "Pause sample replay" : "Play sample replay"}>{playing ? "Ⅱ" : "▶"}</button>
        <input aria-label="Sample replay position" type="range" min="0" max={frames.length - 1} value={frameIndex} onChange={(event) => { setPlaying(false); setFrameIndex(Number(event.target.value)); }}/>
        <span>00:{String(frameIndex * 2).padStart(2, "0")} <span className="muted">/ 00:10</span></span>
      </div>
      <p className="demo-caption">Illustrative telemetry. Play or scrub to explore the interface.</p>
    </section>
  );
}
