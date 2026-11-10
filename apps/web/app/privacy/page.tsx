import Link from "next/link";

export default function PrivacyPage() {
  return <main className="privacy-page landing-shell"><span className="eyebrow cyan">PROTOTYPE DATA HANDLING</span><h1>Your footage. Clear expectations.</h1><p>The video preview runs in your browser. Selecting a file does not upload it. Clicking Upload video sends it to the local Python API running on your computer.</p><p>The API saves uploaded videos in its backend/uploads folder under a generated filename. Files remain there until you delete them. This prototype does not currently offer account storage or automatic deletion.</p><p>The telemetry demonstration uses illustrative data and does not read your uploaded footage. A production privacy policy will be needed before a hosted service is offered.</p><Link className="secondary-action" href="/">← Back to CourtVision</Link></main>;
}

