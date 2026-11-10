"use client";

import { useEffect, useRef, useState } from "react";

const MAX_VIDEO_SIZE_BYTES = 500 * 1024 * 1024;

function formatFileSize(sizeBytes: number) {
  return `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function UploadPage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  function handleVideoChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];

    setError(null);
    setStatusMessage(null);

    if (!file) return;

    if (file.type && !file.type.startsWith("video/")) {
      setSelectedFile(null);
      setVideoUrl(null);
      setError("Choose a video file to continue.");
      event.target.value = "";
      return;
    }

    if (file.size > MAX_VIDEO_SIZE_BYTES) {
      setSelectedFile(null);
      setVideoUrl(null);
      setError("This video is larger than the 500 MB limit.");
      event.target.value = "";
      return;
    }

    setSelectedFile(file);
    setVideoUrl(URL.createObjectURL(file));
  }

  function removeVideo() {
    setSelectedFile(null);
    setVideoUrl(null);
    setError(null);
    setStatusMessage(null);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  async function uploadVideo() {
    if (!selectedFile || isUploading) return;

    setError(null);
    setStatusMessage(null);
    setIsUploading(true);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://localhost:8000/uploads", {
        method: "POST",
        body: formData,
      });
      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail ?? "The upload failed.");
      }

      setStatusMessage(
        `Video received (${result.upload_id}). Analysis has not started yet.`,
      );
    } catch (uploadError) {
      setError(
        uploadError instanceof Error
          ? uploadError.message
          : "Could not connect to the upload service. Is the Python API running?",
      );
    } finally {
      setIsUploading(false);
    }
  }

  return (
    <main className="mx-auto min-h-[75vh] max-w-[90rem] px-6 py-12 sm:py-16 lg:px-10">
      <div className="mb-9">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-lime-300">Video workspace · 01</p>
        <h1 className="mt-3 text-4xl font-semibold tracking-tight text-white">Bring your rally into focus.</h1>
        <p className="mt-3 max-w-3xl text-base leading-7 text-slate-300">Choose a clip to preview. When you’re ready, upload it to your local CourtVision workspace.</p>
      </div>

      <div className="grid gap-5 lg:grid-cols-[1.5fr_0.7fr]">
        <section className="rounded-3xl border border-white/10 bg-white/[0.035] p-5 sm:p-8" aria-label="Video upload">
          <div className="flex items-center justify-between gap-4">
            <div><h2 className="font-semibold text-white">Your video</h2><p className="mt-1 text-sm text-slate-400">MP4, MOV, AVI, MKV, WEBM, or M4V</p></div>
            <span className="rounded-full border border-white/10 px-3 py-1.5 text-xs text-slate-400">Up to 500 MB</span>
          </div>

          <div className="mt-6 rounded-2xl border border-dashed border-lime-200/25 bg-[#0a1513] p-5 sm:p-7">
            <div className="flex flex-col items-center py-6 text-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-lime-300/20 bg-lime-300/10 text-2xl text-lime-300" aria-hidden="true">↑</div>
              <p className="mt-4 font-medium text-white">Select a video from your device</p>
              <p className="mt-1 text-sm text-slate-500">A local preview appears before upload.</p>
              <label htmlFor="video-file" className="mt-5 cursor-pointer rounded-lg border border-white/15 bg-white/[0.05] px-4 py-2.5 text-sm font-semibold text-white transition hover:border-lime-300/40 hover:bg-white/[0.08] focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-lime-300">
                Browse files
              </label>
              <input
                ref={fileInputRef}
                id="video-file"
                className="sr-only"
                type="file"
                accept="video/*"
                aria-describedby={error ? "video-error" : "video-help"}
                onChange={handleVideoChange}
              />
              <p id="video-help" className="mt-3 text-xs text-slate-600">Your video stays on this machine until you upload it.</p>
            </div>
          </div>

          {error && <p id="video-error" role="alert" className="mt-4 rounded-xl border border-red-400/20 bg-red-400/5 px-4 py-3 text-sm text-red-300">{error}</p>}

          {selectedFile && (
            <div className="mt-5 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-white/10 bg-white/[0.03] p-4">
              <div className="flex min-w-0 items-center gap-3">
                <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-lime-300/10 text-lime-300" aria-hidden="true">▶</span>
                <div className="min-w-0"><p className="truncate text-sm font-medium text-white">{selectedFile.name}</p><p className="mt-1 text-xs text-slate-500">{formatFileSize(selectedFile.size)} · Ready to upload</p></div>
              </div>
              <button type="button" onClick={removeVideo} disabled={isUploading} className="rounded-lg px-3 py-2 text-sm text-slate-400 transition hover:bg-white/[0.06] hover:text-white disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-lime-300">Remove</button>
            </div>
          )}

          {videoUrl && <video className="mt-5 max-h-[480px] w-full rounded-xl border border-white/10 bg-black" src={videoUrl} controls aria-label="Selected video preview" />}

          <div className="mt-6 flex flex-wrap items-center gap-4">
            <button type="button" disabled={!selectedFile || isUploading} onClick={uploadVideo} className="inline-flex items-center gap-2 rounded-xl bg-lime-300 px-5 py-3 font-semibold text-slate-950 transition hover:bg-lime-200 disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-lime-300">
              {isUploading ? "Uploading…" : "Upload video"} {!isUploading && <span aria-hidden="true">→</span>}
            </button>
            <p className="text-xs text-slate-500">Uploading does not start analysis yet.</p>
          </div>

          {statusMessage && <p role="status" className="mt-4 rounded-xl border border-lime-300/20 bg-lime-300/5 px-4 py-3 text-sm text-lime-200">{statusMessage}</p>}
        </section>

        <aside className="rounded-3xl border border-white/10 bg-[#101b1b] p-6 sm:p-7">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">What happens next</p>
          <ol className="mt-6 space-y-6">
            <li className="flex gap-4"><span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-lime-300/30 text-xs font-semibold text-lime-300">1</span><div><p className="text-sm font-medium text-white">Choose your clip</p><p className="mt-1 text-sm leading-5 text-slate-500">Preview it here and check you picked the right rally.</p></div></li>
            <li className="flex gap-4"><span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-white/10 text-xs font-semibold text-slate-400">2</span><div><p className="text-sm font-medium text-white">Upload to your workspace</p><p className="mt-1 text-sm leading-5 text-slate-500">The local API saves the video and returns an upload ID.</p></div></li>
            <li className="flex gap-4"><span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-white/10 text-xs font-semibold text-slate-400">3</span><div><p className="text-sm font-medium text-white">Analyze your movement</p><p className="mt-1 text-sm leading-5 text-slate-500">Analysis is still being connected to the upload flow.</p></div></li>
          </ol>
          <div className="mt-8 rounded-xl border border-white/[0.07] bg-black/10 p-4"><p className="text-xs font-semibold text-slate-300">Local development</p><p className="mt-1.5 text-xs leading-5 text-slate-500">Keep the Python API running on port 8000 while uploading from this page.</p></div>
        </aside>
      </div>
    </main>
  );
}
