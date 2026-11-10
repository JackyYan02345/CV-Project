"""Local FastAPI service for receiving tennis video uploads."""

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


MAX_VIDEO_SIZE_BYTES = 500 * 1024 * 1024
CHUNK_SIZE_BYTES = 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"

app = FastAPI(title="CourtVision Upload API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Report that the local API is ready."""
    return {"status": "ok"}


@app.post("/uploads", status_code=201)
def upload_video(file: UploadFile = File(...)):
    """Save one validated video and return its local upload identifier."""
    original_name = Path(file.filename or "video").name
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format. Use MP4, MOV, AVI, MKV, WEBM, or M4V.",
        )

    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="The uploaded file is not a video.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_id = uuid4().hex
    destination = UPLOAD_DIR / f"{upload_id}{extension}"
    size_bytes = 0

    try:
        with destination.open("xb") as output:
            while chunk := file.file.read(CHUNK_SIZE_BYTES):
                size_bytes += len(chunk)
                if size_bytes > MAX_VIDEO_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds the 500 MB upload limit.",
                    )
                output.write(chunk)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        file.file.close()

    return {
        "upload_id": upload_id,
        "filename": original_name,
        "size_bytes": size_bytes,
        "status": "uploaded",
        "message": "Video received. Analysis has not started yet.",
    }
