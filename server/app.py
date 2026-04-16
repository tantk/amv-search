import asyncio
import os
import re
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

# Pre-import heavy libs at startup so first request isn't slow
# (librosa is ~2s to import, numpy another ~0.3s)
import librosa  # noqa: F401
import numpy  # noqa: F401

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Music Video Generator")

_background_tasks: set = set()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create dirs
(BASE_DIR / "jobs").mkdir(exist_ok=True)
(BASE_DIR / "static").mkdir(exist_ok=True)

# Mount static files — jobs dir for audio/video serving, static for frontend
app.mount("/static/jobs", StaticFiles(directory=str(BASE_DIR / "jobs")), name="jobs")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")


@app.get("/health")
async def health():
    """Lightweight endpoint to wake the Space from scale-to-zero."""
    return {"status": "ok"}


@app.get("/history")
async def history():
    """Return the list of previously generated videos from HF dataset."""
    try:
        import httpx as _httpx
        r = _httpx.get(
            "https://huggingface.co/datasets/tantk/amv-history/resolve/main/history.json",
            timeout=10, follow_redirects=True,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


class GenerateRequest(BaseModel):
    prompt: str
    mode: str = "anime"  # "anime" or "normal"


@app.post("/generate")
async def generate(req: GenerateRequest):
    job_id = uuid.uuid4().hex[:12]
    mode = req.mode if req.mode in ("anime", "normal") else "anime"

    async def _run():
        from pipeline import run_pipeline

        await run_pipeline(job_id, req.prompt, mode=mode)

    task = asyncio.create_task(_run())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"job_id": job_id}


class RegenerateRequest(BaseModel):
    source_job_id: str
    mode: str | None = None  # optionally change mode


@app.post("/regenerate")
async def regenerate(req: RegenerateRequest):
    """Regenerate a video from a prior job's audio+metadata.

    Skips ElevenLabs (saves ~9s). Reuses existing audio.mp3 + metadata.json
    from the source job's directory. Must be called while source job's
    files still exist on disk (free-tier Spaces wipe on restart).
    """
    _validate_job_id(req.source_job_id)

    src_dir = BASE_DIR / "jobs" / req.source_job_id
    if not (src_dir / "audio.mp3").exists() or not (src_dir / "metadata.json").exists():
        raise HTTPException(
            status_code=404,
            detail="Source job's audio/metadata not found (may have been wiped)",
        )

    import json as _json
    try:
        metadata = _json.loads((src_dir / "metadata.json").read_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load metadata: {e}")

    # Make a new job that reuses the prior audio
    job_id = uuid.uuid4().hex[:12]
    new_dir = BASE_DIR / "jobs" / job_id
    new_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(src_dir / "audio.mp3", new_dir / "audio.mp3")
    shutil.copy2(src_dir / "metadata.json", new_dir / "metadata.json")

    # Preserve or override mode
    mode = req.mode or "anime"
    if req.mode not in ("anime", "normal"):
        # Try to use the source's mode from timeline.json
        tl_path = src_dir / "timeline.json"
        if tl_path.exists():
            try:
                tl = _json.loads(tl_path.read_text())
                mode = tl.get("mode") or mode
            except Exception:
                pass

    # Get the original prompt too
    src_tl = src_dir / "timeline.json"
    prompt = ""
    if src_tl.exists():
        try:
            prompt = _json.loads(src_tl.read_text()).get("prompt", "")
        except Exception:
            pass

    async def _run():
        from pipeline import run_pipeline_from_audio
        await run_pipeline_from_audio(job_id, prompt=prompt, mode=mode, metadata=metadata)

    task = asyncio.create_task(_run())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"job_id": job_id, "source_job_id": req.source_job_id}


def _validate_job_id(job_id: str):
    if not re.match(r'^[a-f0-9]{12}$', job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")


@app.get("/job/{job_id}")
async def job_status(job_id: str):
    _validate_job_id(job_id)
    from pipeline import get_job

    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "error": job.error,
        "stages": job.stages,
        "title": job.title,
    }


@app.get("/preview/{job_id}")
async def preview(job_id: str):
    _validate_job_id(job_id)
    from pipeline import get_job

    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.timeline is None:
        return JSONResponse(status_code=202, content={"detail": "Timeline not ready"})
    return job.timeline


@app.get("/render/{job_id}")
async def render_info(job_id: str):
    _validate_job_id(job_id)
    from pipeline import get_job

    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.render_command is None:
        return JSONResponse(status_code=202, content={"detail": "Render not ready"})
    return {
        "command": job.render_command,
        "assets": job.render_assets,
        "video_url": job.video_url,
    }


@app.get("/video/{job_id}")
async def video(job_id: str):
    _validate_job_id(job_id)
    from pipeline import get_job

    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == "failed":
        return JSONResponse(
            status_code=500, content={"detail": job.error or "Pipeline failed"}
        )
    video_path = BASE_DIR / "jobs" / job_id / "video.mp4"
    if not video_path.exists():
        return JSONResponse(status_code=202, content={"detail": "Video not ready"})
    return FileResponse(str(video_path), media_type="video/mp4")
