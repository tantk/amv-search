"""Generate a fast anime song and run full pipeline locally."""
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "server"))
load_dotenv(ROOT / ".env")

from pipeline import (  # noqa: E402
    Job,
    _jobs,
    _job_dir,
    _stage_1_generate_music,
    _stage_2_build_timeline,
    _stage_3_search_clips,
    _stage_4_download_clips,
)

PROMPT = "Fast anime action opening, aggressive J-rock, 170bpm, heavy guitar, intense drums, dramatic female vocals"
JOB_ID = f"fast_{uuid.uuid4().hex[:8]}"


async def main():
    t_all = time.time()
    job = Job(job_id=JOB_ID, prompt=PROMPT, mode="anime", duration_ms=30_000)
    _jobs[JOB_ID] = job

    # Stage 1 — generate music
    t = time.time()
    _audio, metadata = _stage_1_generate_music(job)
    print(f"[local] Stage 1: {time.time()-t:.1f}s — title={job.title!r}")

    dst_dir = _job_dir(JOB_ID)

    # Stage 2
    t = time.time()
    timeline = _stage_2_build_timeline(job, metadata)
    print(f"[local] Stage 2: {time.time()-t:.1f}s, {len(timeline)} cuts")

    # Stage 3
    t = time.time()
    timeline = _stage_3_search_clips(job, timeline)
    print(f"[local] Stage 3: {time.time()-t:.1f}s")

    # Stage 4
    t = time.time()
    await _stage_4_download_clips(job, timeline)
    print(f"[local] Stage 4: {time.time()-t:.1f}s")

    # Stage 5 — local ffmpeg
    norm_tasks = job.timeline.get("_norm_tasks", []) if job.timeline else []
    if norm_tasks:
        results = await asyncio.gather(*[tsk for _, tsk in norm_tasks])
        seg_paths = [p for (_, p) in sorted(results, key=lambda x: x[0]) if p]
    else:
        seg_paths = []

    merged_ts = dst_dir / "merged.ts"
    with open(merged_ts, "wb") as out:
        for p in seg_paths:
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    out.write(chunk)

    output_mp4 = dst_dir / "video_local.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(merged_ts),
        "-i", str(dst_dir / "audio.mp3"),
        "-c:v", "copy", "-c:a", "copy", "-shortest",
        "-movflags", "+faststart",
        str(output_mp4),
    ], capture_output=True, check=True)
    print(f"[local] Final video: {output_mp4} ({output_mp4.stat().st_size/1024:.1f} KB)")
    print(f"[local] Total time: {time.time()-t_all:.1f}s")

    # Per-section pacing
    sections = metadata["composition_plan"]["sections"]
    cursor = 0
    print("\n=== Pacing ===")
    for s in sections:
        dur = s["duration_ms"]
        name = s["section_name"]
        cuts = [e for e in timeline if cursor <= e["start_ms"] < cursor + dur]
        rate = len(cuts) / (dur / 1000) if dur else 0
        print(f"  [{name:<12}] {dur/1000:.1f}s — {len(cuts):2d} cuts ({rate:.2f}/s)")
        cursor += dur


asyncio.run(main())
