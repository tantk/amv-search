"""Three-case pacing test: slow-fast mix, fast, slow.

Generates each song fresh via ElevenLabs, runs stages 2-5 locally,
prints per-section pacing and total cut count.
"""
import asyncio
import json
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
    Job, _jobs, _job_dir,
    _stage_1_generate_music, _stage_2_build_timeline,
    _stage_3_search_clips, _stage_4_download_clips,
)

CASES = [
    ("MIX",  "Anime opening with quiet piano intro building into explosive j-rock chorus"),
    ("FAST", "Fast anime action opening, aggressive J-rock, 170bpm, heavy guitar, intense drums"),
    ("SLOW", "Very slow peaceful anime ending, solo piano lullaby, 65bpm, no drums, no percussion, whispered vocals, ambient"),
]


async def run_case(label: str, prompt: str):
    job_id = f"{label.lower()}_{uuid.uuid4().hex[:6]}"
    print(f"\n{'='*70}")
    print(f"CASE: {label} — {prompt[:60]}")
    print(f"{'='*70}")

    t_all = time.time()
    job = Job(job_id=job_id, prompt=prompt, mode="anime", duration_ms=30_000)
    _jobs[job_id] = job

    _audio, metadata = _stage_1_generate_music(job)
    dst_dir = _job_dir(job_id)

    timeline = _stage_2_build_timeline(job, metadata)
    timeline = _stage_3_search_clips(job, timeline)
    await _stage_4_download_clips(job, timeline)

    norm_tasks = job.timeline.get("_norm_tasks", []) if job.timeline else []
    results = await asyncio.gather(*[tsk for _, tsk in norm_tasks]) if norm_tasks else []
    seg_paths = [p for (_, p) in sorted(results, key=lambda x: x[0]) if p]

    merged_ts = dst_dir / "merged.ts"
    with open(merged_ts, "wb") as out:
        for p in seg_paths:
            out.write(Path(p).read_bytes())

    output_mp4 = dst_dir / "video_local.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(merged_ts), "-i", str(dst_dir / "audio.mp3"),
        "-c:v", "copy", "-c:a", "copy", "-shortest",
        "-movflags", "+faststart", str(output_mp4),
    ], capture_output=True, check=True)

    sections = metadata["composition_plan"]["sections"]
    cursor = 0
    pacing = []
    for s in sections:
        dur = s["duration_ms"]
        name = s["section_name"]
        cuts = [e for e in timeline if cursor <= e["start_ms"] < cursor + dur]
        pacing.append((name, dur/1000, len(cuts)))
        cursor += dur

    total_time = time.time() - t_all
    return {
        "label": label, "job_id": job_id, "title": job.title,
        "pacing": pacing, "total_cuts": len(timeline),
        "video": str(output_mp4), "time": total_time,
    }


async def main():
    results = []
    for label, prompt in CASES:
        r = await run_case(label, prompt)
        results.append(r)

    print(f"\n\n{'#'*70}\n# SUMMARY\n{'#'*70}")
    for r in results:
        rate = r["total_cuts"] / 30
        print(f"\n[{r['label']}] {r['title']!r} — {r['total_cuts']} cuts ({rate:.2f}/s avg) — {r['time']:.1f}s")
        for name, dur_s, n in r["pacing"]:
            cut_rate = n / dur_s if dur_s else 0
            print(f"  {name:<20} {dur_s:.1f}s  {n:2d} cuts ({cut_rate:.2f}/s)")
        print(f"  video: {r['video']}")


asyncio.run(main())
