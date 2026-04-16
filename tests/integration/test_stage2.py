"""Quick local test of the new pacing algorithm.

Runs only _stage_2_build_timeline on a pre-fetched audio + metadata
and reports per-section cut counts vs. targets.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "server"))

from pipeline import _stage_2_build_timeline, Job  # noqa: E402

JOB_ID = "181130502dc8"

job = Job(job_id=JOB_ID, prompt="", mode="anime", duration_ms=30_000)
metadata = json.loads((ROOT / "server/jobs" / JOB_ID / "metadata.json").read_text())

timeline = _stage_2_build_timeline(job, metadata)

# Per-section breakdown
sections = metadata["composition_plan"]["sections"]
cursor = 0
section_bounds = []
for s in sections:
    dur = s["duration_ms"]
    section_bounds.append((s["section_name"], cursor, cursor + dur))
    cursor += dur

rates = {"Intro": 0.35, "Pre-Chorus": 1.2, "Chorus": 1.4}

print()
print(f"=== Timeline: {len(timeline)} cuts total ===")
for name, start, end in section_bounds:
    dur_s = (end - start) / 1000
    cuts = [e for e in timeline if start <= e["start_ms"] < end]
    n = len(cuts)
    rate_actual = n / dur_s if dur_s else 0
    rate_target = rates.get(name, 0.6)
    target_n = round(rate_target * dur_s)
    status = "OK" if abs(n - target_n) <= 2 else "OFF"
    print(f"  [{name:<12}] {n:2d} cuts ({rate_actual:.2f}/s) — target {target_n} ({rate_target:.2f}/s) [{status}]")

print()
print("Cut times:")
for e in timeline:
    t = e["start_ms"] / 1000
    sec = e.get("section", "")
    intensity = e.get("intensity", 0)
    print(f"  {t:5.2f}s [{sec:<12}] intensity={intensity:.2f} speed={e.get('speed')}")
