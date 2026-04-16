"""
AMV-style anime music video generator — beat-synced rapid cuts.

Usage:
    python anime_video_v2.py test_runs/003_anime_chorus
    python anime_video_v2.py test_runs/004_anime_english

Differences from v1:
- Cuts on musical beats (librosa beat detection), not lyric boundaries
- Multiple clips per lyric line (one per beat)
- Cross-fade transitions between clips
- Intensity matching: action clips for chorus, calm for verse
"""

import json
import os
import subprocess
import sys
import tempfile
import time

import librosa
import numpy as np
import torch
import turbopuffer as tpuf

# ── Config ────────────────────────────────────────────────────
TURBOPUFFER_API_KEY = os.environ.get("TURBOPUFFER_API_KEY", "")
NAMESPACE = "anime-clips"
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIPS_BASE = os.path.join(BASE_DIR, "clips")

_embedder = None


def load_model():
    global _embedder
    if _embedder is not None:
        return _embedder
    print("Loading Qwen3-VL-Embedding-2B...", flush=True)
    sys.path.insert(0, os.path.join(BASE_DIR, "..", "indexing"))
    from qwen3_vl_embedding import Qwen3VLEmbedder
    _embedder = Qwen3VLEmbedder(model_name_or_path=MODEL_NAME, torch_dtype=torch.float16)
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB\n", flush=True)
    return _embedder


def embed_query(text: str) -> list[float]:
    embedder = load_model()
    inputs = [{"text": text, "instruction": "Find a video clip that matches this description."}]
    embeddings = embedder.process(inputs)
    return embeddings[0].cpu().numpy().tolist()


def search_clips(query: str, top_k: int = 10, exclude_ids: set = None) -> list[dict]:
    client = tpuf.Turbopuffer(api_key=TURBOPUFFER_API_KEY, region="gcp-us-central1")
    ns = client.namespace(NAMESPACE)
    query_vector = embed_query(query)

    # Hybrid search: vector + BM25 via multi_query + client-side RRF
    response = ns.multi_query(queries=[
        {
            "rank_by": ("vector", "ANN", query_vector),
            "top_k": top_k * 3,
            "include_attributes": ["category", "video_id", "caption"],
        },
        {
            "rank_by": ("caption", "BM25", query),
            "top_k": top_k * 3,
            "include_attributes": ["category", "video_id", "caption"],
        },
    ])

    # Reciprocal rank fusion
    rrf_scores = {}
    rrf_data = {}
    k = 60
    for result_set in response.results:
        for rank, row in enumerate(result_set.rows):
            d = row.to_dict()
            cid = d.get("id", "")
            score = 1.0 / (k + rank + 1)
            rrf_scores[cid] = rrf_scores.get(cid, 0) + score
            if cid not in rrf_data:
                rrf_data[cid] = d

    ranked = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    results = []
    for clip_id in ranked:
        if exclude_ids and clip_id in exclude_ids:
            continue
        parts = clip_id.rsplit("_clip_", 1)
        if len(parts) == 2:
            clip_path = os.path.join(CLIPS_BASE, parts[0], f"clip_{parts[1]}.mp4")
        else:
            continue
        if os.path.exists(clip_path):
            d = rrf_data[clip_id]
            results.append({
                "id": clip_id,
                "score": rrf_scores[clip_id],
                "category": d.get("category", ""),
                "video_id": d.get("video_id", ""),
                "path": clip_path,
                "caption": d.get("caption", ""),
            })
        if len(results) >= top_k:
            break
    return results


def detect_beats(audio_path: str):
    """Detect beat times and per-beat intensity using librosa."""
    y, sr = librosa.load(audio_path, sr=22050)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    # Onset strength envelope for intensity scoring
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.times_like(onset_env, sr=sr)

    # Compute per-beat intensity (0.0 - 1.0)
    beat_intensities = []
    for bt in beat_times:
        # Average onset strength in a window around each beat
        mask = np.abs(onset_times - bt) < 0.2
        if mask.any():
            beat_intensities.append(float(np.mean(onset_env[mask])))
        else:
            beat_intensities.append(0.0)

    # Normalize to 0-1
    if beat_intensities:
        max_i = max(beat_intensities) or 1.0
        beat_intensities = [i / max_i for i in beat_intensities]

    # Ensure we have a beat at 0
    if not beat_times or beat_times[0] > 0.1:
        beat_times.insert(0, 0.0)
        beat_intensities.insert(0, beat_intensities[0] if beat_intensities else 0.0)

    return beat_times, beat_intensities


def get_section_for_time(sections: list[dict], time_ms: int) -> dict:
    """Find which section a timestamp falls in."""
    cursor = 0
    for s in sections:
        end = cursor + s["duration_ms"]
        if time_ms < end:
            return s
        cursor = end
    return sections[-1] if sections else {}


def build_beat_timeline(metadata: dict, beat_times: list[float],
                        beat_intensities: list[float]) -> list[dict]:
    """Build a timeline entry per beat, with intensity scoring and speed tags."""
    sections = metadata["composition_plan"]["sections"]
    words = metadata.get("words_timestamps", [])

    # Classify sections by energy level
    high_energy_names = {"chorus", "climax", "drop", "bridge"}
    low_energy_names = {"intro", "outro", "interlude"}

    timeline = []
    total_dur_ms = sum(s["duration_ms"] for s in sections)

    for i, beat_t in enumerate(beat_times):
        beat_ms = int(beat_t * 1000)
        if beat_ms >= total_dur_ms:
            break

        # End time is next beat or song end
        if i + 1 < len(beat_times):
            end_ms = min(int(beat_times[i + 1] * 1000), total_dur_ms)
        else:
            end_ms = total_dur_ms

        # Find current section
        section = get_section_for_time(sections, beat_ms)
        section_name = section.get("section_name", "")
        styles = " ".join(section.get("positive_local_styles", [])[:2])

        # Intensity: combine audio onset strength with section type
        audio_intensity = beat_intensities[i] if i < len(beat_intensities) else 0.5
        section_lower = section_name.lower()
        if any(s in section_lower for s in high_energy_names):
            section_boost = 0.3
        elif any(s in section_lower for s in low_energy_names):
            section_boost = -0.2
        else:
            section_boost = 0.0
        intensity = max(0.0, min(1.0, audio_intensity + section_boost))

        # Speed: slow-mo for low intensity, normal/fast for high
        if intensity < 0.3:
            speed = 0.7  # slow-mo for calm sections
        elif intensity > 0.7:
            speed = 1.3  # speed up for action
        else:
            speed = 1.0

        # Find nearby words for context
        nearby_words = [w["word"] for w in words if abs(w["start_ms"] - beat_ms) < 2000]
        lyric_context = " ".join(nearby_words[:6]) if nearby_words else ""

        # Find the full line for this section
        lines = section.get("lines", [])
        line_idx = 0
        if lines:
            line_dur = section["duration_ms"] // len(lines)
            offset_in_section = beat_ms - sum(
                s["duration_ms"] for s in sections[:sections.index(section)]
            )
            line_idx = min(offset_in_section // max(line_dur, 1), len(lines) - 1)

        current_line = lines[line_idx] if lines else ""

        # Enrich search query with intensity hints
        if intensity > 0.7:
            mood = "intense action dramatic explosive"
        elif intensity < 0.3:
            mood = "calm peaceful quiet contemplative"
        else:
            mood = ""

        search_query = f"{current_line} {lyric_context} {styles} {mood}".strip()
        if not search_query:
            search_query = f"{section_name} {styles} {mood}"

        timeline.append({
            "start_ms": beat_ms,
            "end_ms": end_ms,
            "section": section_name,
            "lyric": current_line or f"[{section_name}]",
            "search_query": search_query,
            "intensity": round(intensity, 2),
            "speed": speed,
        })

    # Hold the last word of the song — find where the final word starts
    # and merge all beats from there to the end into one held segment
    if words and len(timeline) >= 3:
        last_word = words[-1]
        last_word_start_ms = last_word["start_ms"]

        # Find the first beat at or after the last word starts
        merge_start = None
        for i, entry in enumerate(timeline):
            if entry["start_ms"] >= last_word_start_ms - 200:  # 200ms tolerance
                merge_start = i
                break

        if merge_start is not None and merge_start < len(timeline) - 1:
            timeline[merge_start]["end_ms"] = timeline[-1]["end_ms"]
            timeline[merge_start]["lyric"] = last_word["word"]
            timeline[merge_start]["intensity"] = max(e["intensity"] for e in timeline[merge_start:])
            timeline[merge_start]["speed"] = 1.0
            timeline = timeline[:merge_start + 1]

    return timeline


def _make_black_segment(path: str, duration_s: float, width: int, height: int):
    """Generate a black video segment for gap filling."""
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        f"-i", f"color=c=black:s={width}x{height}:d={duration_s}:r=30",
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-bsf:v", "h264_mp4toannexb", "-f", "mpegts",
        path,
    ], capture_output=True)


def render_amv(timeline: list[dict], audio_path: str, output_path: str,
               width: int = 1920, height: int = 1080):
    """Render AMV with speed ramping and precise audio sync via MPEG-TS concat."""
    import random

    if not timeline:
        raise ValueError("Empty timeline")

    timeline = sorted(timeline, key=lambda e: e["start_ms"])
    valid = [e for e in timeline if "clip_path" in e and (e["end_ms"] - e["start_ms"]) > 50]

    if not valid:
        raise ValueError("No valid segments")

    with tempfile.TemporaryDirectory() as tmpdir:
        segments: list[str] = []
        cursor_ms = 0

        for idx, entry in enumerate(valid):
            start_ms = entry["start_ms"]
            end_ms = entry["end_ms"]
            need_s = (end_ms - start_ms) / 1000.0
            speed = entry.get("speed", 1.0)
            clip_path = entry["clip_path"]

            # Fill gap before this clip with black frames (maintains sync)
            gap_ms = start_ms - cursor_ms
            if gap_ms > 50:
                gap_path = os.path.join(tmpdir, f"gap_{idx:04d}.ts")
                _make_black_segment(gap_path, gap_ms / 1000.0, width, height)
                segments.append(gap_path)

            # Source clip duration
            try:
                probe = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", clip_path],
                    capture_output=True, check=True,
                )
                clip_dur = float(json.loads(probe.stdout)["format"]["duration"])
            except Exception:
                clip_dur = 5.0

            # How much source footage we need (accounting for speed change)
            source_need = need_s * speed
            max_offset = max(0.0, clip_dur - source_need)
            offset_s = random.uniform(0, max_offset) if max_offset > 0 else 0.0

            # Build video filter: scale + speed + fps
            vf_parts = [
                f"scale={width}:{height}:force_original_aspect_ratio=decrease",
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            ]
            if speed != 1.0:
                vf_parts.append(f"setpts=PTS/{speed}")
            vf_parts.append("fps=30")
            vf = ",".join(vf_parts)

            seg_path = os.path.join(tmpdir, f"seg_{idx:04d}.ts")
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(offset_s),
                "-i", clip_path,
                "-t", str(source_need),
                "-an",
                "-vf", vf,
                # Force exact output duration to prevent drift
                "-t", str(need_s),
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-bsf:v", "h264_mp4toannexb",
                "-f", "mpegts",
                seg_path,
            ], capture_output=True)

            if os.path.exists(seg_path) and os.path.getsize(seg_path) > 0:
                segments.append(seg_path)

            cursor_ms = end_ms

        if not segments:
            raise ValueError("No segments rendered")

        # Fill trailing gap to match audio duration
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", audio_path],
                capture_output=True, check=True,
            )
            audio_dur_ms = int(float(json.loads(probe.stdout)["format"]["duration"]) * 1000)
        except Exception:
            audio_dur_ms = cursor_ms

        trailing_ms = audio_dur_ms - cursor_ms
        if trailing_ms > 50:
            trail_path = os.path.join(tmpdir, "gap_trail.ts")
            _make_black_segment(trail_path, trailing_ms / 1000.0, width, height)
            segments.append(trail_path)

        # Concat all MPEG-TS segments (preserves sync)
        concat_input = "|".join(segments)
        concat_video = os.path.join(tmpdir, "concat.ts")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", f"concat:{concat_input}",
            "-c", "copy",
            concat_video,
        ], capture_output=True, check=True)

        # Mux with audio into final MP4
        subprocess.run([
            "ffmpeg", "-y",
            "-i", concat_video,
            "-i", audio_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-movflags", "+faststart",
            output_path,
        ], capture_output=True, check=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: python anime_video_v2.py <test_run_dir>")
        sys.exit(1)

    run_dir = sys.argv[1]
    metadata_path = os.path.join(run_dir, "metadata.json")
    audio_path = os.path.join(run_dir, "audio.mp3")

    with open(metadata_path) as f:
        metadata = json.load(f)

    title = metadata.get("song_metadata", {}).get("title", "untitled")
    print(f"Song: {title}\n")

    # ── Stage 1: Beat detection ───────────────────────────────
    print("=" * 60)
    print("STAGE 1: Detecting beats...")
    print("=" * 60, flush=True)

    beat_times, beat_intensities = detect_beats(audio_path)
    avg_bpm = 60 / np.mean(np.diff(beat_times)) if len(beat_times) > 1 else 120
    print(f"  Detected {len(beat_times)} beats, ~{avg_bpm:.0f} BPM")
    avg_intensity = np.mean(beat_intensities) if beat_intensities else 0.5
    print(f"  Average intensity: {avg_intensity:.2f}")

    # ── Stage 2: Build beat-synced timeline ───────────────────
    print("\n" + "=" * 60)
    print("STAGE 2: Building beat-synced timeline with intensity scoring...")
    print("=" * 60, flush=True)

    timeline = build_beat_timeline(metadata, beat_times, beat_intensities)
    print(f"  {len(timeline)} beat segments")

    # Group by unique search queries to batch searches
    unique_queries = {}
    for entry in timeline:
        q = entry["search_query"]
        if q not in unique_queries:
            unique_queries[q] = []
        unique_queries[q].append(entry)

    print(f"  {len(unique_queries)} unique search queries\n")

    # ── Stage 3: Search clips ─────────────────────────────────
    print("=" * 60)
    print("STAGE 3: Searching clips for each query...")
    print("=" * 60, flush=True)

    load_model()

    used_ids = set()

    # Process timeline sequentially so we can track used clips globally
    # Sort timeline by start time first
    timeline.sort(key=lambda e: e["start_ms"])

    # Cache query results to avoid re-searching the same query
    query_cache = {}

    for i, entry in enumerate(timeline):
        query = entry["search_query"]

        # Search if not cached, requesting lots of results
        if query not in query_cache:
            results = search_clips(query, top_k=30, exclude_ids=set())
            query_cache[query] = results
            lyric = entry["lyric"][:40]
            print(f"  [{len(query_cache)}/{len(unique_queries)}] \"{lyric}\" -> {len(results)} clips", flush=True)

        # Pick first unused clip from results
        results = query_cache[query]
        chosen = None
        for r in results:
            if r["id"] not in used_ids:
                chosen = r
                break

        # If all results used, pick least-recently-used
        if chosen is None and results:
            chosen = results[i % len(results)]

        if chosen:
            entry["clip_path"] = chosen["path"]
            entry["clip_id"] = chosen["id"]
            used_ids.add(chosen["id"])

    # Filter entries without clips
    timeline = [e for e in timeline if "clip_path" in e]
    print(f"\n  {len(timeline)} beats with clips assigned")

    # Show sample with intensity and speed
    for e in timeline[:10]:
        dur = e["end_ms"] - e["start_ms"]
        intensity = e.get("intensity", 0.5)
        speed = e.get("speed", 1.0)
        bar = "█" * int(intensity * 10) + "░" * (10 - int(intensity * 10))
        print(f"    {e['start_ms']/1000:.2f}s ({dur}ms) [{bar}] spd={speed}x -> {e['clip_id']}")
    if len(timeline) > 10:
        print(f"    ... and {len(timeline) - 10} more")

    # ── Stage 4: Render AMV ───────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 4: Rendering AMV...")
    print("=" * 60, flush=True)

    output_path = os.path.join(run_dir, "video_amv.mp4")
    render_amv(timeline, audio_path, output_path)

    # Save timeline
    timeline_path = os.path.join(run_dir, "timeline_amv.json")
    with open(timeline_path, "w") as f:
        json.dump(timeline, f, indent=2)

    print(f"\nDONE! AMV saved to: {output_path}")
    print(f"  Beats: {len(timeline)}")
    print(f"  BPM: ~{avg_bpm:.0f}")
    print(f"  Unique queries: {len(unique_queries)}")


if __name__ == "__main__":
    main()
