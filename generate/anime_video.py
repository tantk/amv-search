"""
Generate a music video using anime clips from turbopuffer.

Usage:
    python anime_video.py test_runs/001_among_the_stars
    python anime_video.py test_runs/002_windowpane_dreams

Reads metadata.json + audio.mp3 from the given directory,
searches anime-clips namespace with Qwen3-VL-Embedding-2B,
and renders a final video using local anime clips.
"""

import json
import os
import sys
import time

import numpy as np
import torch
import turbopuffer as tpuf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from render import render_video

# ── Config ────────────────────────────────────────────────────
TURBOPUFFER_API_KEY = os.environ.get("TURBOPUFFER_API_KEY", "")
NAMESPACE = "anime-clips"
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_embedder = None


def load_model():
    global _embedder
    if _embedder is not None:
        return _embedder

    print("Loading Qwen3-VL-Embedding-2B...", flush=True)
    sys.path.insert(0, os.path.join(BASE_DIR, "..", "indexing"))
    from qwen3_vl_embedding import Qwen3VLEmbedder

    _embedder = Qwen3VLEmbedder(
        model_name_or_path=MODEL_NAME,
        torch_dtype=torch.float16,
    )
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB\n", flush=True)
    return _embedder


def embed_query(text: str) -> list[float]:
    embedder = load_model()
    inputs = [{"text": text, "instruction": "Find a video clip that matches this description."}]
    embeddings = embedder.process(inputs)
    return embeddings[0].cpu().numpy().tolist()


def search_clips(query: str, top_k: int = 5, exclude_ids: set = None) -> list[dict]:
    """Search anime-clips namespace, return list of clip dicts."""
    client = tpuf.Turbopuffer(api_key=TURBOPUFFER_API_KEY, region="gcp-us-central1")
    ns = client.namespace(NAMESPACE)

    query_vector = embed_query(query)

    # Hybrid search: vector ANN + BM25 on captions via multi-query + RRF
    response = ns.multi_query(
        queries=[
            {
                "rank_by": ("vector", "ANN", query_vector),
                "limit": {
                    "per": {"attributes": ["category"], "limit": 3},
                    "total": top_k * 3,
                },
                "include_attributes": ["category", "video_id", "caption"],
            },
            {
                "rank_by": ("caption", "BM25", query),
                "limit": {
                    "per": {"attributes": ["category"], "limit": 3},
                    "total": top_k * 3,
                },
                "include_attributes": ["category", "video_id", "caption"],
            },
        ],
    )

    # Reciprocal rank fusion
    rrf_scores = {}
    rrf_data = {}
    k = 60  # RRF constant
    for qi, result_set in enumerate(response.results):
        for rank, row in enumerate(result_set.rows):
            d = row.to_dict()
            cid = d.get("id", "")
            score = 1.0 / (k + rank + 1)
            rrf_scores[cid] = rrf_scores.get(cid, 0) + score
            if cid not in rrf_data:
                rrf_data[cid] = d

    # Sort by fused score
    ranked = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    results = []
    clips_base = os.path.join(BASE_DIR, "clips")
    for clip_id in ranked:
        if exclude_ids and clip_id in exclude_ids:
            continue
        d = rrf_data[clip_id]
        # Derive path from clip_id: "aot_s3e14_clip_0120" -> clips/aot_s3e14/clip_0120.mp4
        parts = clip_id.rsplit("_clip_", 1)
        if len(parts) == 2:
            category = parts[0]
            clip_file = f"clip_{parts[1]}.mp4"
            clip_path = os.path.join(clips_base, category, clip_file)
        else:
            clip_path = d.get("path", "")
        if clip_path and os.path.exists(clip_path):
            results.append({
                "id": clip_id,
                "similarity": rrf_scores[clip_id],
                "category": d.get("category", parts[0] if len(parts) == 2 else ""),
                "video_id": clip_file if len(parts) == 2 else d.get("video_id", ""),
                "path": clip_path,
                "timestamp": d.get("timestamp", 0),
                "caption": d.get("caption", ""),
            })
        if len(results) >= top_k:
            break

    return results


def build_timeline(metadata: dict) -> list[dict]:
    """Build timeline entries from ElevenLabs metadata.

    Groups lyrics into segments based on sections, creating one timeline
    entry per section or per meaningful phrase.
    """
    sections = metadata["composition_plan"]["sections"]
    words = metadata.get("words_timestamps", [])

    timeline = []
    cursor_ms = 0

    for section in sections:
        section_name = section["section_name"]
        duration_ms = section["duration_ms"]
        lines = section.get("lines", [])
        start_ms = cursor_ms
        end_ms = cursor_ms + duration_ms

        if lines:
            # Split section into per-line entries
            line_dur = duration_ms // len(lines)
            for i, line in enumerate(lines):
                entry_start = start_ms + i * line_dur
                entry_end = start_ms + (i + 1) * line_dur
                # Use section styles + lyric line as search query
                styles = " ".join(section.get("positive_local_styles", [])[:2])
                search_query = f"{line} {styles}".strip()

                timeline.append({
                    "start_ms": entry_start,
                    "end_ms": entry_end,
                    "lyric": line,
                    "section": section_name,
                    "search_query": search_query,
                })
        else:
            # Instrumental section — use styles as query
            styles = " ".join(section.get("positive_local_styles", [])[:3])
            description = metadata.get("song_metadata", {}).get("description", "")
            search_query = f"{styles} {description[:50]}".strip()

            timeline.append({
                "start_ms": start_ms,
                "end_ms": end_ms,
                "lyric": f"[{section_name}]",
                "section": section_name,
                "search_query": search_query,
            })

        cursor_ms = end_ms

    return timeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python anime_video.py <test_run_dir>")
        sys.exit(1)

    run_dir = sys.argv[1]
    metadata_path = os.path.join(run_dir, "metadata.json")
    audio_path = os.path.join(run_dir, "audio.mp3")

    if not os.path.exists(metadata_path):
        print(f"Missing: {metadata_path}")
        sys.exit(1)
    if not os.path.exists(audio_path):
        print(f"Missing: {audio_path}")
        sys.exit(1)

    with open(metadata_path) as f:
        metadata = json.load(f)

    title = metadata.get("song_metadata", {}).get("title", "untitled")
    print(f"Song: {title}")
    print(f"Audio: {audio_path}\n")

    # ── Stage 1: Build timeline from metadata ─────────────────
    print("=" * 60)
    print("STAGE 1: Building timeline from lyrics...")
    print("=" * 60)
    timeline = build_timeline(metadata)
    for entry in timeline:
        print(f"  [{entry['start_ms']/1000:.1f}s - {entry['end_ms']/1000:.1f}s] {entry['lyric']}")
        print(f"    query: {entry['search_query'][:80]}")
    print()

    # ── Stage 2: Search anime clips ───────────────────────────
    print("=" * 60)
    print("STAGE 2: Searching anime clips for each segment...")
    print("=" * 60)

    load_model()  # warm up model once

    used_ids = set()
    used_categories = []

    for entry in timeline:
        results = search_clips(entry["search_query"], top_k=5, exclude_ids=used_ids)
        if not results:
            print(f"  WARNING: no clips found for '{entry['lyric']}'", flush=True)
            continue

        # Pick best result, preferring variety across categories
        chosen = results[0]
        for r in results:
            if r["category"] not in used_categories[-2:]:
                chosen = r
                break

        entry["clip_path"] = chosen["path"]
        entry["clip_id"] = chosen["id"]
        used_ids.add(chosen["id"])
        used_categories.append(chosen["category"])

        print(f"  [{entry['start_ms']/1000:.1f}s] \"{entry['lyric'][:40]}\" -> {chosen['category']}/{chosen['video_id']} (sim={chosen['similarity']:.3f})", flush=True)
    print()

    # Filter out entries without clips
    timeline = [e for e in timeline if "clip_path" in e]

    if not timeline:
        print("ERROR: No clips matched any lyrics!")
        sys.exit(1)

    # ── Stage 3: Render video ─────────────────────────────────
    print("=" * 60)
    print("STAGE 3: Rendering video...")
    print("=" * 60)

    output_path = os.path.join(run_dir, "video_anime.mp4")

    render_timeline = [
        {"start_ms": e["start_ms"], "end_ms": e["end_ms"], "clip_path": e["clip_path"]}
        for e in timeline
    ]

    # Save timeline for reference
    timeline_path = os.path.join(run_dir, "timeline_anime.json")
    with open(timeline_path, "w") as f:
        json.dump(timeline, f, indent=2)

    render_video(render_timeline, audio_path, output_path)

    print(f"\nDONE! Video saved to: {output_path}")
    print(f"Timeline saved to: {timeline_path}")


if __name__ == "__main__":
    main()
