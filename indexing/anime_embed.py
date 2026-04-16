"""
Anime Video Embedding Pipeline — Qwen3-VL-Embedding-2B (direct visual embedding)

Usage:
    python anime_embed.py "https://www.youtube.com/watch?v=XXXXX" --name "my_anime"

What it does:
1. Downloads the video from YouTube via yt-dlp
2. Splits into 5-second clips via ffmpeg
3. Extracts 3 frames per clip
4. Embeds frames directly with Qwen3-VL-Embedding-2B (no captioning)
5. Uploads vectors to turbopuffer namespace "anime-clips"

Requirements:
    pip install transformers qwen-vl-utils torch turbopuffer yt-dlp
    ffmpeg must be on PATH
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

# ── Config ────────────────────────────────────────────────────
TURBOPUFFER_API_KEY = os.environ.get("TURBOPUFFER_API_KEY", "")
NAMESPACE = "anime-clips"
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
UPSERT_BATCH = 200
EMBED_BATCH = 4  # clips per batch (each clip = 3 frames, so 12 images per batch)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIPS_DIR = os.path.join(BASE_DIR, "clips")
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")

_embedder = None  # loaded once, reused across batch


def download_video(url: str, name: str) -> str:
    """Download video from YouTube using yt-dlp."""
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)
    output_path = os.path.join(DOWNLOADS_DIR, f"{name}.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--merge-output-format", "mp4",
        "-o", output_path,
        url,
    ]
    print(f"Downloading: {url}")
    subprocess.run(cmd, check=True)
    # Find the downloaded file
    matches = glob.glob(os.path.join(DOWNLOADS_DIR, f"{name}.*"))
    if not matches:
        raise FileNotFoundError(f"Download failed — no file found for {name}")
    return matches[0]


def split_video(video_path: str, name: str, start: str = None, end: str = None) -> list[str]:
    """Split video into 5-second clips using ffmpeg, with optional time range."""
    clip_dir = os.path.join(CLIPS_DIR, name)
    os.makedirs(clip_dir, exist_ok=True)
    cmd = ["ffmpeg", "-y"]
    if start:
        cmd += ["-ss", start]
    cmd += ["-i", video_path]
    if end:
        cmd += ["-to", end]
    cmd += [
        "-c", "copy", "-f", "segment",
        "-segment_time", "5",
        "-reset_timestamps", "1",
        os.path.join(clip_dir, "clip_%04d.mp4"),
    ]
    range_str = f" ({start or '0:00'} - {end or 'end'})" if start or end else ""
    print(f"Splitting into 5s clips{range_str}...")
    subprocess.run(cmd, check=True, capture_output=True)
    clips = sorted(glob.glob(os.path.join(clip_dir, "clip_*.mp4")))
    print(f"Created {len(clips)} clips")
    return clips


def get_clip_duration(clip_path: str) -> float:
    """Get clip duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", clip_path],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


def extract_frames(clip_path: str, name: str) -> list[str]:
    """Extract 3 frames from a clip at 25%, 50%, 75% of duration."""
    frame_dir = os.path.join(FRAMES_DIR, name)
    os.makedirs(frame_dir, exist_ok=True)
    clip_name = os.path.splitext(os.path.basename(clip_path))[0]

    duration = get_clip_duration(clip_path)
    if duration < 0.5:
        return []  # too short

    timestamps = [duration * pct for pct in [0.25, 0.5, 0.75]]
    frames = []
    for i, t in enumerate(timestamps):
        out = os.path.join(frame_dir, f"{clip_name}_f{i}.jpg")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(t), "-i", clip_path,
             "-frames:v", "1", "-q:v", "2", out],
            capture_output=True,
        )
        if os.path.exists(out) and os.path.getsize(out) > 0:
            frames.append(out)
    return frames


def main():
    parser = argparse.ArgumentParser(description="Embed anime video clips with Qwen3-VL-Embedding-2B")
    parser.add_argument("url", nargs="?", help="YouTube video URL")
    parser.add_argument("--name", help="Name tag for this video (e.g. 'naruto_ep1')")
    parser.add_argument("--start", help="Start time (e.g. '1:00' or '90')")
    parser.add_argument("--end", help="End time (e.g. '22:00' or '1320')")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, use existing file")
    parser.add_argument("--video-path", help="Path to local video file (skip download)")
    parser.add_argument("--batch", help="Path to batch JSON file (process multiple videos)")
    args = parser.parse_args()

    if args.batch:
        # Batch mode — process multiple videos from JSON config
        with open(args.batch) as f:
            jobs = json.load(f)
        print(f"Batch mode: {len(jobs)} videos to process\n")
        for i, job in enumerate(jobs):
            print(f"\n{'#' * 60}")
            print(f"# Video {i+1}/{len(jobs)}: {job['name']}")
            print(f"{'#' * 60}\n")
            process_video(
                url=job["url"],
                name=job["name"],
                start=job.get("start"),
                end=job.get("end"),
            )
        print(f"\nAll {len(jobs)} videos processed!")
        return

    if not args.url:
        parser.error("url is required (or use --batch)")
    if not args.name:
        parser.error("--name is required")

    process_video(
        url=args.url,
        name=args.name,
        start=args.start,
        end=args.end,
        skip_download=args.skip_download,
        video_path=args.video_path,
    )


def process_video(url: str, name: str, start: str = None, end: str = None,
                   skip_download: bool = False, video_path: str = None):
    """Process a single video: download, split, embed, upload."""

    # ── Step 1: Download ──────────────────────────────────────
    if video_path:
        vpath = video_path
        print(f"Using local file: {vpath}")
    elif skip_download:
        matches = glob.glob(os.path.join(DOWNLOADS_DIR, f"{name}.*"))
        if not matches:
            print(f"No downloaded file found for {name}")
            return
        vpath = matches[0]
        print(f"Using existing: {vpath}")
    else:
        vpath = download_video(url, name)
    print(f"Video: {vpath}\n")

    # ── Step 2: Split into clips ──────────────────────────────
    clips = split_video(vpath, name, start=start, end=end)
    if not clips:
        print("No clips created!")
        return

    # ── Step 3: Extract frames ────────────────────────────────
    print(f"\nExtracting frames from {len(clips)} clips...")
    clip_frames = []
    for i, clip_path in enumerate(clips):
        frames = extract_frames(clip_path, name)
        clip_frames.append(frames)
        if (i + 1) % 100 == 0:
            print(f"  Extracted frames for {i + 1}/{len(clips)} clips")
    print(f"Done extracting frames\n")

    # ── Step 4: Load model (once, cached globally) ────────────
    global _embedder
    if _embedder is None:
        print("=" * 60)
        print(f"Loading {MODEL_NAME}...")
        print("=" * 60)

        sys.path.insert(0, BASE_DIR)
        from qwen3_vl_embedding import Qwen3VLEmbedder

        _embedder = Qwen3VLEmbedder(
            model_name_or_path=MODEL_NAME,
            torch_dtype=torch.float16,
        )
        print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB\n")

    # ── Step 5: Embed clips ───────────────────────────────────
    print("=" * 60)
    print(f"Embedding {len(clips)} clips...")
    print("=" * 60)

    all_vectors = []
    embed_start = time.time()

    skipped = 0
    for i in range(len(clips)):
        frames = clip_frames[i]
        if not frames:
            # No frames (clip too short) — use a zero vector as placeholder
            skipped += 1
            all_vectors.append(np.zeros((1, 2048), dtype=np.float32))
            continue

        try:
            # Embed frames one at a time to avoid batch issues, then mean-pool
            frame_vecs = []
            for f in frames:
                vec = _embedder.process([{"image": f}])
                frame_vecs.append(vec)
            clip_vec = torch.cat(frame_vecs, dim=0).mean(dim=0, keepdim=True)
            clip_vec = torch.nn.functional.normalize(clip_vec, p=2, dim=-1)
            all_vectors.append(clip_vec.cpu().numpy())
        except Exception as e:
            print(f"  WARNING: clip {i} failed ({e}), skipping", flush=True)
            all_vectors.append(np.zeros((1, 2048), dtype=np.float32))
            skipped += 1

        done = i + 1
        if done % 20 == 0 or done == len(clips):
            elapsed = time.time() - embed_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(clips) - done) / rate if rate > 0 else 0
            print(f"  {done}/{len(clips)} ({100*done//len(clips)}%) - {rate:.1f} clips/sec - ETA {eta:.0f}s", flush=True)

    all_vectors = np.vstack(all_vectors)
    embed_time = time.time() - embed_start
    if skipped:
        print(f"\n  Skipped {skipped} clips (too short or failed)")
    print(f"\nEmbedding done in {embed_time:.1f}s")
    print(f"Vector shape: {all_vectors.shape}")

    # Save vectors to disk
    vectors_path = os.path.join(BASE_DIR, f"vectors_{name}.npy")
    np.save(vectors_path, all_vectors)
    print(f"Saved to {vectors_path}\n")

    # ── Step 6: Upload to turbopuffer ─────────────────────────
    print("=" * 60)
    print(f"Uploading {len(clips)} vectors to turbopuffer...")
    print("=" * 60)

    import turbopuffer as tpuf
    client = tpuf.Turbopuffer(api_key=TURBOPUFFER_API_KEY, region="gcp-us-central1")
    ns = client.namespace(NAMESPACE)

    upload_start = time.time()
    total = len(clips)

    for batch_start in range(0, total, UPSERT_BATCH):
        batch_end = min(batch_start + UPSERT_BATCH, total)
        rows = []
        for idx in range(batch_start, batch_end):
            clip_path = clips[idx]
            clip_name = os.path.splitext(os.path.basename(clip_path))[0]
            clip_id = f"{name}_{clip_name}"
            rows.append({
                "id": clip_id,
                "vector": all_vectors[idx].tolist(),
                "source": "youtube",
                "category": name,
                "video_id": os.path.basename(clip_path),
                "path": os.path.abspath(clip_path),
                "timestamp": idx * 5,
            })
        ns.write(upsert_rows=rows, distance_metric="cosine_distance")
        elapsed = time.time() - upload_start
        print(f"  Upserted {batch_end}/{total} ({100*batch_end//total}%) - {elapsed:.0f}s", flush=True)

    total_time = time.time() - embed_start
    print(f"\n{'=' * 60}")
    print(f"DONE: {name}")
    print(f"  Clips:     {total}")
    print(f"  Embedded:  {embed_time:.1f}s")
    print(f"  Uploaded:  {time.time() - upload_start:.0f}s")
    print(f"  Total:     {total_time:.1f}s")
    print(f"  Namespace: {NAMESPACE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
