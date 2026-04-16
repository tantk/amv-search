"""Render a music video from a timeline of clips + audio using ffmpeg."""

import json
import os
import random
import subprocess
import tempfile


def _get_clip_duration(clip_path: str) -> float:
    """Get duration of a video clip in seconds."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", clip_path],
            capture_output=True, check=True,
        )
        return float(json.loads(probe.stdout)["format"]["duration"])
    except (subprocess.CalledProcessError, KeyError, ValueError):
        return 0.0


def _make_black_segment(path: str, duration_s: float, width: int, height: int):
    """Generate a black video segment."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"color=c=black:s={width}x{height}:d={duration_s}:r=30",
            "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-bsf:v", "h264_mp4toannexb", "-f", "mpegts",
            path,
        ],
        capture_output=True, check=True,
    )


def render_video(
    timeline: list[dict],
    audio_path: str,
    output_path: str,
    width: int = 1920,
    height: int = 1080,
) -> None:
    """Stitch timeline clips into a single MP4 synced to audio timestamps.

    Each clip is placed at its exact start_ms position. Gaps between clips
    are filled with black frames to maintain audio sync.

    Args:
        timeline: List of dicts with keys ``start_ms``, ``end_ms``, ``clip_path``.
        audio_path: Path to the audio file to overlay.
        output_path: Destination path for the rendered MP4.
        width: Output video width in pixels.
        height: Output video height in pixels.
    """
    if not timeline:
        raise ValueError("timeline must not be empty")

    # Sort by start time
    timeline = sorted(timeline, key=lambda e: e["start_ms"])

    with tempfile.TemporaryDirectory() as tmpdir:
        segments: list[str] = []
        cursor_ms = 0

        for idx, entry in enumerate(timeline):
            start_ms = entry["start_ms"]
            end_ms = entry["end_ms"]
            need_s = (end_ms - start_ms) / 1000.0

            # Fill gap before this clip with black frames
            gap_ms = start_ms - cursor_ms
            if gap_ms > 50:  # ignore tiny gaps < 50ms
                gap_path = os.path.join(tmpdir, f"gap_{idx:04d}.ts")
                _make_black_segment(gap_path, gap_ms / 1000.0, width, height)
                segments.append(gap_path)

            # Trim the clip with a random offset into the source video
            trimmed = os.path.join(tmpdir, f"trimmed_{idx:04d}.ts")
            clip_dur = _get_clip_duration(entry["clip_path"])
            max_offset = max(0.0, clip_dur - need_s)
            offset_s = random.uniform(0, max_offset) if max_offset > 0 else 0.0

            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", str(offset_s),
                    "-i", entry["clip_path"],
                    "-t", str(need_s),
                    "-an",
                    "-vf", (
                        f"scale={width}:{height}:"
                        "force_original_aspect_ratio=decrease,"
                        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
                        "fps=30"
                    ),
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-bsf:v", "h264_mp4toannexb",
                    "-f", "mpegts",
                    trimmed,
                ],
                capture_output=True, check=True,
            )
            segments.append(trimmed)
            cursor_ms = end_ms

        # Fill trailing gap to match audio duration
        audio_dur = _get_clip_duration(audio_path)
        if audio_dur > 0:
            trailing_ms = int(audio_dur * 1000) - cursor_ms
            if trailing_ms > 50:
                trail_path = os.path.join(tmpdir, "gap_trail.ts")
                _make_black_segment(trail_path, trailing_ms / 1000.0, width, height)
                segments.append(trail_path)

        # Concatenate all segments
        concat_input = "|".join(segments)
        concat_video = os.path.join(tmpdir, "concat.ts")

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", f"concat:{concat_input}",
                "-c", "copy",
                concat_video,
            ],
            capture_output=True, check=True,
        )

        # Mux video + audio into final MP4
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", concat_video,
                "-i", audio_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-shortest",
                "-movflags", "+faststart",
                output_path,
            ],
            capture_output=True, check=True,
        )
