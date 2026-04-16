import os
import tempfile
import subprocess
import pytest


def _make_test_clip(path, duration=3.0, width=640, height=360):
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c=blue:s={width}x{height}:d={duration}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-t", str(duration), path,
    ], capture_output=True, check=True)


def _make_test_audio(path, duration=10.0):
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"anullsrc=r=44100:cl=stereo",
        "-t", str(duration), "-c:a", "aac", path,
    ], capture_output=True, check=True)


def test_render_video_from_timeline():
    from render import render_video
    with tempfile.TemporaryDirectory() as tmpdir:
        clip_paths = []
        for i in range(3):
            p = os.path.join(tmpdir, f"clip_{i}.mp4")
            _make_test_clip(p, duration=4.0)
            clip_paths.append(p)
        audio_path = os.path.join(tmpdir, "audio.m4a")
        _make_test_audio(audio_path, duration=10.0)
        timeline = [
            {"start_ms": 0, "end_ms": 3000, "clip_path": clip_paths[0]},
            {"start_ms": 3000, "end_ms": 7000, "clip_path": clip_paths[1]},
            {"start_ms": 7000, "end_ms": 10000, "clip_path": clip_paths[2]},
        ]
        output = os.path.join(tmpdir, "output.mp4")
        render_video(timeline, audio_path, output, width=640, height=360)
        assert os.path.exists(output)
        assert os.path.getsize(output) > 1000


def test_render_video_empty_timeline_raises():
    from render import render_video
    with pytest.raises(ValueError, match="timeline must not be empty"):
        render_video([], "audio.m4a", "output.mp4")
