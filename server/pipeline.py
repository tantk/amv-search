"""Beat-synced music video pipeline for the web server.

Based on generate/anime_video_v2.py — ported for serverless deployment:
- Embeddings via API (SiliconFlow/DashScope) instead of local GPU
- Rendi cloud rendering with local ffmpeg fallback
- Async-friendly with run_in_executor for blocking stages

Stages:
    1. Generate music via ElevenLabs API
    2. Detect beats + build beat-synced timeline with intensity scoring
    3. Search for matching video clips via turbopuffer
    4. Download clips in parallel
    5. Render via Rendi or return ffmpeg command
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import librosa
import numpy as np

from search import search_clips

# ── In-memory job store ──────────────────────────────────────────

_jobs: dict[str, "Job"] = {}

STAGE_NAMES = [
    "generating_music",
    "building_timeline",
    "searching_clips",
    "downloading_clips",
    "rendering_video",
]


@dataclass
class Job:
    job_id: str
    prompt: str
    mode: str = "anime"
    duration_ms: int = 30_000
    status: str = "pending"
    error: str | None = None
    stages: dict = field(default_factory=lambda: {s: "pending" for s in STAGE_NAMES})
    timeline: dict | None = None
    render_command: str | None = None
    render_assets: dict | None = None
    video_url: str | None = None
    title: str | None = None


def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)


# ── Helpers ──────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent


def _job_dir(job_id: str) -> Path:
    d = BASE_DIR / "jobs" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _set_stage(job: Job, stage: str, value: str) -> None:
    job.stages[stage] = value


HISTORY_DATASET = "tantk/amv-history"
HISTORY_FILE_URL = f"https://huggingface.co/datasets/{HISTORY_DATASET}/resolve/main/history.json"


def _load_history() -> list[dict]:
    """Fetch the current history from the HF dataset (public URL)."""
    try:
        import httpx as _httpx
        r = _httpx.get(HISTORY_FILE_URL, timeout=10, follow_redirects=True)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def _append_history(entry: dict) -> None:
    """Append a successful generation to the HF dataset `tantk/amv-history`.

    Uses huggingface_hub to upload `history.json` atomically.
    Persists across Space restarts since it lives in a separate repo.
    """
    try:
        from huggingface_hub import HfApi
        import tempfile

        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            print("[History] no HF_TOKEN — skipping")
            return

        entries = _load_history()
        entries.insert(0, entry)
        entries = entries[:100]  # cap

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(entries, f, indent=2)
            tmp_path = f.name

        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="history.json",
            repo_id=HISTORY_DATASET,
            repo_type="dataset",
            commit_message=f"add: {entry.get('title') or entry.get('job_id')}",
        )
        os.unlink(tmp_path)
        print(f"[History] appended {entry.get('title') or entry.get('job_id')}", flush=True)
    except Exception as e:
        print(f"[History] append failed: {e}", flush=True)


# ── Stage 1: Generate music ─────────────────────────────────────

def _stage_1_generate_music(job: Job) -> tuple[bytes, dict]:
    """Call ElevenLabs music composition API."""
    from elevenlabs import ElevenLabs

    _set_stage(job, "generating_music", "in_progress")

    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    response = client.music.compose_detailed(
        prompt=job.prompt,
        music_length_ms=job.duration_ms,
        with_timestamps=True,
    )

    audio_bytes: bytes | None = None
    metadata: dict = {}

    if hasattr(response, "audio") and hasattr(response, "json"):
        try:
            audio_bytes = response.audio
            meta_raw = response.json
            metadata = meta_raw if isinstance(meta_raw, dict) else json.loads(meta_raw)
        except Exception:
            pass

    if audio_bytes is None and isinstance(response, dict):
        audio_bytes = response.get("audio")
        metadata = response.get("json") or response.get("metadata") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

    if audio_bytes is None:
        try:
            parts = list(response)
            for part in parts:
                if isinstance(part, bytes):
                    audio_bytes = part
                elif isinstance(part, dict):
                    metadata = part
                elif hasattr(part, "audio"):
                    audio_bytes = part.audio
                    meta_raw = getattr(part, "json", {})
                    metadata = meta_raw if isinstance(meta_raw, dict) else json.loads(meta_raw)
        except TypeError:
            pass

    if audio_bytes is None:
        if isinstance(response, bytes):
            audio_bytes = response
        else:
            raise RuntimeError(f"Could not extract audio. Type: {type(response)}")

    audio_path = _job_dir(job.job_id) / "audio.mp3"
    audio_path.write_bytes(audio_bytes)

    # Extract song title from ElevenLabs response if present
    song_meta = metadata.get("song_metadata") or {}
    title = song_meta.get("title") or ""
    if title:
        job.title = title
        print(f"[Music] Title: {title}")

    # Persist full ElevenLabs metadata (composition_plan, words_timestamps,
    # song_metadata, etc.) for later re-use or debugging.
    metadata_path = _job_dir(job.job_id) / "metadata.json"
    try:
        metadata_path.write_text(json.dumps(metadata, indent=2))
    except (TypeError, OSError):
        pass  # Don't fail the pipeline over a metadata write

    _set_stage(job, "generating_music", "done")
    return audio_bytes, metadata


# ── Stage 2: Beat detection + timeline ───────────────────────────

def _get_section_for_time(sections: list[dict], time_ms: int) -> dict:
    cursor = 0
    for s in sections:
        end = cursor + s["duration_ms"]
        if time_ms < end:
            return s
        cursor = end
    return sections[-1] if sections else {}


def _local_normalize(values: np.ndarray, times: np.ndarray, window_sec: float = 6.0) -> np.ndarray:
    """Divide each value by the 70th percentile in a ±window_sec neighborhood.

    This local normalization fixes the 'piano onset in quiet intro' problem:
    a value loud in absolute terms isn't necessarily loud relative to its
    local neighborhood.
    """
    out = np.zeros_like(values, dtype=np.float64)
    for i, t in enumerate(times):
        mask = np.abs(times - t) <= window_sec
        local = values[mask]
        p70 = np.percentile(local, 70) if len(local) > 0 else 1.0
        out[i] = values[i] / max(p70, 1e-6)
    return out


def _foote_novelty_peaks(y: np.ndarray, sr: int, n_fft: int = 2048) -> list[float]:
    """Foote (2000) structural novelty → timestamps of major section boundaries.

    Computes a chroma self-similarity matrix, convolves a checkerboard kernel
    along the diagonal, then finds peaks in the novelty curve.
    """
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=512)
        ssm = librosa.segment.recurrence_matrix(chroma, mode="affinity", sym=True)

        # Checkerboard kernel
        N = min(32, ssm.shape[0] // 8) or 8
        ker = np.kron([[1, -1], [-1, 1]], np.ones((N, N)))
        ker = ker / np.abs(ker).sum()

        # Novelty = convolve kernel along diagonal
        novelty = np.zeros(ssm.shape[0])
        for i in range(N, ssm.shape[0] - N):
            novelty[i] = np.sum(ker * ssm[i - N:i + N, i - N:i + N])

        # Find peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(novelty, distance=sr * 2 // 512,  # ≥2s apart
                              prominence=np.std(novelty) * 0.5)
        times = librosa.frames_to_time(peaks, sr=sr, hop_length=512)
        return [float(t) for t in times]
    except Exception as e:
        print(f"[Foote] novelty failed: {e}, skipping boundaries", flush=True)
        return []


def _bpm_from_metadata(metadata: dict) -> float | None:
    """Extract BPM from ElevenLabs metadata.

    ElevenLabs embeds tempo as strings like "170bpm" or "140 bpm" inside
    `composition_plan.positive_global_styles` and `song_metadata.description`.
    This is authoritative (the model tells us what it generated) and
    sidesteps librosa's well-known octave error on sparse/harmonic music.
    """
    import re
    plan = metadata.get("composition_plan") or {}
    song = metadata.get("song_metadata") or {}
    haystacks: list[str] = []
    haystacks.extend(plan.get("positive_global_styles") or [])
    desc = song.get("description")
    if isinstance(desc, str):
        haystacks.append(desc)

    pat = re.compile(r"(\d{2,3})\s*bpm", re.IGNORECASE)
    for s in haystacks:
        if not isinstance(s, str):
            continue
        m = pat.search(s)
        if m:
            bpm = int(m.group(1))
            if 40 <= bpm <= 240:
                return float(bpm)
    return None


def _fill_sparse_beats(
    times: list[float], is_downbeat: list[bool], audio_dur_s: float,
) -> tuple[list[float], list[bool]]:
    """Fill sparse regions + extend grid to cover [0, audio_dur_s].

    librosa.beat_track has two failure modes:
      1. Misses beats in quiet/harmonic sections (e.g. soft pre-chorus)
      2. Loses track when tempo/dynamics shift dramatically — e.g.
         detects beats only 0-12s, nothing in the chorus 15-30s
    Without this, a section with a 1.4 cuts/sec target can produce
    0 cuts if the detector found no beats inside its window.
    """
    if len(times) < 2:
        return times, is_downbeat
    diffs = np.diff(times)
    median_gap = float(np.median(diffs))
    threshold = median_gap * 1.5

    # Fill interior gaps
    new_times = [times[0]]
    new_db = [is_downbeat[0]]
    for i in range(1, len(times)):
        gap = times[i] - times[i - 1]
        if gap > threshold:
            n_insert = int(round(gap / median_gap)) - 1
            for k in range(1, n_insert + 1):
                new_times.append(times[i - 1] + k * gap / (n_insert + 1))
                new_db.append(False)
        new_times.append(times[i])
        new_db.append(is_downbeat[i])

    # Extend backward to 0 if first beat is far in
    while new_times[0] > median_gap:
        new_times.insert(0, new_times[0] - median_gap)
        new_db.insert(0, False)

    # Extend forward to audio_dur_s (critical for songs where librosa
    # loses tracking in the chorus)
    while new_times[-1] < audio_dur_s - median_gap:
        new_times.append(new_times[-1] + median_gap)
        new_db.append(False)

    return new_times, new_db


def _detect_downbeats(audio_path: str) -> tuple[list[float], list[bool]]:
    """madmom RNN+DBN downbeat detection.

    Returns (beat_times, is_downbeat flags). Falls back to librosa if madmom
    fails to load or errors.
    """
    try:
        from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
        act = RNNDownBeatProcessor()(audio_path)
        dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
        data = dbn(act)  # (N, 2) — [time, beat_number_in_bar]
        beat_times = [float(t) for t in data[:, 0]]
        is_downbeat = [int(n) == 1 for n in data[:, 1]]
        return beat_times, is_downbeat
    except Exception as e:
        print(f"[Downbeats] madmom failed: {e}, using librosa beat_track", flush=True)
        y, sr = librosa.load(audio_path, sr=22050)
        _, bf = librosa.beat.beat_track(y=y, sr=sr)
        times = librosa.frames_to_time(bf, sr=sr).tolist()
        # No downbeat info — mark every 4th beat as downbeat
        return times, [i % 4 == 0 for i in range(len(times))]


def _stage_2_build_timeline(job: Job, metadata: dict) -> list[dict]:
    """Beat detection + adaptive-density cut selection.

    Pipeline:
      1. madmom downbeat detection (fallback: librosa beats)
      2. Multi-signal per-beat intensity:
         - Spectral flux (librosa onset_strength)
         - Short-term RMS loudness
         - HPCP harmonic change rate
         - All locally normalized (±6s window)
      3. Structural novelty (Foote checkerboard) → must-cut boundaries
      4. Vocal onsets from ElevenLabs words_timestamps (free — no Demucs)
      5. Adaptive cut density: target = 0.5 × mean_intensity × (tempo/120) cuts/sec
      6. Greedy selection:
         - Always: section boundaries + structural novelty peaks
         - Then: top-N beats by intensity (downbeats get +0.3 bonus)
         - Subject to 0.35s min gap, 2.5s max gap
    """
    _set_stage(job, "building_timeline", "in_progress")
    import threading

    audio_path = str(_job_dir(job.job_id) / "audio.mp3")

    # Launch madmom in parallel with librosa feature extraction (CPU bound,
    # but different processing paths — we gain ~1-2s on 2 vCPU)
    beat_times: list[float] = []
    is_downbeat: list[bool] = []

    def _run_downbeats():
        nonlocal beat_times, is_downbeat
        beat_times, is_downbeat = _detect_downbeats(audio_path)

    dbt_thread = threading.Thread(target=_run_downbeats)
    dbt_thread.start()

    # librosa features — load audio once at 22050Hz (madmom prefers 44100
    # but we already hand it the path, so it does its own load)
    y, sr = librosa.load(audio_path, sr=22050)
    audio_dur_s = len(y) / sr

    hop = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    # HPCP change rate — L2 norm of chroma time-derivative
    hpcp_change = np.pad(np.linalg.norm(np.diff(chroma, axis=1), axis=0), (1, 0))

    frame_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop)
    novelty_peaks = _foote_novelty_peaks(y, sr)

    dbt_thread.join()
    if not beat_times:
        # madmom returned nothing — emergency grid
        beat_times = list(np.linspace(0, audio_dur_s, max(8, int(audio_dur_s))))
        is_downbeat = [i % 4 == 0 for i in range(len(beat_times))]

    # Ensure beat 0 exists
    if beat_times[0] > 0.1:
        beat_times.insert(0, 0.0)
        is_downbeat.insert(0, True)

    # Fill sparse regions + extend grid to full duration — librosa often
    # misses beats in quiet sections OR loses tracking entirely after
    # dynamic shifts. Both leave snap-to-beat with empty sections.
    beat_times, is_downbeat = _fill_sparse_beats(beat_times, is_downbeat, audio_dur_s)

    # ── Per-beat intensity ──────────────────────────────────
    def _sample_at(values: np.ndarray, t: float) -> float:
        idx = int(np.clip(t * sr / hop, 0, len(values) - 1))
        # Average ±200ms around the beat to reduce noise
        lo = max(0, idx - int(0.2 * sr / hop))
        hi = min(len(values), idx + int(0.2 * sr / hop) + 1)
        return float(np.mean(values[lo:hi]))

    beat_flux = np.array([_sample_at(onset_env, t) for t in beat_times])
    beat_rms = np.array([_sample_at(rms, t) for t in beat_times])
    beat_hpcp = np.array([_sample_at(hpcp_change, t) for t in beat_times])
    bt_arr = np.array(beat_times)

    # Local normalization (fixes piano-in-intro problem)
    flux_n = _local_normalize(beat_flux, bt_arr)
    rms_n = _local_normalize(beat_rms, bt_arr)
    hpcp_n = _local_normalize(beat_hpcp, bt_arr)

    # Weighted intensity — spectral flux matters most for rhythmic cuts
    intensity = 0.5 * flux_n + 0.3 * rms_n + 0.2 * hpcp_n
    # Clip and scale to 0-1 using global max for stable speed-ramping decisions
    intensity = np.clip(intensity / max(intensity.max(), 1e-6), 0, 1)

    librosa_bpm = 60 / np.mean(np.diff(beat_times)) if len(beat_times) > 1 else 120
    meta_bpm = _bpm_from_metadata(metadata)
    avg_bpm = meta_bpm if meta_bpm is not None else librosa_bpm
    bpm_src = "metadata" if meta_bpm is not None else "librosa"
    mean_intensity = float(np.mean(intensity))
    print(f"[Beats] {len(beat_times)} beats, ~{avg_bpm:.0f} BPM ({bpm_src}; librosa={librosa_bpm:.0f}), mean intensity {mean_intensity:.2f}")

    # ── Vocal onsets from ElevenLabs ────────────────────────
    plan = metadata.get("composition_plan") or {}
    sections = plan.get("sections") or []
    words = metadata.get("words_timestamps") or []
    total_dur_ms = sum(s.get("duration_ms", 0) for s in sections) or job.duration_ms
    vocal_onsets = [w["start_ms"] / 1000.0 for w in words if (w.get("word") or "").strip()]
    print(f"[Timeline] {len(sections)} sections, {len(words)} words, {len(novelty_peaks)} novelty peaks")

    # ── Per-section target rates (AMV editor conventions) ──────
    # Section metadata is the PRIMARY density driver because transient-based
    # intensity signals (spectral flux / HPCP) mislead on piano ballads.
    # Within a section, local intensity decides WHICH beats to pick.
    # Base rates at 120BPM. Actual rate = base × (bpm/120)² so tempo
    # difference is amplified — a 170BPM song gets ~2× the cut density
    # of a 120BPM song, and ~4× the density of a 85BPM ballad. This
    # matches AMV editing convention: fast songs get aggressive pacing.
    SECTION_RATES = {
        "drop": 1.7, "climax": 1.7, "hook": 1.5, "refrain": 1.5,
        "chorus": 1.5,
        "pre-chorus": 1.3, "bridge": 1.2,
        "verse": 0.9, "post-chorus": 1.0,
        "intro": 0.4, "outro": 0.3, "interlude": 0.4,
    }
    DEFAULT_RATE = 0.8

    tempo_mult = float(np.clip((avg_bpm / 120.0) ** 2, 0.45, 2.5))
    # Upper cap: never exceed one cut per beat (physical limit — can't
    # cut faster than the music is hitting).
    max_rate = avg_bpm / 60.0

    def _rate_for_section(name: str) -> float:
        n = (name or "").lower()
        base = DEFAULT_RATE
        for key, rate in SECTION_RATES.items():
            if key in n:
                base = rate
                break
        return min(base * tempo_mult, max_rate)

    # Build a list of (section_start_s, section_end_s, name, target_count)
    section_plan = []
    cursor_s = 0.0
    for s in sections:
        dur_s = s.get("duration_ms", 0) / 1000.0
        name = s.get("section_name", "") or ""
        rate = _rate_for_section(name)
        target_n = max(1, int(round(rate * dur_s)))
        section_plan.append({
            "start_s": cursor_s,
            "end_s": cursor_s + dur_s,
            "name": name,
            "rate": rate,
            "target_n": target_n,
        })
        cursor_s += dur_s
    if not section_plan:
        # Unlabeled — treat whole song as one
        section_plan = [{
            "start_s": 0.0, "end_s": audio_dur_s,
            "name": "", "rate": DEFAULT_RATE,
            "target_n": max(1, int(round(DEFAULT_RATE * audio_dur_s))),
        }]

    for sp in section_plan:
        print(f"[Plan] [{sp['name']:<15}] {sp['end_s']-sp['start_s']:.1f}s target {sp['target_n']} cuts ({sp['rate']:.2f}/s)")

    # Score: local intensity + downbeat + vocal onset bonus.
    # Used to pick WHICH beat to snap to within each slot.
    scores = intensity.copy()
    for i, db in enumerate(is_downbeat):
        if db:
            scores[i] += 0.2
    for i, bt in enumerate(bt_arr):
        for vt in vocal_onsets:
            if abs(bt - vt) < 0.15:
                scores[i] += 0.25
                break

    # ── Per-section slotted selection ────────────────────────────
    # Divide each section into `target_n` equal-width slots. For each
    # slot, pick the best-scoring beat inside it. Empty slots fall
    # back to the nearest *unused* beat, then to highest-score unused,
    # so the target count is hit whenever enough beats exist.
    chosen_set: set[int] = set()
    for sp in section_plan:
        target_n = sp["target_n"]
        duration = max(sp["end_s"] - sp["start_s"], 1e-3)
        slot_w = duration / target_n

        sec_idxs = [i for i, bt in enumerate(bt_arr)
                    if sp["start_s"] <= bt < sp["end_s"]]
        if not sec_idxs:
            continue
        # First beat of section (boundary cut)
        chosen_set.add(min(sec_idxs, key=lambda i: bt_arr[i]))

        section_chosen: set[int] = set()
        for k in range(target_n):
            slot_start = sp["start_s"] + k * slot_w
            slot_end = slot_start + slot_w
            in_slot = [i for i in sec_idxs
                       if slot_start <= bt_arr[i] < slot_end
                       and i not in section_chosen]
            if in_slot:
                best = max(in_slot, key=lambda i: scores[i])
            else:
                # Empty slot — pick nearest UNUSED beat in section
                unused = [i for i in sec_idxs if i not in section_chosen]
                if not unused:
                    break
                center = slot_start + slot_w / 2
                best = min(unused, key=lambda i: abs(bt_arr[i] - center))
            section_chosen.add(best)

        chosen_set.update(section_chosen)

    chosen_idxs = sorted(chosen_set, key=lambda i: bt_arr[i])
    print(f"[Beats] selected {len(chosen_idxs)}/{len(beat_times)} cuts across {len(section_plan)} sections")

    # Build timeline entries from chosen_idxs only (already filtered by greedy).
    timeline = []
    for k, idx in enumerate(chosen_idxs):
        beat_ms = int(bt_arr[idx] * 1000)
        if beat_ms >= total_dur_ms:
            break
        # end = next chosen beat's start (or song end)
        next_idx = chosen_idxs[k + 1] if k + 1 < len(chosen_idxs) else None
        end_ms = int(bt_arr[next_idx] * 1000) if next_idx is not None else total_dur_ms

        section = _get_section_for_time(sections, beat_ms)
        section_name = section.get("section_name", "") or ""
        styles_list = section.get("positive_local_styles") or []
        styles = " ".join(styles_list[:2])

        beat_intensity = float(intensity[idx])

        # Speed ramping
        if beat_intensity < 0.3:
            speed = 0.7
        elif beat_intensity > 0.7:
            speed = 1.3
        else:
            speed = 1.0

        # Context from nearby words (within ±2s)
        nearby_words = [w["word"] for w in words if abs(w["start_ms"] - beat_ms) < 2000]
        lyric_context = " ".join(nearby_words[:6]) if nearby_words else ""

        # Current line from section
        lines = section.get("lines") or []
        line_idx = 0
        if lines:
            line_dur = section.get("duration_ms", 10000) // len(lines)
            sec_offset = beat_ms - sum(s.get("duration_ms", 0) for s in sections[:sections.index(section)] if sections and section in sections)
            line_idx = min(max(sec_offset, 0) // max(line_dur, 1), len(lines) - 1)
        current_line = lines[line_idx] if lines else ""

        # Mood hints from intensity
        if beat_intensity > 0.7:
            mood = "intense action dramatic explosive"
        elif beat_intensity < 0.3:
            mood = "calm peaceful quiet contemplative"
        else:
            mood = ""

        search_query = f"{current_line} {lyric_context} {styles} {mood}".strip()
        if not search_query:
            search_query = f"{section_name} {styles} {mood}".strip() or "cinematic"

        timeline.append({
            "start_ms": beat_ms,
            "end_ms": end_ms,
            "section": section_name,
            "lyric": current_line or f"[{section_name}]",
            "search_query": search_query,
            "intensity": round(beat_intensity, 2),
            "speed": speed,
            "is_downbeat": is_downbeat[idx] if idx < len(is_downbeat) else False,
        })

    print(f"[Timeline] {len(timeline)} final segments")
    _set_stage(job, "building_timeline", "done")
    return timeline


# ── Stage 3: Search clips ────────────────────────────────────────

def _stage_3_search_clips(job: Job, timeline: list[dict]) -> list[dict]:
    """Search turbopuffer for clips matching each beat segment. Parallel queries."""
    from concurrent.futures import ThreadPoolExecutor
    _set_stage(job, "searching_clips", "in_progress")

    namespace = "anime-clips" if job.mode == "anime" else "music-video-clips"

    timeline.sort(key=lambda e: e["start_ms"])

    # Unique queries — run these in parallel via thread pool
    unique_queries = list({e["search_query"] for e in timeline})
    print(f"[Search] {len(unique_queries)} unique queries ({len(timeline)} beats)", flush=True)

    def _one_query(q: str):
        try:
            return q, search_clips(q, top_k=20, used_video_ids=set(), namespace=namespace)
        except Exception as e:
            print(f"[Search] error for '{q[:40]}': {e}", flush=True)
            return q, []

    query_cache: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for q, results in pool.map(_one_query, unique_queries):
            query_cache[q] = results

    used_ids: set[str] = set()
    last_chosen: dict | None = None
    # Build a pool of all results across all queries as a last-resort fallback
    all_results = []
    for rs in query_cache.values():
        all_results.extend(rs)

    for i, entry in enumerate(timeline):
        query = entry["search_query"]
        # Pick first unused clip from this query's results
        chosen = None
        for r in query_cache.get(query, []):
            vid = r.get("video_id", r.get("id", ""))
            if vid not in used_ids:
                chosen = r
                break

        # Fallback 1: borrow an unused clip from ANY query's results
        # (avoids LRU cycling which would cause duplicate clips)
        if chosen is None:
            for r in all_results:
                vid = r.get("video_id", r.get("id", ""))
                if vid not in used_ids:
                    chosen = r
                    break

        # Fallback 2: all clips used — LRU cycle within own query (may duplicate)
        if chosen is None and query_cache.get(query):
            chosen = query_cache[query][i % len(query_cache[query])]

        # Fallback 3: empty query — reuse previous beat's clip
        if chosen is None and last_chosen is not None:
            chosen = last_chosen

        # Fallback 4: last resort — any result
        if chosen is None and all_results:
            chosen = all_results[i % len(all_results)]

        if chosen:
            entry["video_id"] = chosen.get("video_id", "")
            entry["source"] = chosen.get("source", "")
            entry["category"] = chosen.get("category", "")
            entry["caption"] = chosen.get("caption", "")
            entry["dataset_path"] = chosen.get("path", "")
            used_ids.add(chosen.get("video_id", chosen.get("id", "")))
            last_chosen = chosen

            # Stash backup candidates for this entry — stage 4 tries these
            # if the primary clip fails to download (common for stale
            # Pexels/Pixabay IDs that no longer resolve).
            backups = []
            seen_bk = {chosen.get("video_id", "")}
            # First: more from the beat's own query
            for r in query_cache.get(query, []):
                vid = r.get("video_id", r.get("id", ""))
                if vid and vid not in seen_bk and vid not in used_ids:
                    backups.append({
                        "video_id": vid,
                        "source": r.get("source", ""),
                        "path": r.get("path", ""),
                    })
                    seen_bk.add(vid)
                    if len(backups) >= 5:
                        break
            # Then: any from the global pool
            if len(backups) < 5:
                for r in all_results:
                    vid = r.get("video_id", r.get("id", ""))
                    if vid and vid not in seen_bk and vid not in used_ids:
                        backups.append({
                            "video_id": vid,
                            "source": r.get("source", ""),
                            "path": r.get("path", ""),
                        })
                        seen_bk.add(vid)
                        if len(backups) >= 5:
                            break
            entry["backups"] = backups
        else:
            print(f"[Search] No clips for: {query[:50]}")

    matched = sum(1 for e in timeline if e.get("video_id"))
    print(f"[Search] {matched}/{len(timeline)} beats matched ({len(query_cache)} unique queries)")
    if matched == 0:
        raise RuntimeError("Search found zero matching clips.")

    # Persist timeline — includes everything needed to reconstruct the video:
    # the job's mode/prompt, every beat's clip selection, search query,
    # source metadata, intensity/speed for rendering.
    job.timeline = {
        "job_id": job.job_id,
        "prompt": job.prompt,
        "mode": job.mode,
        "title": job.title,
        "audio_url": f"/static/jobs/{job.job_id}/audio.mp3",
        "duration_ms": job.duration_ms,
        "clips": timeline,
    }
    timeline_path = _job_dir(job.job_id) / "timeline.json"
    timeline_path.write_text(json.dumps(job.timeline, indent=2))

    _set_stage(job, "searching_clips", "done")
    return timeline


# ── Stage 4: Download clips ──────────────────────────────────────

def _download_clip(video_id: str, source: str, output_path: str,
                   dataset_path: str = "", fallback_query: str = "") -> bool:
    """Download a single clip.

    - pixabay/pexels: look up URL via their APIs
    - http(s) in dataset_path: try local file first (Space hosts anime clips)
    - local path in dataset_path: copy the file
    """
    # If dataset_path is already a URL, first check if it's one of our own
    # static clips already on disk (no network needed — just symlink/copy).
    if dataset_path.startswith(("http://", "https://")):
        # Parse URL to see if it points to a local static file
        # e.g. https://tantk-music-video-gen.hf.space/static/anime_clips/aot/clip_0.mp4
        #   → /app/static/anime_clips/aot/clip_0.mp4
        try:
            from urllib.parse import urlparse
            parsed = urlparse(dataset_path)
            if "/static/" in parsed.path:
                # BASE_DIR is /app when running in Docker
                relative = parsed.path.split("/static/", 1)[1]
                local_path = BASE_DIR / "static" / relative
                if local_path.exists():
                    try:
                        if os.path.exists(output_path):
                            os.remove(output_path)
                        os.symlink(str(local_path), output_path)
                        return True
                    except OSError:
                        # Fallback to copy if symlink fails
                        import shutil
                        shutil.copy2(str(local_path), output_path)
                        return True
        except Exception:
            pass

        # Not a local file — stream from HTTP
        try:
            with httpx.stream("GET", dataset_path, timeout=60, follow_redirects=True) as resp:
                resp.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        f.write(chunk)
            return True
        except (httpx.HTTPError, OSError):
            return False

    # Pixabay/Pexels: look up URL via API
    if source in ("pixabay", "pexels"):
        from download import get_download_url
        url = get_download_url(video_id, source, fallback_query)
        if not url:
            return False
        try:
            with httpx.stream("GET", url, timeout=60, follow_redirects=True) as resp:
                resp.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        f.write(chunk)
            return True
        except (httpx.HTTPError, OSError):
            return False

    # Local file (indexing machine)
    if dataset_path and os.path.exists(dataset_path):
        import shutil
        shutil.copy2(dataset_path, output_path)
        return True

    return False


async def _stage_4_download_clips(job: Job, timeline: list[dict]) -> None:
    """Download clips in parallel. As each clip arrives, kick off its
    normalize subprocess immediately (pipeline download + normalize).

    The normalize tasks are stashed on the job for stage 5 to await.
    """
    _set_stage(job, "downloading_clips", "in_progress")

    clips_dir = _job_dir(job.job_id) / "clips"
    segs_dir = _job_dir(job.job_id) / "segs"
    clips_dir.mkdir(exist_ok=True)
    segs_dir.mkdir(exist_ok=True)

    loop = asyncio.get_running_loop()
    dl_sem = asyncio.Semaphore(8)   # 8 concurrent downloads
    norm_sem = asyncio.Semaphore(2)  # 2 concurrent normalizes (2 vCPU)

    # Track all clip IDs successfully downloaded, so parallel tasks don't
    # independently pick the same backup ID for two different beats.
    taken_ids: set[str] = set()
    taken_lock = asyncio.Lock()

    width, height, fps = 1280, 720, 30

    # Normalize tasks keyed by timeline index (list of (i, future))
    norm_tasks: list[tuple[int, "asyncio.Task"]] = []

    async def _normalize(i: int, entry: dict) -> tuple[int, str | None]:
        """Run normalize for one clip, gated by semaphore."""
        import random
        need_s = (entry["end_ms"] - entry["start_ms"]) / 1000.0
        speed = entry.get("speed", 1.0)
        source_need = need_s * speed
        offset_s = round(random.uniform(0, 0.5), 2)
        seg_path = str(segs_dir / f"seg_{i:04d}.ts")

        async with norm_sem:
            ok = await loop.run_in_executor(
                None, _normalize_clip,
                entry["clip_path"], seg_path,
                offset_s, source_need, speed,
                width, height, fps,
            )
        return i, (seg_path if ok else None)

    async def _dl_and_kickoff(i: int, entry: dict) -> tuple[int, bool]:
        vid = entry.get("video_id", "")
        if not vid:
            return i, False
        path = str(clips_dir / f"clip_{i}.mp4")
        async with dl_sem:
            # Build ordered candidate list (primary + backups)
            candidates = [
                (vid, entry.get("source", ""), entry.get("dataset_path", "")),
            ] + [
                (b["video_id"], b.get("source", ""), b.get("path", ""))
                for b in entry.get("backups", [])
            ]

            ok = False
            for try_vid, try_src, try_path in candidates:
                if not try_vid:
                    continue
                # Skip if another beat is already using this clip
                async with taken_lock:
                    if try_vid in taken_ids:
                        continue
                    # Tentatively claim the ID
                    taken_ids.add(try_vid)

                ok = await loop.run_in_executor(
                    None, _download_clip, try_vid, try_src,
                    path, try_path, entry.get("search_query", ""),
                )
                if ok:
                    if try_vid != vid:
                        print(f"[Download] beat {i} primary {vid} failed → backup {try_vid}")
                        entry["video_id"] = try_vid
                        entry["source"] = try_src
                        entry["dataset_path"] = try_path
                    break
                else:
                    # Release the claim on failure so another beat can try it
                    async with taken_lock:
                        taken_ids.discard(try_vid)
        if ok:
            entry["clip_path"] = path
            entry["video_url"] = f"/static/jobs/{job.job_id}/clips/clip_{i}.mp4"
            # Pipeline: kick off normalize as soon as download finishes
            norm_tasks.append((i, asyncio.create_task(_normalize(i, entry))))
        return i, ok

    results = await asyncio.gather(*[_dl_and_kickoff(i, e) for i, e in enumerate(timeline)])
    downloaded = sum(1 for _, ok in results if ok)

    # Stash normalize tasks on the job so stage 5 can await them
    job.timeline["_norm_tasks"] = norm_tasks  # type: ignore
    job.timeline["_segs_dir"] = str(segs_dir)  # type: ignore

    if job.timeline:
        # Don't include internal _norm_tasks in persisted JSON
        persist = {k: v for k, v in job.timeline.items() if not k.startswith("_")}
        persist["clips"] = timeline
        timeline_path = _job_dir(job.job_id) / "timeline.json"
        timeline_path.write_text(json.dumps(persist, indent=2))

    print(f"[Download] {downloaded}/{len(timeline)} clips downloaded, {len(norm_tasks)} normalize tasks in flight")
    _set_stage(job, "downloading_clips", "done")


# ── Stage 5: Render ──────────────────────────────────────────────

def _probe_clip(clip_path: str) -> tuple[int, int, float] | None:
    """Return (width, height, fps) of a clip, or None on failure."""
    import subprocess
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-select_streams", "v:0", clip_path],
            capture_output=True, check=True, timeout=5,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        w = int(stream.get("width", 0))
        h = int(stream.get("height", 0))
        fr = stream.get("r_frame_rate", "0/1").split("/")
        fps = float(fr[0]) / float(fr[1]) if len(fr) == 2 and float(fr[1]) > 0 else 0
        return (w, h, fps)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            KeyError, IndexError, ValueError, json.JSONDecodeError):
        return None


def _normalize_clip(
    clip_path: str, seg_path: str, offset_s: float, duration_s: float,
    speed: float, width: int, height: int, fps: int,
) -> bool:
    """Normalize one clip to MPEG-TS. Skips scale/crop if source already matches.

    Uses scale-and-crop (not pad) to fill the frame edge-to-edge — slightly
    faster than pad and no black bars.
    """
    import subprocess

    info = _probe_clip(clip_path)
    vf_parts = []

    if info:
        src_w, src_h, src_fps = info
        # Skip scale/crop if dimensions already match
        if (src_w, src_h) != (width, height):
            # Scale-to-fill then crop: fills the frame edge-to-edge
            # (faster than pad, no black bars to encode)
            vf_parts.append(
                f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height}"
            )
        # Skip fps filter if already close to target
        if src_fps < fps - 1 or src_fps > fps + 1:
            vf_parts.append(f"fps={fps}")
    else:
        # Probe failed — apply full normalization to be safe
        vf_parts.append(
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
        )
        vf_parts.append(f"fps={fps}")

    if speed != 1.0:
        vf_parts.append(f"setpts=PTS/{speed}")

    # -stream_loop -1 loops the input forever; -t caps the output duration.
    # This covers the case where source clip is shorter than duration_s
    # (e.g. 15s intro covered by a 3s Pixabay source — just loops the clip).
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-ss", str(offset_s),
        "-t", f"{duration_s:.3f}",
        "-an",
        "-i", clip_path,
    ]
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]

    cmd += [
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "fastdecode",
        "-crf", "26",
        "-pix_fmt", "yuv420p",
        "-bsf:v", "h264_mp4toannexb",
        "-f", "mpegts",
        seg_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        return os.path.exists(seg_path) and os.path.getsize(seg_path) > 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


async def _stage_5_render(job: Job, timeline: list[dict]) -> None:
    """Hybrid render: normalize on HF Space (parallel), concat via Rendi.

    Flow:
      1. On the Space: normalize each clip to MPEG-TS in parallel (2 workers)
         — handles any number of clips, no 60s limit
      2. Via Rendi: concat demuxer with -c copy + mux audio (~2-5s)
         — byte-level stitch, no re-encoding

    Fallback: if Rendi fails, user gets the ffmpeg command to run locally.
    """
    import random
    import asyncio as _aio
    _set_stage(job, "rendering_video", "in_progress")

    clips = [e for e in timeline if e.get("clip_path")]
    if not clips:
        raise RuntimeError("No clips to render")

    width, height, fps = 1280, 720, 30
    base_url = os.environ.get("HF_SPACE_URL", "https://tantk-music-video-gen.hf.space")
    segs_dir = _job_dir(job.job_id) / "segs"
    segs_dir.mkdir(exist_ok=True)

    # Phase 1 already running — stage 4 kicked off normalize tasks per clip
    # as soon as each download finished. Just await the remaining ones.
    t0 = time.time()
    norm_tasks = job.timeline.get("_norm_tasks", []) if job.timeline else []

    if norm_tasks:
        print(f"[Render] Awaiting {len(norm_tasks)} normalize tasks (started in stage 4)...", flush=True)
        results = await _aio.gather(*[t for _, t in norm_tasks])
        seg_paths = [path for (_, path) in sorted(results, key=lambda x: x[0]) if path]
    else:
        seg_paths = []

    t1 = time.time()
    print(f"[Render] Phase 1 done: {len(seg_paths)} segments (pipelined with download, +{t1-t0:.1f}s wait)")

    if not seg_paths:
        raise RuntimeError("No segments normalized successfully")

    # ── Store asset info for frontend fallback ────────────────────
    job.render_assets = {
        "audio": f"/static/jobs/{job.job_id}/audio.mp3",
        "clips": [
            {
                "filename": os.path.basename(c["clip_path"]),
                "url": f"/static/jobs/{job.job_id}/clips/{os.path.basename(c['clip_path'])}",
                "start_ms": c["start_ms"],
                "end_ms": c["end_ms"],
                "intensity": c.get("intensity", 0.5),
                "speed": c.get("speed", 1.0),
            }
            for c in clips
        ],
    }

    # Local fallback command: concat the already-normalized segments + mux audio
    concat_str = "|".join(os.path.basename(p) for p in seg_paths)
    job.render_command = (
        f'ffmpeg -y -i "concat:{concat_str}" -i audio.mp3 '
        f"-c:v copy -c:a copy -shortest -movflags +faststart output.mp4"
    )

    # ── Phase 2: Rendi concats the pre-normalized segments ───────
    rendi_key = os.environ.get("RENDI_API_KEY", "")
    if not rendi_key:
        print("[Render] No RENDI_API_KEY, returning local command")
        _set_stage(job, "rendering_video", "done")
        return

    try:
        # Rendi blocks the `concat:` protocol. Instead, we concatenate the
        # MPEG-TS segments ourselves into one big .ts on the Space, then
        # send just the merged .ts + audio to Rendi for final muxing.
        merged_ts = _job_dir(job.job_id) / "merged.ts"
        with open(merged_ts, "wb") as out:
            for p in seg_paths:
                with open(p, "rb") as f:
                    while chunk := f.read(65536):
                        out.write(chunk)

        input_files = {
            "in_video": f"{base_url}/static/jobs/{job.job_id}/merged.ts",
            "in_audio": f"{base_url}/static/jobs/{job.job_id}/audio.mp3",
        }
        rendi_cmd = (
            "-i {{in_video}} -i {{in_audio}} "
            "-c:v copy -c:a copy -shortest -movflags +faststart {{out_1}}"
        )

        t2 = time.time()
        print(f"[Render] Phase 2: Rendi concat of {len(seg_paths)} segs...", flush=True)
        resp = httpx.post(
            "https://api.rendi.dev/v1/run-ffmpeg-command",
            headers={"X-API-KEY": rendi_key, "Content-Type": "application/json"},
            json={
                "input_files": input_files,
                "output_files": {"out_1": "video.mp4"},
                "ffmpeg_command": rendi_cmd,
            },
            timeout=30,
        )

        if resp.status_code == 200:
            cmd_id = resp.json().get("command_id")
            for _ in range(40):
                await _aio.sleep(2)
                r = httpx.get(
                    f"https://api.rendi.dev/v1/commands/{cmd_id}",
                    headers={"X-API-KEY": rendi_key},
                    timeout=10,
                )
                d = r.json()
                status = d.get("status")
                if status == "SUCCESS":
                    video_url = d.get("output_files", {}).get("out_1", {}).get("storage_url", "")
                    if video_url:
                        job.video_url = video_url
                        print(f"[Render] Phase 2 done in {time.time()-t2:.1f}s: {video_url}", flush=True)
                        _append_history({
                            "job_id": job.job_id,
                            "title": job.title or "",
                            "prompt": job.prompt,
                            "mode": job.mode,
                            "video_url": video_url,
                            "created_at": int(time.time()),
                        })
                    break
                elif status in ("FAILED", "ERROR"):
                    print(f"[Render] Rendi FAILED after {time.time()-t2:.1f}s: {d.get('error_message', '')[:300]}", flush=True)
                    break
        else:
            print(f"[Render] Rendi submit failed: {resp.status_code} {resp.text[:200]}", flush=True)
    except Exception as e:
        print(f"[Render] Rendi error: {e}", flush=True)

    _set_stage(job, "rendering_video", "done")


# ── Main entry point ─────────────────────────────────────────────

async def run_pipeline_from_audio(
    job_id: str,
    prompt: str = "",
    mode: str = "anime",
    metadata: dict | None = None,
    duration_ms: int = 30_000,
) -> Job:
    """Run stages 2-5 using an existing audio.mp3 + metadata.json already on disk.

    Used by the /regenerate endpoint. Skips ElevenLabs (saves ~9s).
    """
    job = Job(job_id=job_id, prompt=prompt, mode=mode, duration_ms=duration_ms)
    _jobs[job_id] = job
    job.status = "running"
    # Stage 1 is effectively pre-done
    _set_stage(job, "generating_music", "done")

    if metadata is None:
        metadata_path = _job_dir(job_id) / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text())
        except Exception:
            metadata = {}

    # Extract title if present
    song_meta = metadata.get("song_metadata") or {}
    job.title = song_meta.get("title") or ""

    loop = asyncio.get_running_loop()

    try:
        import time as _t
        t0 = _t.time()
        timeline = await loop.run_in_executor(None, _stage_2_build_timeline, job, metadata)
        print(f"[Timing] Stage 2 (beats): {_t.time()-t0:.1f}s", flush=True)

        t0 = _t.time()
        timeline = await loop.run_in_executor(None, _stage_3_search_clips, job, timeline)
        print(f"[Timing] Stage 3 (search): {_t.time()-t0:.1f}s", flush=True)

        t0 = _t.time()
        await _stage_4_download_clips(job, timeline)
        print(f"[Timing] Stage 4 (download): {_t.time()-t0:.1f}s", flush=True)

        t0 = _t.time()
        await _stage_5_render(job, timeline)
        print(f"[Timing] Stage 5 (render): {_t.time()-t0:.1f}s", flush=True)

        job.status = "done"
    except Exception as e:
        import traceback
        traceback.print_exc()
        job.status = "failed"
        job.error = f"{type(e).__name__}: {e}"

    return job


async def run_pipeline(
    job_id: str | None = None,
    prompt: str = "",
    duration_ms: int = 30_000,
    mode: str = "anime",
) -> Job:
    if job_id is None:
        job_id = uuid.uuid4().hex[:12]

    job = Job(job_id=job_id, prompt=prompt, mode=mode, duration_ms=duration_ms)
    _jobs[job_id] = job
    job.status = "running"

    loop = asyncio.get_running_loop()

    try:
        # Stage 1 — generate music
        _audio_bytes, metadata = await loop.run_in_executor(
            None, _stage_1_generate_music, job
        )

        import time as _t
        t0 = _t.time()
        # Stage 2 — beat detection + timeline
        timeline = await loop.run_in_executor(
            None, _stage_2_build_timeline, job, metadata
        )
        print(f"[Timing] Stage 2 (beats): {_t.time()-t0:.1f}s", flush=True)

        t0 = _t.time()
        # Stage 3 — search clips
        timeline = await loop.run_in_executor(
            None, _stage_3_search_clips, job, timeline
        )
        print(f"[Timing] Stage 3 (search): {_t.time()-t0:.1f}s", flush=True)

        t0 = _t.time()
        # Stage 4 — download clips
        await _stage_4_download_clips(job, timeline)
        print(f"[Timing] Stage 4 (download): {_t.time()-t0:.1f}s", flush=True)

        t0 = _t.time()
        # Stage 5 — render (async, awaits in-flight normalize tasks from stage 4)
        await _stage_5_render(job, timeline)
        print(f"[Timing] Stage 5 (render): {_t.time()-t0:.1f}s", flush=True)

        job.status = "done"

    except Exception as e:
        import traceback
        traceback.print_exc()
        job.status = "failed"
        job.error = f"{type(e).__name__}: {e}"

    return job
