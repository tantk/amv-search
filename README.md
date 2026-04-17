# AMV.Search

**Prompt → music video in ~60s.** Type a text prompt, get back a fully beat-synced music video with AI-generated vocals + lyrics and search-matched footage — either anime clips (AMV mode) or stock footage (MV mode).

- **Live demo:** https://static-mu-liard.vercel.app
- **Demo video:** https://www.youtube.com/watch?v=k597okobXsc&t=18s

## Overview

AMV.Search turns a single text prompt into a finished music video by chaining specialized AI services: ElevenLabs writes the song, turbopuffer finds the best clips for each beat, and a lightweight cloud renderer stitches them into a beat-synced cut. Cuts per second, speed ramps, and clip choice are all derived from the song's actual tempo and section structure — slow piano intros get sparse cuts, explosive choruses get rapid ones, all automatically.

Two modes share the same pipeline:
- **AMV mode** — 1,802 pre-indexed anime clips served from HF Space static storage
- **MV mode** — 434K stock clips from Pixabay + Pexels, downloaded on demand

## Tech Stack

| Layer | Stack |
|---|---|
| Music generation | ElevenLabs `compose_detailed` (song + lyrics + word-level timestamps) |
| Audio analysis | librosa (beat tracking, onset strength, chroma, HPCP) |
| Text embedding | DashScope `text-embedding-v4` (2048-dim, anime) / SiliconFlow `Qwen3-Embedding-0.6B` (1024-dim, stock) |
| Vector search | turbopuffer (ANN vectors + BM25 hybrid via `multi_query` + RRF fusion) |
| Backend | FastAPI + asyncio on HF Spaces (Docker, 2 vCPU, free tier) |
| Frontend | Static HTML/CSS/JS on Vercel |
| Video rendering | ffmpeg on HF Space (parallel normalize) + Rendi (mux + CDN) |
| History storage | HF Datasets (`tantk/amv-history`) |
| Stock clip APIs | Pixabay + Pexels |

**Total infrastructure cost: minimal** — ElevenLabs (song generation) and turbopuffer (vector search) are paid services, but both sponsors provided hackathon credits. Every other service runs on a free tier.

## Pipeline at a Glance

```
Prompt → ElevenLabs (song + lyrics + word timestamps)
      ↓
       librosa (beat detection + intensity scoring + section pacing)
      ↓
       DashScope/SiliconFlow (text embedding)
      ↓
       turbopuffer (vector search + BM25 hybrid → RRF fusion)
      ↓
       HF Space (parallel ffmpeg normalize → merged.ts)
      ↓
       Rendi (concat + mux → CDN mp4)
```

5 async stages, total ~50–60s end-to-end.

---

## How It Works (Detailed)

1. **Generate song** — ElevenLabs composes a 30s song with word-level lyric timestamps from a text prompt. BPM is parsed from the metadata's `positive_global_styles` (authoritative) and falls back to librosa detection.
2. **Beat detection + pacing plan** — librosa finds rhythmic hits; sparse regions are filled with interpolated beats and extended to the song end. Per-section cut rate = `base_rate × (bpm/120)²`, with `base_rate` set per section name (0.4 for intro, 1.5 for chorus, etc.). This makes a 170 BPM song get ~4× the cut density of an 85 BPM ballad.
3. **Slot-based beat selection** — each section is divided into `target_n` equal-width slots; the highest-scoring beat in each slot wins (score = local onset intensity + downbeat + vocal-onset bonus). Empty slots fall back to the nearest unused beat.
4. **Hybrid search** — each beat's lyric + section style + intensity becomes a search query. Vector ANN (2048-dim anime / 1024-dim stock) + BM25 over captions in turbopuffer, fused via Reciprocal Rank Fusion. Run 8-way in parallel across unique queries.
5. **Intensity-driven speed ramping** — slow-mo (0.7×) for quiet beats, speed-up (1.3×) for peak intensity beats.
6. **Pipelined render** — as each clip downloads, an ffmpeg subprocess immediately normalizes it to MPEG-TS. 2 parallel normalizers match the 2 vCPU on the HF Space. Finished segments are byte-concatenated into one `merged.ts`.
7. **Final mux** — Rendi's cloud ffmpeg takes `merged.ts` + `audio.mp3` and produces a final mp4 via `-c copy` (pure mux, no re-encoding). Finishes in ~3–5s regardless of clip count.

## Architecture

### Frontend (Vercel)

Static HTML/CSS/JS. Two themes toggled client-side via CSS variables:
- **Anime mode** (dark + acid green, Outfit/Space Mono)
- **Normal mode** (warm cream + terracotta, Syne/DM Sans)

Terminal-style generation log at the bottom reads from the HF dataset history.

### Backend (HF Space, Docker, free tier)

FastAPI app with 5-stage async pipeline. 2 vCPU, ffmpeg + librosa baked into the container. Ephemeral per-job storage under `jobs/{id}/` — audio, metadata, clips, segments, merged video.

All API keys live in HF Space secrets (never in code).

### Clip library

- **Anime clips** (1,802 clips, ~1.8GB) — committed to the HF Space's git LFS at `static/anime_clips/`. Categorized by episode (e.g. `aot_s3e17/`). Served from the Space's own static URL.
- **Stock clips** (434K) — downloaded on demand from Pixabay/Pexels via their APIs. We only store the returned clip locally during a job.

### Generation history

Persisted to a separate HF dataset repo: [huggingface.co/datasets/tantk/amv-history](https://huggingface.co/datasets/tantk/amv-history). Each successful render appends `{job_id, title, prompt, mode, video_url, created_at}` to `history.json`.

Survives Space restarts. Capped at 100 entries.

### Video rendering

Two-phase split to stay under Rendi's 60s free-tier limit:
- **Phase 1 (HF Space, async)** — as each clip downloads, immediately spawn an ffmpeg subprocess to normalize it (scale+crop to 1280x720, re-encode to MPEG-TS with ultrafast preset). 2 parallel normalizers (matching 2 vCPU).
- **Phase 2 (Rendi, free tier)** — pre-merged `.ts` bytes + audio.mp3 → final mp4 via `-c copy -c:a copy`. Pure mux, no re-encoding, finishes in ~3-5s regardless of clip count.

### Embedding providers

| Mode | Model | Dimensions | Provider |
|---|---|---|---|
| Anime | text-embedding-v4 | 2048 | DashScope (free tier) |
| Normal | Qwen3-Embedding-0.6B | 1024 | SiliconFlow (free tier, same weights as the indexed vectors) |

Both are proven compatible with the vectors already in turbopuffer (verified empirically).

## Project Structure

```
├── server/                    # Web backend (deployed to HF Space)
│   ├── app.py                 # FastAPI — routes, CORS, health, history, regenerate
│   ├── pipeline.py            # 5-stage async pipeline
│   ├── search.py              # DashScope/SiliconFlow → turbopuffer
│   ├── download.py            # Pixabay/Pexels download helpers
│   ├── requirements.txt       # Server deps (no torch/transformers)
│   └── static/                # Frontend (copied to Vercel too)
│       ├── index.html         # Two-theme UI with toggle + history log
│       └── player.js          # Polling, state machine, render display
│
├── indexing/                  # One-time: index anime clips (requires GPU)
│   ├── anime_embed.py         # YouTube → split → embed → turbopuffer
│   ├── caption_clips.py       # Caption frames → BM25 in turbopuffer
│   ├── batch.json             # Video URLs and time ranges
│   └── qwen3_vl_embedding.py  # Qwen3-VL wrapper
│
├── generate/                  # Standalone CLI scripts (run directly on a GPU machine)
│   ├── anime_video_v2.py      # Beat-synced AMV generator (same logic as server)
│   ├── anime_video.py         # Simple lyric-matched version (legacy)
│   ├── render.py              # FFmpeg timeline renderer
│   └── audio_analysis.py      # CLAP mood analysis
│
├── docs/                      # GitHub Pages presentation
│   └── index.html             # Slide deck with embedded demo
│
├── archive/                   # Superseded stock-only pipeline
├── Dockerfile                 # HF Space image (Python 3.11-slim + ffmpeg)
└── vercel.json                # Frontend static deploy
```

## API

### `POST /generate`
```json
{"prompt": "Fast anime battle, 160bpm", "mode": "anime"}
→ {"job_id": "abc123def456"}
```

### `GET /job/{job_id}`
```json
{"status": "running", "stages": {...}, "title": "Heroes Never Fade Away"}
```

### `GET /preview/{job_id}`
Returns the full timeline (beats, clips, lyrics, intensities).

### `GET /render/{job_id}`
```json
{
  "video_url": "https://storage.rendi.dev/.../video.mp4",
  "assets": {"audio": "...", "clips": [...]},
  "command": "ffmpeg ..."  // fallback for local render
}
```

### `POST /regenerate`
```json
{"source_job_id": "abc123def456"}
→ {"job_id": "newid789xyz", "source_job_id": "abc123def456"}
```
Skips ElevenLabs (saves ~9s) — reuses stored audio + metadata to produce a new video with re-selected clips.

### `GET /history`
Returns the last 100 generations (title, prompt, mode, video_url, created_at).

### `GET /health`
Wake the Space from scale-to-zero. Frontend pings this on page load.

## Local Development

```bash
# Install server deps
cd server
pip install -r requirements.txt

# Set env vars (see .env.example)
export ELEVENLABS_API_KEY=...
export TURBOPUFFER_API_KEY=...
export DASHSCOPE_API_KEY=...
export SILICONFLOW_API_KEY=...
export RENDI_API_KEY=...
export HF_TOKEN=...       # for writing to amv-history dataset
export PIXABAY_API_KEY=... # normal mode only
export PEXELS_API_KEY=...  # normal mode only

uvicorn app:app --reload --port 7860
```

Open http://localhost:7860 — static files served by FastAPI.

## Indexing Anime Clips (one-time, GPU)

Requires a GPU for Qwen3-VL (~4GB VRAM).

```bash
cd indexing
python anime_embed.py --batch batch.json      # YouTube → split → embed → turbopuffer
python caption_clips.py                        # Caption frames → BM25 in turbopuffer
```

After this, upload the clips to a Hugging Face Space's `static/anime_clips/` via git LFS and update each turbopuffer row's `path` attribute to point to the public URL. The helper prompt for this is in `docs/upload_anime_clips_prompt.md`.

## Cost Breakdown

Most of the stack runs free; the two paid services covered their hackathon cost with sponsor credits:

| Component | Provider | Tier |
|---|---|---|
| Music generation | ElevenLabs API | Paid — covered by ElevenHacks sponsor credits |
| Vector search + BM25 | turbopuffer | Paid — covered by ElevenHacks sponsor credits |
| Text embedding (anime) | DashScope text-embedding-v4 | 1M free tokens / 90 days |
| Text embedding (stock) | SiliconFlow Qwen3-Embedding-0.6B | Free |
| Video rendering | Rendi | 50GB/month free |
| Backend hosting | HF Spaces (cpu-basic) | Free |
| Frontend hosting | Vercel | Free |
| History storage | HF Datasets | Free |
| Clip storage | HF Spaces git LFS | Free |

**Total infrastructure cost: minimal** for demo volumes — paid services were sponsor-comped, everything else is free-tier.

## Performance

| Stage | Time | Notes |
|---|---|---|
| Music generation | 9s | ElevenLabs API |
| Beat detection | 13s | librosa @11025Hz (downsampled for speed) |
| Search | 5-8s | 8-way parallel via ThreadPoolExecutor |
| Download | 0.1s (anime) / 3s (stock) | Symlinks for anime, parallel HTTP for stock |
| Render | 20s | Pipelined normalize + Rendi concat |
| **Total** | **~50-60s** | |

## License

MIT
