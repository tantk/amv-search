# Upload Anime Clips to HF Space via Git LFS

## Context

I have ~1,800 anime clips (~2.67s each, ~1MB each, ~1.8GB total) stored locally at `generate/clips/{category}/clip_N.mp4`. These clips are indexed in turbopuffer's `anime-clips` namespace with their local filesystem paths stored in the `path` attribute.

I need to upload these clips to a Hugging Face Space so that the web backend (running on HF Spaces) can serve them via static file URLs during music video generation.

## Goal

1. Upload all anime clips from `generate/clips/` to the HF Space `tantk/music-video-gen` via git LFS under `/static/anime_clips/`
2. Update the turbopuffer `anime-clips` namespace so each row's `path` attribute points to the new public URL

## Target Space

- **Space**: `https://huggingface.co/spaces/tantk/music-video-gen`
- **Clone URL**: `https://huggingface.co/spaces/tantk/music-video-gen`
- **After upload, public URL pattern**: `https://tantk-music-video-gen.hf.space/static/anime_clips/{category}/clip_N.mp4`

## Prerequisites

You'll need:

1. HF token with write access (check `~/.cache/huggingface/token` or ask user)
2. Turbopuffer API key (check `.env` in the `turbopuffer` project root, variable `TURBOPUFFER_API_KEY`)
3. `git lfs` installed: `apt install git-lfs` or `brew install git-lfs`

## Step 1 — Clone the Space repo with LFS

```bash
git lfs install
HF_TOKEN=$(cat ~/.cache/huggingface/token)
cd /tmp
git clone https://tantk:${HF_TOKEN}@huggingface.co/spaces/tantk/music-video-gen
cd music-video-gen
```

## Step 2 — Configure LFS for video files

```bash
git lfs track "*.mp4"
git add .gitattributes
```

## Step 3 — Copy anime clips into the Space

Find where the anime clips live on this machine. Look for `generate/clips/` under the AMV project (likely `~/project/elevenlabs/turbopuff/generate/clips/` or similar). The structure is:

```
generate/clips/
  Attack_on_Titan/
    clip_0.mp4
    clip_1.mp4
    ...
  Demon_Slayer/
    clip_0.mp4
    ...
  ... (7 anime series total)
```

Copy them to the Space repo's `static/anime_clips/` directory:

```bash
# Adjust source path to where clips actually are on this machine
SRC="$HOME/project/elevenlabs/turbopuff/generate/clips"
DST="/tmp/music-video-gen/static/anime_clips"
mkdir -p "$DST"
cp -r "$SRC"/* "$DST/"

# Verify
find "$DST" -name "*.mp4" | wc -l  # Should be ~1800
du -sh "$DST"  # Should be ~1.8GB
```

## Step 4 — Commit and push in batches

Pushing 1.8GB in one commit may timeout. Commit per-category in batches:

```bash
cd /tmp/music-video-gen

for dir in static/anime_clips/*/; do
    category=$(basename "$dir")
    echo "Adding $category..."
    git add "$dir"
    git -c user.name="tantk" -c user.email="tantk7@gmail.com" \
        commit -m "feat: upload $category anime clips"
    git push
done
```

## Step 5 — Verify public access

```bash
# Pick any clip and check it's reachable
curl -I "https://tantk-music-video-gen.hf.space/static/anime_clips/Attack_on_Titan/clip_0.mp4"
# Should return 200 OK
```

## Step 6 — Update turbopuffer records

The clips in turbopuffer have `path` attributes pointing to local filesystem paths. Update them to HTTPS URLs.

**Location of the indexing code**: there's a script at `indexing/anime_embed.py` in this project that originally uploaded these clips. The key is the `path` attribute stored on each row.

Look at what's currently stored:

```python
import turbopuffer as tpuf
import os

client = tpuf.Turbopuffer(api_key=os.environ["TURBOPUFFER_API_KEY"], region="gcp-us-central1")
ns = client.namespace("anime-clips")

# Sample a few rows
import numpy as np
vec = np.random.randn(2048).tolist()
r = ns.query(top_k=5, rank_by=("vector", "ANN", vec), include_attributes=["path", "category", "video_id"])
for row in (r.rows or []):
    a = row.model_extra or {}
    print(a.get("path"), a.get("category"), a.get("video_id"))
```

Then batch-update the `path` attribute for all ~1,800 rows to point to `https://tantk-music-video-gen.hf.space/static/anime_clips/{category}/{filename}`.

Approach:
- Turbopuffer supports `ns.write(...)` with upsert to update specific attributes
- You need each row's `id` + the new `path` value
- Use `ns.query` with a large `top_k` and wildcard filter to fetch all rows, or iterate by category

Pseudocode:

```python
BASE_URL = "https://tantk-music-video-gen.hf.space/static/anime_clips"

# Fetch all rows (you may need pagination — turbopuffer has a max per-query limit)
all_rows = []
# ... fetch in batches ...

updates = []
for row in all_rows:
    old_path = row.attributes.get("path", "")
    # Extract category + filename from old path
    # e.g. /home/clawd/.../generate/clips/Attack_on_Titan/clip_42.mp4
    parts = old_path.split("/clips/")
    if len(parts) != 2:
        continue
    relative = parts[1]  # "Attack_on_Titan/clip_42.mp4"
    new_path = f"{BASE_URL}/{relative}"
    updates.append({"id": row.id, "attributes": {"path": new_path}})

# Batch upsert (turbopuffer accepts batches of 1000)
for i in range(0, len(updates), 1000):
    batch = updates[i:i+1000]
    ns.write(upsert_rows=batch, distance_metric="cosine_distance")
```

Verify a few records after update to confirm the path changed.

## Constraints / things to watch out for

- **LFS bandwidth**: HF has generous LFS bandwidth but very large pushes can be rate-limited. If a push hangs, split the commit smaller.
- **File naming**: HF Spaces serves `static/` folder at `/static/` URL path automatically. No extra config needed.
- **Don't commit individual MP4s without LFS** — they'd bloat the repo history. Make sure `*.mp4` is in `.gitattributes` before `git add`.
- **Don't overwrite other Space files** — only add things under `static/anime_clips/`. The Space already has `app.py`, `pipeline.py`, etc. Leave them alone.
- **Check storage limit** — free HF Spaces have 50GB ephemeral; git LFS files are separate and unlimited for HF, so this is fine.

## Expected Result

After completing all steps:

1. Browsing `https://huggingface.co/spaces/tantk/music-video-gen/tree/main/static/anime_clips/` shows the uploaded clips
2. Direct URL like `https://tantk-music-video-gen.hf.space/static/anime_clips/Demon_Slayer/clip_7.mp4` plays the clip in the browser
3. Turbopuffer `anime-clips` namespace has rows with `path` = `https://tantk-music-video-gen.hf.space/static/anime_clips/...`
4. The music video pipeline on the Space can now download anime clips via HTTP instead of failing on local paths

Report back total clips uploaded, any failures, and a sample verification URL.
