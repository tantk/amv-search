"""
GPU Embedding Script — Run this on your machine with a GPU.

What it does:
1. Downloads the caption metadata from HuggingFace (312MB JSON)
2. Embeds all 434K captions using Qwen3-Embedding-0.6B on GPU
3. Uploads all vectors directly to turbopuffer

Requirements:
    pip install sentence-transformers huggingface-hub turbopuffer python-dotenv torch

Estimated time: ~5-10 minutes on a modern GPU (RTX 3060+)
Estimated GPU RAM: ~1.2 GB
"""

import json
import os
import re
import time

# ── Step 0: Config ──────────────────────────────────────────────
TURBOPUFFER_API_KEY = os.environ.get("TURBOPUFFER_API_KEY", "")
NAMESPACE = "music-video-clips"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 64  # embedding batch size
UPSERT_BATCH = 500  # turbopuffer upsert batch size


# ── Step 1: Download captions ──────────────────────────────────
print("=" * 60)
print("STEP 1: Downloading caption metadata from HuggingFace...")
print("=" * 60)

from huggingface_hub import hf_hub_download

captions_path = hf_hub_download(
    "LanguageBind/Open-Sora-Plan-v1.0.0",
    "llava_path_cap_64x512x512.json",
    repo_type="dataset",
)
print(f"Downloaded to: {captions_path}")

with open(captions_path) as f:
    clips = json.load(f)
print(f"Loaded {len(clips)} clips\n")

# Parse captions
captions = []
for clip in clips:
    cap = clip.get("cap", "")
    if isinstance(cap, list):
        cap = cap[0] if cap else ""
    captions.append(cap or "no description")


# ── Step 2 & 3: Embed or load from cache ─────────────────────
import numpy as np

VECTORS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectors_1024.npy")

if os.path.exists(VECTORS_PATH):
    print("=" * 60)
    print(f"STEP 2-3: Loading cached vectors from {VECTORS_PATH}...")
    print("=" * 60)
    all_vectors = np.load(VECTORS_PATH)
    embed_time = 0
    print(f"Loaded {all_vectors.shape} vectors from cache\n")
else:
    print("=" * 60)
    print(f"STEP 2: Loading {MODEL_NAME} on GPU...")
    print("=" * 60)

    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: No GPU detected. This will be very slow.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float16).to(device)
    model.eval()

    dim = model.config.hidden_size
    print(f"Model loaded. Embedding dimension: {dim}")
    print(f"GPU memory used: {torch.cuda.memory_allocated()/1024**2:.0f} MB\n")

    # ── Step 3: Embed all captions ────────────────────────────────
    print("=" * 60)
    print(f"STEP 3: Embedding {len(captions)} captions in batches of {BATCH_SIZE}...")
    print("=" * 60)

    def embed_batch(texts):
        """Embed a batch using last-token pooling + normalize."""
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        attention_mask = inputs["attention_mask"]
        last_idx = attention_mask.sum(dim=1) - 1
        hidden = outputs.last_hidden_state
        embeddings = hidden[torch.arange(hidden.size(0), device=device), last_idx]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    start_time = time.time()
    all_vectors = []
    total_batches = (len(captions) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(captions), BATCH_SIZE):
        batch = captions[i : i + BATCH_SIZE]
        vecs = embed_batch(batch)
        all_vectors.append(vecs)
        done = min(i + BATCH_SIZE, len(captions))
        batch_num = done // BATCH_SIZE
        if batch_num % 20 == 0 or done == len(captions):
            elapsed = time.time() - start_time
            rate = done / elapsed
            eta = (len(captions) - done) / rate if rate > 0 else 0
            print(f"  {done}/{len(captions)} ({100*done//len(captions)}%) - {rate:.0f} texts/sec - ETA {eta:.0f}s", flush=True)

    all_vectors = np.vstack(all_vectors)
    embed_time = time.time() - start_time
    print(f"\nEmbedding done in {embed_time:.1f}s ({len(captions)/embed_time:.0f} texts/sec)")
    print(f"Vector shape: {all_vectors.shape}")

    # Save to disk so we never lose 60 min of GPU work
    np.save(VECTORS_PATH, all_vectors)
    print(f"Saved vectors to {VECTORS_PATH}\n")


# ── Step 4: Upload to turbopuffer ─────────────────────────────
print("=" * 60)
print(f"STEP 4: Uploading {len(clips)} vectors to turbopuffer...")
print("=" * 60)

import turbopuffer as tpuf

client = tpuf.Turbopuffer(
    api_key=TURBOPUFFER_API_KEY,
    region="gcp-us-central1",
)
ns = client.namespace(NAMESPACE)

# Delete existing namespace (old 384-dim vectors)
print("  Deleting old namespace data...")
try:
    ns.delete_all()
    print("  Old data deleted.\n")
except Exception as e:
    print(f"  No existing data to delete ({e})\n")


def parse_path(path):
    parts = path.split("/")
    source = None
    category = None
    for i, p in enumerate(parts):
        if p in ("pexels", "mixkit", "pixabay"):
            source = p
            if i + 1 < len(parts):
                category = parts[i + 1]
            break
    filename = parts[-1] if parts else ""
    numbers = re.findall(r"_(\d{5,})", filename)
    video_id = numbers[0] if numbers else None
    return source, category, video_id


upload_start = time.time()
total = len(clips)

for start in range(0, total, UPSERT_BATCH):
    end = min(start + UPSERT_BATCH, total)
    rows = []
    for i in range(start, end):
        source, category, video_id = parse_path(clips[i]["path"])
        rows.append({
            "id": i,
            "vector": all_vectors[i].tolist(),
            "caption": captions[i][:500],
            "source": source or "unknown",
            "category": category or "unknown",
            "video_id": video_id or "",
            "path": clips[i]["path"],
        })
    ns.write(upsert_rows=rows, distance_metric="cosine_distance")
    if end % 50000 < UPSERT_BATCH or end == total:
        elapsed = time.time() - upload_start
        print(f"  Upserted {end}/{total} ({100*end//total}%) - {elapsed:.0f}s")

total_time = time.time() - start_time
print(f"\n{'=' * 60}")
print(f"DONE!")
print(f"  Embedded:  {total} captions in {embed_time:.1f}s")
print(f"  Uploaded:  {total} vectors to turbopuffer")
print(f"  Total:     {total_time:.1f}s")
print(f"  Namespace: {NAMESPACE}")
print(f"{'=' * 60}")
