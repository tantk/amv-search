"""
Caption all anime clips using Qwen3-VL-2B-Instruct, then update turbopuffer.

Usage:
    python caption_clips.py

What it does:
1. Loads Qwen3-VL-2B-Instruct (~4 GB VRAM)
2. For each clip, captions the middle frame (f1.jpg)
3. Updates turbopuffer rows with the caption text attribute
"""

import glob
import json
import os
import sys
import time

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import turbopuffer as tpuf

# ── Config ────────────────────────────────────────────────────
TURBOPUFFER_API_KEY = os.environ.get("TURBOPUFFER_API_KEY", "")
NAMESPACE = "anime-clips"
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
FRAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames")
CAPTIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anime_captions.json")


def main():
    # ── Step 1: Collect all clips and their middle frames ─────
    print("Collecting frames...", flush=True)
    categories = sorted(os.listdir(FRAMES_DIR))
    clip_frames = {}  # clip_id -> frame_path

    for cat in categories:
        cat_dir = os.path.join(FRAMES_DIR, cat)
        if not os.path.isdir(cat_dir):
            continue
        # Find unique clip names
        frames = sorted(glob.glob(os.path.join(cat_dir, "*_f1.jpg")))  # middle frame
        for f in frames:
            clip_name = os.path.basename(f).replace("_f1.jpg", "")
            clip_id = f"{cat}_{clip_name}"
            clip_frames[clip_id] = f

    print(f"Found {len(clip_frames)} clips to caption\n", flush=True)

    # ── Step 2: Load if we have existing captions ─────────────
    existing = {}
    if os.path.exists(CAPTIONS_FILE):
        with open(CAPTIONS_FILE) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing captions", flush=True)

    to_caption = {k: v for k, v in clip_frames.items() if k not in existing}
    print(f"Need to caption: {len(to_caption)} clips\n", flush=True)

    if not to_caption:
        print("All clips already captioned. Skipping to upload.")
    else:
        # ── Step 3: Load captioning model ─────────────────────
        print("=" * 60)
        print(f"Loading {MODEL_NAME}...")
        print("=" * 60, flush=True)

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        model.eval()
        print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB\n", flush=True)

        # ── Step 4: Caption each clip ─────────────────────────
        print("=" * 60)
        print(f"Captioning {len(to_caption)} clips...")
        print("=" * 60, flush=True)

        start_time = time.time()
        captions = dict(existing)  # start with existing

        for i, (clip_id, frame_path) in enumerate(to_caption.items()):
            try:
                image = Image.open(frame_path).convert("RGB")

                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": (
                        "Describe what is visually happening in this anime frame. "
                        "Focus on: setting, character actions, mood, visual elements. "
                        "One concise sentence, 15-25 words. "
                        "Do not name specific characters or anime titles."
                    )},
                ]}]

                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=60, do_sample=False)

                generated = output_ids[0][inputs.input_ids.shape[1]:]
                caption = processor.decode(generated, skip_special_tokens=True).strip()
                captions[clip_id] = caption

            except Exception as e:
                captions[clip_id] = "anime scene"
                print(f"  WARNING: {clip_id} failed: {e}", flush=True)

            done = i + 1
            if done % 20 == 0 or done == len(to_caption):
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(to_caption) - done) / rate if rate > 0 else 0
                sample = caption if 'caption' in dir() else ''
                print(f"  {done}/{len(to_caption)} ({100*done//len(to_caption)}%) - {rate:.1f} clips/sec - ETA {eta:.0f}s", flush=True)
                if done % 100 == 0:
                    print(f"    last: \"{sample[:80]}\"", flush=True)

            # Save periodically
            if done % 100 == 0:
                with open(CAPTIONS_FILE, "w") as f:
                    json.dump(captions, f)

        # Final save
        with open(CAPTIONS_FILE, "w") as f:
            json.dump(captions, f, indent=2)
        print(f"\nCaptioning done. Saved {len(captions)} captions to {CAPTIONS_FILE}", flush=True)

        # Free GPU memory
        del model, processor
        torch.cuda.empty_cache()

    # ── Step 5: Update turbopuffer with captions ──────────────
    print("\n" + "=" * 60)
    print("Updating turbopuffer with captions...")
    print("=" * 60, flush=True)

    with open(CAPTIONS_FILE) as f:
        captions = json.load(f)

    client = tpuf.Turbopuffer(api_key=TURBOPUFFER_API_KEY, region="gcp-us-central1")
    ns = client.namespace(NAMESPACE)

    # Update in batches — we only need to update the caption attribute
    clip_ids = list(captions.keys())
    BATCH = 200
    upload_start = time.time()

    for start in range(0, len(clip_ids), BATCH):
        end = min(start + BATCH, len(clip_ids))
        batch_ids = clip_ids[start:end]

        rows = []
        for cid in batch_ids:
            rows.append({
                "id": cid,
                "caption": captions[cid][:500],
            })

        ns.write(upsert_rows=rows)
        elapsed = time.time() - upload_start
        print(f"  Updated {end}/{len(clip_ids)} ({100*end//len(clip_ids)}%) - {elapsed:.0f}s", flush=True)

    print(f"\nDone! {len(captions)} captions uploaded to turbopuffer.", flush=True)


if __name__ == "__main__":
    main()
