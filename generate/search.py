"""
Embedding + turbopuffer search for matching video clips.

Lazy-loads Qwen3-Embedding-0.6B and searches the music-video-clips
namespace in turbopuffer to find clips that match lyric text.
"""

import os

import httpx
import torch
from sentence_transformers import SentenceTransformer
import turbopuffer as tpuf

# ── Config ───────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
NAMESPACE = "music-video-clips"
REGION = "gcp-us-central1"

# ── Singletons ───────────────────────────────────────────────────
_model: SentenceTransformer | None = None
_namespace = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(
            MODEL_NAME,
            model_kwargs={"torch_dtype": torch.float16},
            tokenizer_kwargs={"padding_side": "left"},
        )
        _model.max_seq_length = 512
    return _model


def _get_namespace():
    """Lazy-load the turbopuffer namespace connection (singleton)."""
    global _namespace
    if _namespace is None:
        api_key = os.environ.get("TURBOPUFFER_API_KEY")
        if not api_key:
            raise RuntimeError("TURBOPUFFER_API_KEY environment variable is not set")
        client = tpuf.Turbopuffer(api_key=api_key, region=REGION)
        _namespace = client.namespace(NAMESPACE)
    return _namespace


def embed_text(text: str) -> list[float]:
    """Encode text with Qwen3, normalize, and return a 1024-dim vector."""
    model = _get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


_verified_cache: dict[str, bool] = {}


def _verify_video_id(video_id: str, source: str) -> bool:
    """Check if a video ID resolves to a real downloadable video."""
    cache_key = f"{source}:{video_id}"
    if cache_key in _verified_cache:
        return _verified_cache[cache_key]

    ok = False
    try:
        if source == "pixabay":
            api_key = os.environ.get("PIXABAY_API_KEY", "")
            resp = httpx.get(
                "https://pixabay.com/api/videos/",
                params={"key": api_key, "id": video_id},
                timeout=10,
            )
            ok = resp.status_code == 200 and bool(resp.json().get("hits"))
        elif source == "pexels":
            api_key = os.environ.get("PEXELS_API_KEY", "")
            resp = httpx.get(
                f"https://api.pexels.com/videos/videos/{video_id}",
                headers={"Authorization": api_key},
                timeout=10,
            )
            ok = resp.status_code == 200
    except (httpx.HTTPError, ValueError):
        pass

    _verified_cache[cache_key] = ok
    return ok


def search_clips(
    query: str,
    top_k: int = 5,
    used_video_ids: set[str] | None = None,
    category_filter: str | None = None,
    verify_ids: bool = True,
) -> list[dict]:
    """
    Embed *query* and search turbopuffer for matching video clips.

    Returns a list of dicts with keys:
        video_id, caption, source, category, path, score

    *used_video_ids* — set of video_id strings to exclude (de-dup).
    *category_filter* — if set, only return clips from this category.
    """
    vector = embed_text(query)
    ns = _get_namespace()

    # Fetch extra results to account for mixkit filtering, bad IDs, and dedup
    fetch_k = top_k * 10

    # Build query kwargs.
    query_kwargs = dict(
        top_k=fetch_k,
        distance_metric="cosine_distance",
        rank_by=("vector", "ANN", vector),
        include_attributes=["caption", "source", "category", "video_id", "path"],
    )

    if category_filter is not None:
        query_kwargs["filters"] = ("category", "Eq", category_filter)

    response = ns.query(**query_kwargs)

    # Sources we can actually download from (mixkit has no API)
    downloadable_sources = {"pexels", "pixabay"}

    clips: list[dict] = []
    seen_video_ids: set[str] = set(used_video_ids or ())

    for row in response.rows or []:
        attrs = row.model_extra or {}
        source = attrs.get("source", "")
        if source not in downloadable_sources:
            continue
        vid = attrs.get("video_id", "") or str(row.id)
        if not vid or vid in seen_video_ids:
            continue
        # Verify the ID actually resolves before returning it
        if verify_ids and not _verify_video_id(vid, source):
            continue
        seen_video_ids.add(vid)
        clips.append({
            "video_id": vid,
            "caption": attrs.get("caption", ""),
            "source": source,
            "category": attrs.get("category", ""),
            "path": attrs.get("path", ""),
            "score": attrs.get("$dist"),
        })
        if len(clips) >= top_k:
            break

    return clips
