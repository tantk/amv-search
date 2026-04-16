"""
Embedding + turbopuffer search for matching video clips.

Two free embedding APIs:
  - normal (music-video-clips): SiliconFlow Qwen3-Embedding-0.6B (1024-dim)
  - anime (anime-clips): DashScope text-embedding-v4 (2048-dim)
"""

import os

import httpx
import turbopuffer as tpuf

# ── Config ───────────────────────────────────────────────────────
REGION = "gcp-us-central1"

SILICONFLOW_URL = "https://api.siliconflow.cn/v1/embeddings"
DASHSCOPE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/embeddings"

# ── Singletons ───────────────────────────────────────────────────
_client = None
_namespaces: dict = {}


def _get_namespace(namespace: str = "music-video-clips"):
    """Get a turbopuffer namespace connection (cached per namespace)."""
    global _client
    if _client is None:
        api_key = os.environ.get("TURBOPUFFER_API_KEY")
        if not api_key:
            raise RuntimeError("TURBOPUFFER_API_KEY environment variable is not set")
        _client = tpuf.Turbopuffer(api_key=api_key, region=REGION)
    if namespace not in _namespaces:
        _namespaces[namespace] = _client.namespace(namespace)
    return _namespaces[namespace]


def _embed_siliconflow(text: str, dimensions: int = 1024) -> list[float]:
    """Embed via SiliconFlow Qwen3-Embedding-0.6B (free, 1024-dim)."""
    api_key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not api_key:
        raise RuntimeError("SILICONFLOW_API_KEY not set")

    resp = httpx.post(
        SILICONFLOW_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "Qwen/Qwen3-Embedding-0.6B", "input": text, "dimensions": dimensions},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def _embed_dashscope(text: str, dimensions: int = 2048) -> list[float]:
    """Embed via DashScope text-embedding-v4."""
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY not set")

    resp = httpx.post(
        DASHSCOPE_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": "text-embedding-v4", "input": text, "dimensions": dimensions},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def embed_text(text: str, namespace: str = "music-video-clips") -> list[float]:
    """Route to the correct embedding backend based on namespace.

    Stock footage tries SiliconFlow first (exact 0.6B match),
    falls back to DashScope (less accurate but functional).
    """
    if namespace == "anime-clips":
        return _embed_dashscope(text, dimensions=2048)
    else:
        # Try SiliconFlow (exact model match), fall back to DashScope
        try:
            return _embed_siliconflow(text, dimensions=1024)
        except Exception as e:
            print(f"[Embed] SiliconFlow failed: {e}, falling back to DashScope")
            return _embed_dashscope(text, dimensions=1024)


def _rrf_fuse(result_lists: list[list], k: int = 60) -> list:
    """Reciprocal rank fusion — combine multiple ranked result lists.

    Each row scored as sum of 1/(k + rank) across the lists it appears in.
    """
    scores: dict = {}
    rowdata: dict = {}
    for results in result_lists:
        for rank, row in enumerate(results):
            rid = str(row.id)
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank)
            rowdata[rid] = row
    ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [rowdata[rid] for rid in ranked_ids]


def search_clips(
    query: str,
    top_k: int = 5,
    used_video_ids: set[str] | None = None,
    category_filter: str | None = None,
    namespace: str = "music-video-clips",
) -> list[dict]:
    """
    Hybrid search on turbopuffer: vector ANN + BM25 on caption, fused via RRF.

    - Vector search: semantic similarity (good when embeddings align)
    - BM25 on caption: literal keyword matching (good as fallback)
    - RRF fusion: clips scoring well on both rank highest
    """
    vector = embed_text(query, namespace=namespace)
    ns = _get_namespace(namespace)
    is_anime = namespace == "anime-clips"

    fetch_k = top_k * 10

    # Multi-query: vector ANN + BM25 on caption in a single round-trip
    try:
        response = ns.multi_query(queries=[
            {
                "rank_by": ("vector", "ANN", vector),
                "top_k": fetch_k,
                "include_attributes": ["caption", "source", "category", "video_id", "path"],
            },
            {
                "rank_by": ("caption", "BM25", query),
                "top_k": fetch_k,
                "include_attributes": ["caption", "source", "category", "video_id", "path"],
            },
        ])
        vector_rows = response.results[0].rows or []
        bm25_rows = response.results[1].rows or []
        merged = _rrf_fuse([vector_rows, bm25_rows])
    except Exception as e:
        # Fallback to vector-only if multi_query fails (e.g. BM25 not indexed yet)
        print(f"[Search] hybrid failed ({e}), vector-only")
        query_kwargs = dict(
            top_k=fetch_k,
            distance_metric="cosine_distance",
            rank_by=("vector", "ANN", vector),
            include_attributes=["caption", "source", "category", "video_id", "path"],
        )
        if category_filter is not None:
            query_kwargs["filters"] = ("category", "Eq", category_filter)
        response = ns.query(**query_kwargs)
        merged = response.rows or []

    clips: list[dict] = []
    seen_video_ids: set[str] = set(used_video_ids or ())

    for row in merged:
        attrs = row.model_extra or {}
        source = attrs.get("source", "")

        if not is_anime and source not in ("pexels", "pixabay"):
            continue

        vid = attrs.get("video_id", "") or str(row.id)
        if not vid or vid in seen_video_ids:
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
