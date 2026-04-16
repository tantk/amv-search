"""Download helpers for Pixabay/Pexels video clips (normal/stock mode)."""

import os
import httpx


def _get_pixabay_url(video_id: str) -> str | None:
    api_key = os.environ.get("PIXABAY_API_KEY")
    if not api_key:
        return None
    try:
        resp = httpx.get("https://pixabay.com/api/videos/",
                         params={"key": api_key, "id": video_id}, timeout=15)
        if resp.status_code != 200:
            return None
        hits = resp.json().get("hits")
        if not hits:
            return None
        for size in ("medium", "small", "large"):
            url = hits[0].get("videos", {}).get(size, {}).get("url")
            if url:
                return url
    except (httpx.HTTPError, ValueError):
        pass
    return None


def _get_pexels_url(video_id: str) -> str | None:
    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        return None
    try:
        resp = httpx.get(f"https://api.pexels.com/videos/videos/{video_id}",
                         headers={"Authorization": api_key}, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        for vf in sorted(data.get("video_files", []), key=lambda v: v.get("width", 0), reverse=True):
            if vf.get("quality") in ("hd", "sd") and vf.get("link") and vf.get("width", 0) <= 1920:
                return vf["link"]
        files = data.get("video_files", [])
        if files:
            return files[0].get("link")
    except (httpx.HTTPError, ValueError):
        pass
    return None


def get_download_url(video_id: str, source: str, fallback_query: str = "") -> str | None:
    """Try both Pixabay and Pexels (dataset labels are unreliable)."""
    if video_id:
        if source == "pixabay":
            url = _get_pixabay_url(video_id)
            if url:
                return url
            url = _get_pexels_url(video_id)
            if url:
                return url
        else:
            url = _get_pexels_url(video_id)
            if url:
                return url
            url = _get_pixabay_url(video_id)
            if url:
                return url
    return None
