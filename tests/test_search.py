import os
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("TURBOPUFFER_API_KEY"),
    reason="TURBOPUFFER_API_KEY not set",
)

def test_search_returns_clips_with_required_fields():
    from search import search_clips
    results = search_clips("a beautiful sunset over the ocean", top_k=3)
    assert len(results) == 3
    for clip in results:
        assert "video_id" in clip
        assert "caption" in clip
        assert "source" in clip
        assert "category" in clip
        assert clip["source"] in ("pexels", "pixabay", "mixkit")

def test_search_avoids_used_ids():
    from search import search_clips
    first = search_clips("a beautiful sunset over the ocean", top_k=1)
    used = {first[0]["video_id"]}
    second = search_clips("a beautiful sunset over the ocean", top_k=1, used_video_ids=used)
    assert second[0]["video_id"] != first[0]["video_id"]
