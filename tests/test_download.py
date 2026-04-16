import os
import tempfile
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("PIXABAY_API_KEY"),
    reason="PIXABAY_API_KEY not set",
)

def test_get_pixabay_download_url():
    from download import get_download_url
    url = get_download_url(video_id="5046217", source="pixabay")
    assert url is not None
    assert url.startswith("http")

def test_download_clip_to_file():
    from download import download_clip
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.mp4")
        result = download_clip(video_id="5046217", source="pixabay", output_path=path)
        assert result is True
        assert os.path.exists(path)
        assert os.path.getsize(path) > 10000
