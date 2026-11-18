"""``POST /transform`` says in ``X-Result-Cached`` whether SHARP's prediction
came from the disk cache: a cached answer takes seconds, a new photo many times
that, and a client timing the calls needs to tell them apart."""

from __future__ import annotations

import io
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from PIL import Image


class FakePipeline:
    """Stands in for the stereo pipeline with a SHARP method, reporting the
    cache result the way ``sharp_predict`` does."""

    def __init__(self, cache: str | None) -> None:
        self.cache = cache
        self.settings = SimpleNamespace(stereo_method="sharp_splat_full")

    def derive(self, **overrides: Any) -> FakePipeline:
        return self

    def synthesize_with_depth_estimator(
        self, image: Image.Image, depth_estimator: Any, *,
        method: Any = None, intermediates: dict[str, Any] | None = None,
    ) -> Image.Image:
        if intermediates is not None and self.cache is not None:
            intermediates["sharp_cache"] = self.cache
        width, height = image.size
        return Image.new("RGB", (width * 2, height), (10, 20, 30))


def _transform(client: Any, **fields: str) -> Any:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (120, 90, 60)).save(buf, format="JPEG")
    buf.seek(0)
    data: dict[str, Any] = dict(fields)
    data["file"] = (buf, "photo.jpg")
    return client.post("/transform", data=data, content_type="multipart/form-data")


@pytest.mark.parametrize(
    ("cache", "header"),
    [("hit", "1"), ("miss", "0"), ("off", "0"), (None, "0")],
)
def test_the_answer_says_whether_it_came_from_the_cache(
    service_on: ModuleType, client: Any, monkeypatch: pytest.MonkeyPatch,
    cache: str | None, header: str,
) -> None:
    pipeline = FakePipeline(cache)
    monkeypatch.setattr(service_on, "_get_pipeline", lambda: pipeline)
    monkeypatch.setattr(service_on, "_get_depth_estimator", lambda **kw: None)

    answer = _transform(client)

    assert answer.status_code == 200, answer.get_data(as_text=True)
    assert answer.mimetype == "image/jpeg"
    assert answer.headers["X-Result-Cached"] == header


def test_a_request_with_its_own_settings_says_so_too(
    service_on: ModuleType, client: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``max_dim`` gives the request a pipeline of its own."""
    pipeline = FakePipeline("hit")
    monkeypatch.setattr(service_on, "_get_pipeline", lambda: pipeline)
    monkeypatch.setattr(service_on, "_get_depth_estimator", lambda **kw: None)

    answer = _transform(client, max_dim="1024")

    assert answer.headers["X-Result-Cached"] == "1"
