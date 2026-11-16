"""``POST /transform`` with ``output=depth``, with a stand-in for the depth
model: what is sent back, at what size and bit depth, and what is refused."""

from __future__ import annotations

import io
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from PIL import Image

PHOTO_SIZE = (1200, 800)


class FakeEstimator:
    """Answers with a left-to-right ramp, 0 at the left edge and 1 at the right."""

    def __init__(self) -> None:
        self.seen: list[tuple[int, int]] = []

    def process_raw(self, image: Image.Image) -> np.ndarray:
        self.seen.append(image.size)
        width, height = image.size
        return np.tile(np.linspace(0.0, 1.0, width, dtype=np.float32), (height, 1))


class FakeDownloads:
    def __init__(self, local: set[str]) -> None:
        self.local = local

    def is_depth_model_local(self, model_size: str) -> bool:
        return model_size in self.local


@pytest.fixture
def estimator(service_on: ModuleType, monkeypatch: pytest.MonkeyPatch) -> FakeEstimator:
    """The service on, the small depth model "downloaded", no real model."""
    from pystereo_core import download

    fake = FakeEstimator()
    monkeypatch.setattr(download, "get_download_manager", lambda: FakeDownloads({"small"}))
    monkeypatch.setattr(service_on, "_ensure_registry", lambda model_size="small": None)
    monkeypatch.setattr(service_on, "_get_depth_estimator", lambda: fake)
    monkeypatch.setattr(service_on, "_model_status_payload", lambda: {"state": "missing"})

    def no_stereo() -> Any:
        raise AssertionError("the stereo pipeline must not run for a depth map")

    monkeypatch.setattr(service_on, "_get_pipeline", no_stereo)
    return fake


def _photo(size: tuple[int, int] = PHOTO_SIZE) -> io.BytesIO:
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 90, 60)).save(buf, format="JPEG")
    buf.seek(0)
    return buf


def _depth(client: Any, **fields: Any) -> Any:
    data: dict[str, Any] = {k: str(v) for k, v in fields.items()}
    data["output"] = "depth"
    data["file"] = (_photo(), "photo.jpg")
    return client.post("/transform", data=data, content_type="multipart/form-data")


def _png(answer: Any) -> Image.Image:
    assert answer.status_code == 200, answer.get_data(as_text=True)
    assert answer.mimetype == "image/png"
    return Image.open(io.BytesIO(answer.data))


def test_the_depth_map_is_an_8_bit_png_of_the_photos_size(
    estimator: FakeEstimator, client: Any,
) -> None:
    answer = _depth(client)
    img = _png(answer)
    assert img.mode == "L"
    assert img.size == PHOTO_SIZE
    assert answer.headers["Content-Disposition"] == 'attachment; filename="depth.png"'
    assert int(answer.headers["Content-Length"]) == len(answer.data)
    assert estimator.seen == [PHOTO_SIZE]


def test_near_is_white_far_is_black_and_values_are_rounded(
    estimator: FakeEstimator, client: Any,
) -> None:
    row = np.asarray(_png(_depth(client)))[0]
    assert row[0] == 0 and row[-1] == 255
    expected = np.rint(np.linspace(0.0, 1.0, PHOTO_SIZE[0]) * 255)
    np.testing.assert_array_equal(row, expected)


def test_16_bits_use_the_whole_range(estimator: FakeEstimator, client: Any) -> None:
    img = _png(_depth(client, depth_bits=16))
    assert img.mode == "I;16"
    row = np.asarray(img)[0]
    assert row.dtype == np.uint16
    assert row[0] == 0 and row[-1] == 65535
    # A ramp over 1200 columns has more levels than 8 bits can hold.
    assert len(np.unique(row)) == PHOTO_SIZE[0]


def test_max_pixels_bounds_the_map_sent_back(estimator: FakeEstimator, client: Any) -> None:
    img = _png(_depth(client, max_pixels=60_000))
    width, height = img.size
    assert width * height <= 60_000
    assert abs(width / height - 1.5) < 0.02
    # The model still saw the whole photo.
    assert estimator.seen == [PHOTO_SIZE]
    row = np.asarray(img)[0]
    assert row[0] < 5 and row[-1] > 250


def test_max_dim_bounds_the_image_the_model_is_given(
    estimator: FakeEstimator, client: Any,
) -> None:
    img = _png(_depth(client, max_dim=600))
    assert estimator.seen == [(600, 400)]
    assert img.size == (600, 400)


def test_max_dim_has_the_same_floor_as_for_stereo(
    estimator: FakeEstimator, client: Any,
) -> None:
    _png(_depth(client, max_dim=10))
    assert estimator.seen == [(512, 341)]


def test_a_max_dim_that_is_not_a_number_is_ignored(
    estimator: FakeEstimator, client: Any,
) -> None:
    assert _png(_depth(client, max_dim="big")).size == PHOTO_SIZE


def test_both_limits_and_16_bits_together(estimator: FakeEstimator, client: Any) -> None:
    img = _png(_depth(client, max_dim=600, max_pixels=60_000, depth_bits=16))
    assert img.mode == "I;16"
    assert img.size[0] * img.size[1] <= 60_000
    assert estimator.seen == [(600, 400)]


def test_the_stereo_method_makes_no_difference(estimator: FakeEstimator, client: Any) -> None:
    plain = _depth(client).data
    assert _depth(client, method="sharp_splat").data == plain
    assert _depth(client, method="no_such_method").data == plain


@pytest.mark.parametrize(
    ("fields", "named"),
    [
        ({"depth_bits": "12"}, "12"),
        ({"depth_bits": "sixteen"}, "sixteen"),
        ({"depth_model": "huge"}, "huge"),
    ],
)
def test_unknown_values_are_refused_before_any_work(
    estimator: FakeEstimator, client: Any, fields: dict[str, str], named: str,
) -> None:
    answer = _depth(client, **fields)
    assert answer.status_code == 400
    assert named in answer.get_json()["error"]
    assert estimator.seen == []


def test_a_depth_model_that_is_not_downloaded_answers_not_ready(
    estimator: FakeEstimator, client: Any,
) -> None:
    answer = _depth(client, depth_model="large")
    assert answer.status_code == 503
    assert answer.get_json()["error"] == "model_not_ready"
    assert estimator.seen == []


def test_only_the_depth_model_is_needed_not_the_stereo_pack(
    estimator: FakeEstimator, client: Any,
) -> None:
    # The stand-in download manager has no ``is_pack_ready``: asking for it
    # would fail the request.
    assert _depth(client).status_code == 200


def test_a_failing_model_is_an_error_not_a_crash(
    estimator: FakeEstimator, client: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(image: Image.Image) -> np.ndarray:
        raise RuntimeError("out of memory")

    monkeypatch.setattr(estimator, "process_raw", broken)
    answer = _depth(client)
    assert answer.status_code == 500
    assert "error" in answer.get_json()


def test_no_estimator_answers_503(
    estimator: FakeEstimator, service_on: ModuleType, client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_on, "_get_depth_estimator", lambda: None)
    assert _depth(client).status_code == 503


def test_the_service_switch_applies_to_depth_too(web: ModuleType, client: Any) -> None:
    assert _depth(client).status_code == 403
