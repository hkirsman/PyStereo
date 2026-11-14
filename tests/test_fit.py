"""The output pixel budget (``stereo/fit.py``)."""

from __future__ import annotations

from PIL import Image

from pystereo_core.stereo.fit import MAX_TEXTURE_EDGE, fit_to_pixel_budget


def _image(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height))


def test_an_image_inside_the_budget_is_returned_untouched() -> None:
    img = _image(400, 300)
    assert fit_to_pixel_budget(img, 400 * 300) is img


def test_no_budget_means_no_change() -> None:
    img = _image(4000, 3000)
    assert fit_to_pixel_budget(img, 0) is img
    assert fit_to_pixel_budget(img, -1) is img


def test_a_larger_image_is_scaled_to_the_budget_keeping_its_shape() -> None:
    out = fit_to_pixel_budget(_image(4000, 3000), 1_000_000)
    width, height = out.size
    assert width * height <= 1_000_000
    # Rounded down per axis, so only just under the budget.
    assert width * height > 990_000
    assert abs(width / height - 4 / 3) < 0.01


def test_it_never_upscales() -> None:
    out = fit_to_pixel_budget(_image(200, 100), 50_000_000)
    assert out.size == (200, 100)


def test_one_budget_means_the_same_area_for_any_shape() -> None:
    wide = fit_to_pixel_budget(_image(8000, 1000), 500_000)
    tall = fit_to_pixel_budget(_image(1000, 8000), 500_000)
    assert wide.size == (tall.size[1], tall.size[0])


def test_even_width_keeps_the_two_eyes_the_same_size() -> None:
    # 1001 x 500 fits the budget but cannot be split into equal halves.
    out = fit_to_pixel_budget(_image(1001, 500), 10_000_000, even_width=True)
    assert out.size == (1000, 500)
    untouched = fit_to_pixel_budget(_image(1001, 500), 10_000_000)
    assert untouched.size == (1001, 500)


def test_an_edge_beyond_the_texture_limit_is_brought_inside_it() -> None:
    out = fit_to_pixel_budget(_image(MAX_TEXTURE_EDGE + 2000, 10), 10**9)
    assert max(out.size) == MAX_TEXTURE_EDGE


def test_a_tiny_budget_still_leaves_a_pixel() -> None:
    out = fit_to_pixel_budget(_image(4000, 10), 1)
    assert out.size == (1, 1)


def test_the_budget_holds_when_one_side_is_down_to_a_pixel() -> None:
    # 4000 x 10 scaled to 100 pixels would be 200 x 0.5. The short side stops
    # at one pixel, so the long side gives way and the shape is not kept.
    assert fit_to_pixel_budget(_image(4000, 10), 100).size == (100, 1)
    assert fit_to_pixel_budget(_image(10, 4000), 100).size == (1, 100)


def test_the_area_never_exceeds_the_budget() -> None:
    shapes = [(4000, 3000), (3000, 4000), (4000, 10), (10, 4000), (5000, 1), (333, 77)]
    for width, height in shapes:
        for budget in (1, 2, 7, 100, 999, 12_345, 1_000_000):
            for even_width in (False, True):
                out = fit_to_pixel_budget(_image(width, height), budget, even_width=even_width)
                w, h = out.size
                case = f"{width}x{height} into {budget}, even_width={even_width}: {w}x{h}"
                assert w >= 1 and h >= 1, case
                assert w * h <= budget, case
                assert w <= width and h <= height, case
