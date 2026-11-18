"""Warp primitives on small synthetic arrays (``stereo/warp.py``)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pystereo_core.stereo.warp import (
    adaptive_max_disparity,
    dilate_occlusion_mask,
    inverse_remap_warp_eye,
)

HEIGHT, WIDTH = 8, 32


def _columns() -> np.ndarray:
    """An image whose red channel is its column number."""
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(WIDTH, dtype=np.uint8)[None, :] * 4
    return image


def _flat(depth: float) -> np.ndarray:
    return np.full((HEIGHT, WIDTH), depth, dtype=np.float32)


def _warp(depth: np.ndarray, eye: str, disparity: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    return inverse_remap_warp_eye(
        _columns(), depth, disparity, eye, interpolation=cv2.INTER_LINEAR,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("eye", ["left", "right"])
def test_the_far_plane_does_not_move(eye: str) -> None:
    warped, _ = _warp(_flat(0.0), eye)
    np.testing.assert_array_equal(warped, _columns())


def test_the_near_plane_moves_half_the_separation_each_way() -> None:
    source = _columns()
    left, _ = _warp(_flat(1.0), "left")
    right, _ = _warp(_flat(1.0), "right")
    # Separation 8: each eye samples 4 columns to its own side.
    np.testing.assert_array_equal(left[:, 4:], source[:, :-4])
    np.testing.assert_array_equal(right[:, :-5], source[:, 4:-1])


def test_nearer_things_move_further() -> None:
    source = _columns()
    half, _ = _warp(_flat(0.5), "left")
    np.testing.assert_array_equal(half[:, 2:], source[:, :-2])


def test_the_side_with_no_source_is_left_black_and_marked_as_a_hole() -> None:
    left, mask = _warp(_flat(1.0), "left")
    assert not left[:, :4].any()
    assert (mask[:, :4] == 255).all()
    # Away from that edge and the last column, a flat plane has no holes.
    assert not mask[:, 4:-1].any()


@pytest.mark.parametrize("eye", ["left", "right"])
def test_a_step_in_depth_is_marked_at_both_of_its_edges(eye: str) -> None:
    depth = _flat(0.0)
    depth[:, 12:20] = 1.0
    warped, mask = _warp(depth, eye)
    # Where the sampling jumps, on either side of the near thing, and nowhere
    # else but the last column, which is always marked.
    assert (mask[:, 12] == 255).all()
    assert (mask[:, 20] == 255).all()
    assert not np.delete(mask, [12, 20, WIDTH - 1], axis=1).any()
    # The far parts stay where they were; only the near thing is resampled.
    np.testing.assert_array_equal(warped[:, :12], _columns()[:, :12])
    np.testing.assert_array_equal(warped[:, 20:], _columns()[:, 20:])
    assert not np.array_equal(warped[:, 12:20], _columns()[:, 12:20])


def test_the_image_and_its_depth_must_match() -> None:
    with pytest.raises(ValueError, match="match"):
        inverse_remap_warp_eye(_columns(), np.zeros((4, 4), np.float32), 8.0, "left")
    with pytest.raises(ValueError, match="RGB"):
        inverse_remap_warp_eye(np.zeros((HEIGHT, WIDTH), np.uint8), _flat(0.0), 8.0, "left")


def test_fixed_separation_is_the_base_ratio_of_the_width() -> None:
    px = adaptive_max_disparity(
        _flat(0.5), 1000, base_ratio=0.03, min_ratio=0.015, max_ratio=0.06, adaptive=False,
    )
    assert px == pytest.approx(30.0)


def test_separation_is_clamped_to_its_limits() -> None:
    low = adaptive_max_disparity(
        _flat(0.5), 1000, base_ratio=0.001, min_ratio=0.015, max_ratio=0.06, adaptive=False,
    )
    high = adaptive_max_disparity(
        _flat(0.5), 1000, base_ratio=0.5, min_ratio=0.015, max_ratio=0.06, adaptive=False,
    )
    assert low == pytest.approx(15.0)
    assert high == pytest.approx(60.0)


def test_a_scene_with_little_depth_gets_less_separation_than_a_deep_one() -> None:
    limits = {"base_ratio": 0.03, "min_ratio": 0.001, "max_ratio": 1.0}
    shallow = adaptive_max_disparity(_flat(0.5), 1000, **limits)
    ramp = np.tile(np.linspace(0.0, 1.0, WIDTH, dtype=np.float32), (HEIGHT, 1))
    deep = adaptive_max_disparity(ramp, 1000, **limits)
    # Half the base for no spread at all, up to one and a half times for a wide one.
    assert shallow == pytest.approx(15.0)
    assert shallow < deep <= 45.0


def test_an_8_bit_depth_map_is_read_as_0_to_1() -> None:
    limits = {"base_ratio": 0.03, "min_ratio": 0.001, "max_ratio": 1.0}
    ramp = np.tile(np.linspace(0.0, 1.0, WIDTH, dtype=np.float32), (HEIGHT, 1))
    as_bytes = np.rint(ramp * 255).astype(np.uint8)
    assert adaptive_max_disparity(as_bytes, 1000, **limits) == pytest.approx(
        adaptive_max_disparity(ramp, 1000, **limits), rel=0.02,
    )


def test_dilating_a_hole_grows_it_and_zero_leaves_it() -> None:
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[4, 4] = 255
    assert dilate_occlusion_mask(mask, 0) is mask
    grown = dilate_occlusion_mask(mask, 2)
    assert grown[4, 2] == 255 and grown[2, 4] == 255
    assert grown[0, 0] == 0
    assert np.count_nonzero(grown) > 1
