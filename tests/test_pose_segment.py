"""Synthetic-image tests for pose localisation and arm segmentation."""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.pose_localize import (
    arm_axis_angle_degrees,
    locate_arm_midpoint,
    midpoint_px,
)
from app.pipeline.segment import measure_width_at_row, segment_arm

# BGR skin tone inside the HSV inRange bands used by segment.py.
SKIN_BGR = (90, 150, 210)
BG_BGR = (30, 30, 30)
CANVAS = 400
ARM_X0, ARM_X1 = 150, 250  # 100 px wide vertical "arm"
ARM_Y0, ARM_Y1 = 50, 350
ARM_WIDTH_PX = ARM_X1 - ARM_X0
MIDPOINT = ((ARM_X0 + ARM_X1) / 2.0, (ARM_Y0 + ARM_Y1) / 2.0)
CROP_SIZE = 200


def _synthetic_arm_image() -> np.ndarray:
    image = np.full((CANVAS, CANVAS, 3), BG_BGR, dtype=np.uint8)
    image[ARM_Y0:ARM_Y1, ARM_X0:ARM_X1] = SKIN_BGR
    return image


def test_midpoint_and_arm_angle_helpers() -> None:
    shoulder = (100.0, 80.0)
    elbow = (100.0, 180.0)
    mid = midpoint_px(shoulder, elbow)
    assert mid == pytest.approx((100.0, 130.0))
    # Vertical arm pointing down → ~90° in image coordinates.
    assert arm_axis_angle_degrees(shoulder, elbow) == pytest.approx(90.0, abs=1e-6)


def test_locate_arm_midpoint_handles_synthetic_image() -> None:
    image = _synthetic_arm_image()
    result = locate_arm_midpoint(image, side="left")

    assert set(result) >= {
        "detected",
        "midpoint_px",
        "shoulder_px",
        "elbow_px",
        "arm_angle_degrees",
        "confidence",
    }
    assert result["detected"] in (True, False)
    assert 0.0 <= result["confidence"] <= 1.0
    if result["detected"]:
        x, y = result["midpoint_px"]
        assert 0 <= x < CANVAS
        assert 0 <= y < CANVAS
        assert result["shoulder_px"] is not None
        assert result["elbow_px"] is not None
        assert result["arm_angle_degrees"] is not None
    else:
        assert result["midpoint_px"] is None


@pytest.mark.parametrize("image", [None, np.array([]), np.zeros((0, 0, 3), dtype=np.uint8)])
def test_locate_arm_midpoint_empty_image(image: np.ndarray | None) -> None:
    result = locate_arm_midpoint(image, side="left")
    assert result["detected"] is False
    assert result["midpoint_px"] is None


def test_segment_arm_on_skin_rectangle() -> None:
    image = _synthetic_arm_image()
    result = segment_arm(image, MIDPOINT, crop_size_px=CROP_SIZE)

    mask = result["mask"]
    assert mask.shape == (CROP_SIZE, CROP_SIZE)
    assert mask.dtype == np.uint8
    assert int(np.count_nonzero(mask)) > 1000
    assert result["centroid_px"] is not None
    cx, cy = result["centroid_px"]
    assert abs(cx - CROP_SIZE / 2) < 30
    assert 0 <= cy < CROP_SIZE


def test_measure_width_at_row_matches_synthetic_arm() -> None:
    image = _synthetic_arm_image()
    result = segment_arm(image, MIDPOINT, crop_size_px=CROP_SIZE)
    row = int(round(result["centroid_px"][1])) if result["centroid_px"] else CROP_SIZE // 2
    width = measure_width_at_row(result["mask"], row)
    assert width == pytest.approx(ARM_WIDTH_PX, rel=0.25, abs=15)


def test_measure_width_at_row_empty_or_oob() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    assert measure_width_at_row(mask, 10) == 0.0
    assert measure_width_at_row(mask, -1) == 0.0
    assert measure_width_at_row(mask, 99) == 0.0
    assert measure_width_at_row(None, 0) == 0.0
