"""Synthetic ArUco tests for reference-marker scale estimation."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.pipeline.reference_detect import (
    detect_reference_marker,
    draw_marker_debug,
)


MARKER_ID = 0
MARKER_SIDE_PX = 200
MARKER_REAL_SIZE_MM = 40.0
CANVAS_H, CANVAS_W = 480, 640
PASTE_Y, PASTE_X = 80, 120


def _aruco_dictionary():
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)


def _generate_marker_image(side_px: int) -> np.ndarray:
    dictionary = _aruco_dictionary()
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, MARKER_ID, side_px)
    return cv2.aruco.drawMarker(dictionary, MARKER_ID, side_px)


def _synthetic_scene() -> np.ndarray:
    """White canvas with a known-size ArUco marker pasted at a fixed offset."""
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 255, dtype=np.uint8)
    marker = _generate_marker_image(MARKER_SIDE_PX)
    if marker.ndim == 2:
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    else:
        marker_bgr = marker
    canvas[
        PASTE_Y : PASTE_Y + MARKER_SIDE_PX,
        PASTE_X : PASTE_X + MARKER_SIDE_PX,
    ] = marker_bgr
    return canvas


def test_detect_synthetic_marker_scale() -> None:
    image = _synthetic_scene()
    result = detect_reference_marker(image, MARKER_REAL_SIZE_MM)

    assert result["detected"] is True
    assert result["scale_px_per_mm"] is not None
    expected_scale = MARKER_SIDE_PX / MARKER_REAL_SIZE_MM
    assert result["scale_px_per_mm"] == pytest.approx(expected_scale, rel=0.05)
    assert result["marker_corners"] is not None
    assert np.asarray(result["marker_corners"]).reshape(-1, 2).shape[0] == 4
    assert result["confidence"] > 0.9


def test_no_marker_returns_undetected() -> None:
    blank = np.full((240, 320, 3), 255, dtype=np.uint8)
    result = detect_reference_marker(blank, MARKER_REAL_SIZE_MM)

    assert result["detected"] is False
    assert result["scale_px_per_mm"] is None
    assert result["marker_corners"] is None
    assert result["confidence"] == 0.0


@pytest.mark.parametrize("image", [None, np.array([]), np.zeros((0, 0, 3), dtype=np.uint8)])
def test_none_or_empty_image_does_not_raise(image: np.ndarray | None) -> None:
    result = detect_reference_marker(image, MARKER_REAL_SIZE_MM)
    assert result["detected"] is False
    assert result["scale_px_per_mm"] is None


def test_draw_marker_debug_annotates_copy() -> None:
    image = _synthetic_scene()
    result = detect_reference_marker(image, MARKER_REAL_SIZE_MM)
    annotated = draw_marker_debug(image, detection=result)

    assert annotated.shape[:2] == image.shape[:2]
    assert annotated.ndim == 3
    assert not np.array_equal(annotated, image)
