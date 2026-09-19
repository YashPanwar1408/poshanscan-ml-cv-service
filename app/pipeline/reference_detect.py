"""Detect an ArUco reference marker and compute pixels-per-millimetre."""

from __future__ import annotations

from typing import Any, Optional, TypedDict

import cv2
import numpy as np

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
DEFAULT_MARKER_SIZE_MM: float = 50.0


class MarkerDetectionResult(TypedDict):
    """Result of :func:`detect_reference_marker`."""

    detected: bool
    scale_px_per_mm: Optional[float]
    marker_corners: Optional[np.ndarray]
    confidence: float
    marker_id: Optional[int]


def _empty_result() -> MarkerDetectionResult:
    return {
        "detected": False,
        "scale_px_per_mm": None,
        "marker_corners": None,
        "confidence": 0.0,
        "marker_id": None,
    }


def _is_empty_image(image: Any) -> bool:
    if image is None:
        return True
    array = np.asarray(image)
    return array.size == 0 or array.ndim < 2 or array.shape[0] == 0 or array.shape[1] == 0


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _get_dictionary() -> Any:
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    return cv2.aruco.Dictionary_get(ARUCO_DICT_ID)


def _detect_aruco(gray: np.ndarray) -> tuple[list[np.ndarray], Optional[np.ndarray]]:
    dictionary = _get_dictionary()
    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _rejected = detector.detectMarkers(gray)
    else:
        parameters = (
            cv2.aruco.DetectorParameters_create()
            if hasattr(cv2.aruco, "DetectorParameters_create")
            else cv2.aruco.DetectorParameters()
        )
        corners, ids, _rejected = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=parameters
        )
    return corners or [], ids


def _quad_side_lengths(corners: np.ndarray) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    return np.array(
        [
            np.linalg.norm(pts[1] - pts[0]),
            np.linalg.norm(pts[2] - pts[1]),
            np.linalg.norm(pts[3] - pts[2]),
            np.linalg.norm(pts[0] - pts[3]),
        ],
        dtype=np.float64,
    )


def _square_confidence(corners: np.ndarray) -> float:
    """Confidence proxy from how equal the four marker sides are.

    A fronto-parallel square has nearly identical side lengths (confidence ~ 1).
    Strong perspective distortion stretches sides unequally (confidence closer to 0).
    """
    sides = _quad_side_lengths(corners)
    longest = float(np.max(sides))
    shortest = float(np.min(sides))
    if longest <= 1e-6:
        return 0.0
    return float(np.clip(shortest / longest, 0.0, 1.0))


def _mean_side_px(corners: np.ndarray) -> float:
    return float(np.mean(_quad_side_lengths(corners)))


def detect_reference_marker(
    image: np.ndarray,
    marker_real_size_mm: float,
) -> MarkerDetectionResult:
    """Detect a DICT_4X4_50 ArUco marker and compute pixels per millimetre.

    Args:
        image: Input image as a numpy array in BGR (or grayscale). ``None``
            and empty arrays are treated as a miss, not an exception.
        marker_real_size_mm: Physical edge length of the printed marker in
            millimetres. Must be positive.

    Returns:
        Dict with ``detected``, ``scale_px_per_mm``, ``marker_corners``,
        and ``confidence``. On failure ``detected`` is False and
        ``scale_px_per_mm`` is None.
    """
    if _is_empty_image(image):
        return _empty_result()
    if marker_real_size_mm is None or marker_real_size_mm <= 0:
        return _empty_result()

    gray = _to_grayscale(np.asarray(image))
    corners_list, ids = _detect_aruco(gray)
    if not corners_list:
        return _empty_result()

    best_corners: Optional[np.ndarray] = None
    best_confidence = -1.0
    best_id: Optional[int] = None
    for index, marker_corners in enumerate(corners_list):
        confidence = _square_confidence(marker_corners)
        if confidence > best_confidence:
            best_confidence = confidence
            best_corners = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
            if ids is not None and len(ids) > index:
                best_id = int(ids[index][0]) if np.ndim(ids[index]) else int(ids[index])

    assert best_corners is not None
    mean_side_px = _mean_side_px(best_corners)
    scale_px_per_mm = mean_side_px / float(marker_real_size_mm)

    return {
        "detected": True,
        "scale_px_per_mm": float(scale_px_per_mm),
        "marker_corners": best_corners,
        "confidence": float(best_confidence),
        "marker_id": best_id,
    }


def detect_reference_scale(
    image: np.ndarray,
    marker_size_mm: float = DEFAULT_MARKER_SIZE_MM,
) -> tuple[Optional[float], Optional[np.ndarray]]:
    """Compatibility wrapper: millimetres-per-pixel and marker corners.

    Inverse of ``scale_px_per_mm`` from :func:`detect_reference_marker`.
    """
    result = detect_reference_marker(image, marker_size_mm)
    if not result["detected"] or not result["scale_px_per_mm"]:
        return None, None
    mm_per_px = 1.0 / result["scale_px_per_mm"]
    return mm_per_px, result["marker_corners"]


def draw_marker_debug(
    image: np.ndarray,
    marker_corners: Optional[np.ndarray] = None,
    marker_ids: Optional[np.ndarray] = None,
    detection: Optional[MarkerDetectionResult] = None,
) -> np.ndarray:
    """Draw detected marker outlines (and IDs, if given) for debugging.

    Args:
        image: BGR (or grayscale) image to annotate. Copied; not mutated.
        marker_corners: Raw corners, shape ``(4, 2)`` or OpenCV ``(1, 4, 2)``,
            or a list of such arrays.
        marker_ids: Optional marker IDs, shape ``(N, 1)``, matching OpenCV.
        detection: If provided, ``marker_corners`` is taken from this result
            when ``marker_corners`` is omitted.

    Returns:
        Annotated copy of ``image``. If there is nothing to draw, a copy of
        the original image is returned (or an empty array if ``image`` is
        None/empty).
    """
    if _is_empty_image(image):
        return np.zeros((1, 1, 3), dtype=np.uint8)

    annotated = np.asarray(image).copy()
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

    corners = marker_corners
    if corners is None and detection is not None:
        corners = detection.get("marker_corners")
        if marker_ids is None and detection.get("marker_id") is not None:
            marker_ids = np.array([[detection["marker_id"]]], dtype=np.int32)
    if corners is None:
        return annotated

    if isinstance(corners, np.ndarray) and corners.ndim == 2:
        corners_for_draw: list[np.ndarray] = [
            np.asarray(corners, dtype=np.float32).reshape(1, 4, 2)
        ]
    elif isinstance(corners, np.ndarray) and corners.ndim == 3:
        corners_for_draw = [np.asarray(corners, dtype=np.float32)]
    else:
        corners_for_draw = [
            np.asarray(c, dtype=np.float32).reshape(1, 4, 2) for c in corners
        ]

    ids = None if marker_ids is None else np.asarray(marker_ids)
    cv2.aruco.drawDetectedMarkers(annotated, corners_for_draw, ids)
    return annotated
