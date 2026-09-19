"""Localise the mid-upper-arm measurement site with MediaPipe Pose."""

from __future__ import annotations

import threading
from typing import Any, Optional, TypedDict

import cv2
import numpy as np

# MediaPipe Pose landmark indices (subject's own left/right).
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_ELBOW = 13
_RIGHT_ELBOW = 14

_MIN_LANDMARK_VISIBILITY = 0.3

_pose_lock = threading.Lock()
_pose_model: Any = None


class ArmMidpointResult(TypedDict):
    """Result of :func:`locate_arm_midpoint`."""

    detected: bool
    midpoint_px: Optional[tuple[float, float]]
    shoulder_px: Optional[tuple[float, float]]
    elbow_px: Optional[tuple[float, float]]
    arm_angle_degrees: Optional[float]
    confidence: float


def _empty_result() -> ArmMidpointResult:
    return {
        "detected": False,
        "midpoint_px": None,
        "shoulder_px": None,
        "elbow_px": None,
        "arm_angle_degrees": None,
        "confidence": 0.0,
    }


def _is_empty_image(image: Any) -> bool:
    if image is None:
        return True
    array = np.asarray(image)
    return array.size == 0 or array.ndim < 2 or array.shape[0] == 0 or array.shape[1] == 0


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _get_pose() -> Any:
    """Lazy singleton so the Pose graph is not rebuilt on every call."""
    global _pose_model
    if _pose_model is not None:
        return _pose_model
    with _pose_lock:
        if _pose_model is None:
            import mediapipe as mp

            _pose_model = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
            )
    return _pose_model


def _landmark_to_px(landmark: Any, width: int, height: int) -> tuple[float, float]:
    return float(landmark.x * width), float(landmark.y * height)


def _landmark_visibility(landmark: Any) -> float:
    visibility = getattr(landmark, "visibility", None)
    if visibility is None:
        visibility = getattr(landmark, "presence", 0.0)
    return float(visibility)


def midpoint_px(
    shoulder_px: tuple[float, float],
    elbow_px: tuple[float, float],
) -> tuple[float, float]:
    """Pixel midpoint between shoulder and elbow (MUAC tape location)."""
    return (
        (shoulder_px[0] + elbow_px[0]) / 2.0,
        (shoulder_px[1] + elbow_px[1]) / 2.0,
    )


def arm_axis_angle_degrees(
    shoulder_px: tuple[float, float],
    elbow_px: tuple[float, float],
) -> float:
    """Angle of the shoulder→elbow axis in degrees (OpenCV image coords).

    0° points right, 90° points down. Downstream width measurement should
    be taken perpendicular to this axis, not along image rows.
    """
    dx = elbow_px[0] - shoulder_px[0]
    dy = elbow_px[1] - shoulder_px[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def locate_arm_midpoint(
    image: np.ndarray,
    side: str = "left",
) -> ArmMidpointResult:
    """Detect shoulder/elbow and the mid-upper-arm point for one arm.

    Args:
        image: BGR (or grayscale) numpy array.
        side: ``"left"`` or ``"right"`` from the subject's perspective
            (MediaPipe Pose convention).

    Returns:
        Dict with ``detected``, ``midpoint_px``, ``shoulder_px``,
        ``elbow_px``, ``arm_angle_degrees``, and ``confidence``.
        On failure ``detected`` is False and coordinate fields are None.
    """
    if _is_empty_image(image):
        return _empty_result()

    side_key = (side or "left").strip().lower()
    if side_key not in {"left", "right"}:
        return _empty_result()

    bgr = _to_bgr(np.asarray(image))
    height, width = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    try:
        results = _get_pose().process(rgb)
    except Exception:
        return _empty_result()

    if results.pose_landmarks is None:
        return _empty_result()

    landmarks = results.pose_landmarks.landmark
    if side_key == "left":
        shoulder_lm = landmarks[_LEFT_SHOULDER]
        elbow_lm = landmarks[_LEFT_ELBOW]
    else:
        shoulder_lm = landmarks[_RIGHT_SHOULDER]
        elbow_lm = landmarks[_RIGHT_ELBOW]

    shoulder_vis = _landmark_visibility(shoulder_lm)
    elbow_vis = _landmark_visibility(elbow_lm)
    confidence = (shoulder_vis + elbow_vis) / 2.0

    if confidence < _MIN_LANDMARK_VISIBILITY:
        return {
            "detected": False,
            "midpoint_px": None,
            "shoulder_px": None,
            "elbow_px": None,
            "arm_angle_degrees": None,
            "confidence": float(confidence),
        }

    shoulder_px = _landmark_to_px(shoulder_lm, width, height)
    elbow_px = _landmark_to_px(elbow_lm, width, height)
    mid = midpoint_px(shoulder_px, elbow_px)
    angle = arm_axis_angle_degrees(shoulder_px, elbow_px)

    return {
        "detected": True,
        "midpoint_px": mid,
        "shoulder_px": shoulder_px,
        "elbow_px": elbow_px,
        "arm_angle_degrees": angle,
        "confidence": float(confidence),
    }


def localize_measurement_site(
    image: np.ndarray,
    side: str = "left",
) -> Optional[tuple[float, float]]:
    """Compatibility wrapper returning only the MUAC midpoint, or None."""
    result = locate_arm_midpoint(image, side=side)
    return result["midpoint_px"] if result["detected"] else None
