"""Shared MUAC pipeline used by the HTTP API and the evaluation script."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from app.pipeline.classify import classify_risk_band, compute_confidence_score
from app.pipeline.estimate import CalibrationCorrector, estimate_muac
from app.pipeline.pose_localize import locate_arm_midpoint, locate_from_marker
from app.pipeline.reference_detect import detect_reference_marker
from app.pipeline.segment import (
    fill_polygon_on_mask,
    measure_width_perpendicular,
    segment_arm,
)

logger = logging.getLogger("poshanscan-cv")

MIN_FOREGROUND_PX = 100
MIN_WIDTH_PX = 5.0
_calibrator = CalibrationCorrector()


def locate_either_arm(image: np.ndarray) -> dict[str, Any]:
    left = locate_arm_midpoint(image, side="left")
    if left.get("detected"):
        return left
    return locate_arm_midpoint(image, side="right")


def _crop_size_from_marker(image: np.ndarray, marker_corners: Optional[np.ndarray]) -> int:
    height, width = image.shape[:2]
    cap = int(min(height, width))
    if marker_corners is None:
        return int(np.clip(400, 240, cap))
    pts = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
    sides = [
        float(np.linalg.norm(pts[1] - pts[0])),
        float(np.linalg.norm(pts[2] - pts[1])),
        float(np.linalg.norm(pts[3] - pts[2])),
        float(np.linalg.norm(pts[0] - pts[3])),
    ]
    mean_side = float(np.mean(sides))
    return int(np.clip(mean_side * 4.5, 280, cap))


def _segmentation_ok(mask: np.ndarray, width_px: float) -> bool:
    if mask is None or mask.size == 0:
        return False
    return int(np.count_nonzero(mask)) >= MIN_FOREGROUND_PX and width_px >= MIN_WIDTH_PX


def _segmentation_quality(mask: np.ndarray) -> float:
    if mask is None or mask.size == 0:
        return 0.0
    return float(np.clip(np.count_nonzero(mask) / float(mask.size), 0.0, 1.0))


def run_muac_pipeline(
    image: np.ndarray,
    reference_size_mm: float,
) -> dict[str, Any]:
    """Run reference → pose (or marker fallback) → segment → estimate → classify.

    Returns a dict with ``ok`` True on success, or ``ok`` False and ``error`` set.
    """
    reference = detect_reference_marker(image, float(reference_size_mm))
    logger.info(
        "reference_detect: detected=%s scale_px_per_mm=%s conf=%.3f marker_id=%s",
        reference["detected"],
        reference.get("scale_px_per_mm"),
        reference.get("confidence") or 0.0,
        reference.get("marker_id"),
    )
    if not reference["detected"] or not reference.get("scale_px_per_mm"):
        return {
            "ok": False,
            "error": "reference_object_not_detected",
            "message": "No ArUco reference marker was found in the image.",
        }

    pose = locate_either_arm(image)
    logger.info(
        "pose_localize: detected=%s midpoint=%s angle=%s conf=%.3f",
        pose.get("detected"),
        pose.get("midpoint_px"),
        pose.get("arm_angle_degrees"),
        pose.get("confidence") or 0.0,
    )
    if not pose.get("detected") or pose.get("midpoint_px") is None:
        pose = locate_from_marker(image, reference.get("marker_corners"))
        logger.info(
            "pose_localize: MediaPipe missed; marker fallback detected=%s midpoint=%s angle=%s",
            pose.get("detected"),
            pose.get("midpoint_px"),
            pose.get("arm_angle_degrees"),
        )
    if not pose.get("detected") or pose.get("midpoint_px") is None:
        return {
            "ok": False,
            "error": "arm_not_detected",
            "message": "Could not localise a shoulder–elbow pair for MUAC measurement.",
        }

    corners = reference.get("marker_corners")
    crop_size = _crop_size_from_marker(image, corners)
    segmented = segment_arm(image, pose["midpoint_px"], crop_size_px=crop_size)
    mask = segmented["mask"]
    origin = segmented["crop_origin_px"]
    if corners is not None:
        mask = fill_polygon_on_mask(mask, corners, origin)

    centroid = segmented.get("centroid_px")
    if centroid is None:
        h, w = mask.shape[:2]
        centroid = (w / 2.0, h / 2.0)
    angle = float(pose.get("arm_angle_degrees") if pose.get("arm_angle_degrees") is not None else 90.0)
    width_px = measure_width_perpendicular(mask, centroid, angle)
    seg_ok = _segmentation_ok(mask, width_px)
    seg_quality = _segmentation_quality(mask)
    logger.info(
        "segment: crop=%s centroid=%s width_px=%.2f foreground=%d ok=%s quality=%.3f",
        crop_size,
        centroid,
        width_px,
        int(np.count_nonzero(mask)),
        seg_ok,
        seg_quality,
    )

    estimated = estimate_muac(width_px, float(reference["scale_px_per_mm"]))
    baseline_mm = estimated.get("muac_estimate_mm")
    if baseline_mm is None:
        baseline_mm = 0.0
    muac_mm = _calibrator.predict(float(baseline_mm))
    classified = classify_risk_band(muac_mm)
    confidence = compute_confidence_score(
        float(reference.get("confidence") or 0.0),
        float(pose.get("confidence") or 0.0),
        seg_quality,
    )
    logger.info(
        "estimate/classify: width_mm=%s muac_mm=%.3f band=%s conf=%.3f",
        estimated.get("width_mm"),
        muac_mm,
        classified["risk_band"],
        confidence,
    )
    return {
        "ok": True,
        "error": None,
        "message": "",
        "muac_estimate_mm": float(muac_mm),
        "risk_band": classified["risk_band"],
        "confidence_score": confidence,
        "segmentation_ok": seg_ok,
        "arm_detected": True,
        "reference_detected": True,
    }
