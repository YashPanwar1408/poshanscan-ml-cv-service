"""MUAC estimation pipeline stages."""

from app.pipeline.classify import classify_muac, classify_risk_band, compute_confidence_score
from app.pipeline.estimate import CalibrationCorrector, estimate_muac, estimate_muac_cm
from app.pipeline.pose_localize import locate_arm_midpoint, localize_measurement_site
from app.pipeline.reference_detect import (
    detect_reference_marker,
    detect_reference_scale,
    draw_marker_debug,
)
from app.pipeline.segment import measure_width_at_row, segment_arm

__all__ = [
    "CalibrationCorrector",
    "classify_muac",
    "classify_risk_band",
    "compute_confidence_score",
    "detect_reference_marker",
    "detect_reference_scale",
    "draw_marker_debug",
    "estimate_muac",
    "estimate_muac_cm",
    "locate_arm_midpoint",
    "localize_measurement_site",
    "measure_width_at_row",
    "segment_arm",
]
