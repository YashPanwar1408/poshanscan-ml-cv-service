"""Map estimated MUAC to a WHO community-screening risk band."""

from __future__ import annotations

import math
from typing import Optional, TypedDict

# WHO MUAC cut-offs for children 6–59 months (community screening), in mm.
SAM_THRESHOLD_MM: float = 115.0
NORMAL_THRESHOLD_MM: float = 125.0

# Kept for callers that still think in centimetres.
SAM_THRESHOLD_CM: float = SAM_THRESHOLD_MM / 10.0
MAM_THRESHOLD_CM: float = NORMAL_THRESHOLD_MM / 10.0

# Weights for :func:`compute_confidence_score` (should sum to 1.0).
WEIGHT_REFERENCE: float = 0.40
WEIGHT_POSE: float = 0.35
WEIGHT_SEGMENTATION: float = 0.25


class RiskBandResult(TypedDict):
    """Result of :func:`classify_risk_band`."""

    risk_band: str
    color: str
    muac_mm: float


def classify_risk_band(muac_mm: float) -> RiskBandResult:
    """Classify MUAC using WHO 6–59 month community-screening thresholds.

    * ``"SAM"`` (red) — MUAC < 115 mm
    * ``"MAM"`` (yellow) — 115 mm ≤ MUAC < 125 mm (includes 124.9 mm)
    * ``"Normal"`` (green) — MUAC ≥ 125 mm
    """
    value = float(muac_mm)
    if value < SAM_THRESHOLD_MM:
        band, color = "SAM", "red"
    elif value < NORMAL_THRESHOLD_MM:
        band, color = "MAM", "yellow"
    else:
        band, color = "Normal", "green"
    return {"risk_band": band, "color": color, "muac_mm": value}


def classify_muac(muac_cm: float) -> Optional[str]:
    """Compatibility wrapper: WHO band from a centimetre measurement."""
    if muac_cm is None or not math.isfinite(float(muac_cm)):
        return None
    return classify_risk_band(float(muac_cm) * 10.0)["risk_band"]


def compute_confidence_score(
    reference_confidence: float,
    pose_confidence: float,
    segmentation_quality: float,
) -> float:
    """Combine pipeline confidences into a single score in [0, 1].

    Weighted average using :data:`WEIGHT_REFERENCE`, :data:`WEIGHT_POSE`,
    and :data:`WEIGHT_SEGMENTATION`. Each input is clipped to [0, 1] first.
    """

    def _clip01(value: float) -> float:
        return float(min(1.0, max(0.0, float(value))))

    score = (
        WEIGHT_REFERENCE * _clip01(reference_confidence)
        + WEIGHT_POSE * _clip01(pose_confidence)
        + WEIGHT_SEGMENTATION * _clip01(segmentation_quality)
    )
    return float(min(1.0, max(0.0, score)))
