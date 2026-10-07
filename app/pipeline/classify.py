"""Map estimated MUAC to a WHO community-screening risk band."""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional, TypedDict


class AgeCategory(str, Enum):
    """Valid age categories for MUAC classification.

    Only two groups are supported by published, globally standardised
    MUAC-only thresholds:

    * ``child_6_59m`` — WHO community-screening cutoffs for children
      6–59 months (SAM < 115 mm, MAM 115–124.9 mm, Normal ≥ 125 mm).
    * ``adult`` — Published adult/maternal MUAC screening scale used in
      resource-poor-setting nutrition screening, calibrated against BMI
      < 16.5 kg/m² (Severe) and < 18.5 kg/m² (Moderate).

    Ages 5–19 years are **not** supported: no single globally
    standardised MUAC-only cutoff exists for that bracket.  WHO
    recommends BMI-for-age z-scores there, which this system does not
    implement.
    """

    CHILD_6_59M = "child_6_59m"
    ADULT = "adult"


# ---------------------------------------------------------------------------
# Threshold band configurations
# ---------------------------------------------------------------------------

class _BandConfig(TypedDict):
    sam_threshold_mm: float    # MUAC < this  → severe band
    normal_threshold_mm: float  # MUAC >= this → normal band
    severe_label: str
    moderate_label: str
    normal_label: str


#: WHO community-screening cutoffs for children 6–59 months.
#: Source: WHO 2009 community-based management of severe acute malnutrition.
CHILD_6_59M_BANDS: _BandConfig = {
    "sam_threshold_mm": 115.0,
    "normal_threshold_mm": 125.0,
    "severe_label": "SAM",
    "moderate_label": "MAM",
    "normal_label": "Normal",
}

#: Published adult/maternal MUAC screening thresholds.
#: Severe  < 210 mm (21 cm)  — calibrated against BMI < 16.5 kg/m²
#: Moderate 210–229 mm       — calibrated against BMI 16.5–18.4 kg/m²
#: Normal  ≥ 230 mm (23 cm)
#: Source: Ferro-Luzzi & James (1996); Collins et al. (2000); used in
#: MSF/UNHCR field nutrition protocols.
ADULT_BANDS: _BandConfig = {
    "sam_threshold_mm": 210.0,
    "normal_threshold_mm": 230.0,
    "severe_label": "Severe",
    "moderate_label": "Moderate",
    "normal_label": "Normal",
}

# ---------------------------------------------------------------------------
# Legacy module-level constants — kept for backwards-compatible imports.
# These reflect CHILD_6_59M_BANDS only.
# ---------------------------------------------------------------------------

#: WHO MUAC cut-offs for children 6–59 months (community screening), in mm.
SAM_THRESHOLD_MM: float = CHILD_6_59M_BANDS["sam_threshold_mm"]
NORMAL_THRESHOLD_MM: float = CHILD_6_59M_BANDS["normal_threshold_mm"]

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
    age_category: str


def classify_risk_band(muac_mm: float, age_category: str) -> RiskBandResult:
    """Classify MUAC using the thresholds appropriate for *age_category*.

    Parameters
    ----------
    muac_mm:
        Estimated MUAC in millimetres.
    age_category:
        Must be exactly ``"child_6_59m"`` or ``"adult"``.  No default
        is provided because silently falling back to the wrong scale
        would misclassify people in an unsafe way.

    Returns
    -------
    dict with keys ``risk_band``, ``color``, ``muac_mm``, and
    ``age_category``, so the response is self-documenting.

    Raises
    ------
    ValueError
        If *age_category* is not one of the two supported values.

    **child_6_59m** (WHO 6–59 month community-screening thresholds):

    * ``"SAM"`` (red)    — MUAC < 115 mm
    * ``"MAM"`` (yellow) — 115 mm ≤ MUAC < 125 mm (includes 124.9 mm)
    * ``"Normal"`` (green) — MUAC ≥ 125 mm

    **adult** (published adult/maternal MUAC screening scale):

    * ``"Severe"`` (red)    — MUAC < 210 mm
    * ``"Moderate"`` (yellow) — 210 mm ≤ MUAC < 230 mm
    * ``"Normal"`` (green)  — MUAC ≥ 230 mm
    """
    # Validate age_category first — never infer or default.
    try:
        category = AgeCategory(age_category)
    except ValueError:
        raise ValueError(
            f"age_category must be 'child_6_59m' or 'adult' -- it cannot be inferred "
            f"automatically (got {age_category!r})"
        )

    value = float(muac_mm)

    if category is AgeCategory.CHILD_6_59M:
        cfg = CHILD_6_59M_BANDS
    else:
        cfg = ADULT_BANDS

    if value < cfg["sam_threshold_mm"]:
        band, color = cfg["severe_label"], "red"
    elif value < cfg["normal_threshold_mm"]:
        band, color = cfg["moderate_label"], "yellow"
    else:
        band, color = cfg["normal_label"], "green"

    return {
        "risk_band": band,
        "color": color,
        "muac_mm": value,
        "age_category": category.value,
    }


def classify_muac(muac_cm: float) -> Optional[str]:
    """Compatibility wrapper: WHO child band from a centimetre measurement.

    .. deprecated::
        Use :func:`classify_risk_band` with an explicit *age_category* instead.
    """
    if muac_cm is None or not math.isfinite(float(muac_cm)):
        return None
    return classify_risk_band(float(muac_cm) * 10.0, "child_6_59m")["risk_band"]


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
