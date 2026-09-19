"""Convert measured arm width into a MUAC estimate (millimetres)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, TypedDict, Union

import numpy as np
from sklearn.linear_model import Ridge

try:
    import joblib
except ImportError:  # pragma: no cover - joblib ships with scikit-learn
    from sklearn.externals import joblib  # type: ignore[no-redef]

DEFAULT_SHAPE_FACTOR = 1.28
PathLike = Union[str, Path]


class MuacEstimateResult(TypedDict):
    """Result of :func:`estimate_muac`."""

    width_mm: Optional[float]
    muac_estimate_mm: Optional[float]
    shape_factor_used: float


def estimate_muac(
    arm_width_px: float,
    scale_px_per_mm: float,
    shape_factor: float = DEFAULT_SHAPE_FACTOR,
) -> MuacEstimateResult:
    """Estimate MUAC from apparent arm width and a marker scale.

    Circumference is modelled as an ellipse whose minor/major geometry is
    summarised by ``shape_factor`` (1.0 would be a circle of diameter
    ``width_mm``; the default 1.28 is a typical soft-tissue correction).

        width_mm = arm_width_px / scale_px_per_mm
        muac_mm  = pi * shape_factor * width_mm

    Args:
        arm_width_px: Apparent arm width in pixels at the MUAC site.
        scale_px_per_mm: Pixels per millimetre from the reference marker.
        shape_factor: Ellipse / tissue correction (default 1.28).

    Returns:
        Dict with ``width_mm``, ``muac_estimate_mm``, ``shape_factor_used``.
        Linear measures are None if the scale is missing or non-positive.
    """
    used = float(shape_factor)
    if (
        scale_px_per_mm is None
        or not math.isfinite(float(scale_px_per_mm))
        or float(scale_px_per_mm) <= 0
        or arm_width_px is None
        or not math.isfinite(float(arm_width_px))
    ):
        return {
            "width_mm": None,
            "muac_estimate_mm": None,
            "shape_factor_used": used,
        }

    width_mm = float(arm_width_px) / float(scale_px_per_mm)
    muac_mm = math.pi * used * width_mm
    return {
        "width_mm": width_mm,
        "muac_estimate_mm": muac_mm,
        "shape_factor_used": used,
    }


def estimate_muac_cm(
    arm_width_px: float,
    scale_px_per_mm: float,
    shape_factor: float = DEFAULT_SHAPE_FACTOR,
) -> Optional[float]:
    """Compatibility helper: MUAC in centimetres, or None if scale is invalid."""
    result = estimate_muac(arm_width_px, scale_px_per_mm, shape_factor)
    if result["muac_estimate_mm"] is None:
        return None
    return result["muac_estimate_mm"] / 10.0


class CalibrationCorrector:
    """Learns a linear map from baseline MUAC estimates to tape measurements.

    If :meth:`fit` / :meth:`load` has not been called, :meth:`predict` is a
    pass-through so the pipeline still runs before calibration data exists.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.model: Optional[Ridge] = None
        self.is_fitted: bool = False

    def fit(
        self,
        baseline_estimates: list[float],
        tape_measurements: list[float],
    ) -> "CalibrationCorrector":
        """Train Ridge regression on paired (estimate, tape) millimetre values."""
        x = np.asarray(baseline_estimates, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(tape_measurements, dtype=np.float64).reshape(-1)
        if x.shape[0] != y.shape[0]:
            raise ValueError("baseline_estimates and tape_measurements must have the same length.")
        if x.shape[0] < 2:
            raise ValueError("Need at least two paired samples to fit a calibration model.")
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(x, y)
        self.is_fitted = True
        return self

    def predict(self, baseline_estimate: float) -> float:
        """Return a corrected MUAC in millimetres, or the input if untrained."""
        value = float(baseline_estimate)
        if not self.is_fitted or self.model is None:
            return value
        predicted = self.model.predict(np.array([[value]], dtype=np.float64))
        return float(predicted[0])

    def save(self, path: PathLike) -> None:
        """Persist the corrector with joblib."""
        payload = {
            "alpha": self.alpha,
            "is_fitted": self.is_fitted,
            "model": self.model,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: PathLike) -> "CalibrationCorrector":
        """Load a corrector previously written by :meth:`save`."""
        payload = joblib.load(path)
        instance = cls(alpha=float(payload.get("alpha", 1.0)))
        instance.model = payload.get("model")
        instance.is_fitted = bool(payload.get("is_fitted", instance.model is not None))
        return instance
