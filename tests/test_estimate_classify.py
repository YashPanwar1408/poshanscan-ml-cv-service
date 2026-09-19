"""Unit tests for MUAC estimation, calibration, and WHO risk bands."""

from __future__ import annotations

import math

import pytest

from app.pipeline.classify import (
    WEIGHT_POSE,
    WEIGHT_REFERENCE,
    WEIGHT_SEGMENTATION,
    classify_risk_band,
    compute_confidence_score,
)
from app.pipeline.estimate import CalibrationCorrector, estimate_muac


def test_estimate_muac_ellipse_formula() -> None:
    arm_width_px = 100.0
    scale_px_per_mm = 5.0
    shape_factor = 1.28
    result = estimate_muac(arm_width_px, scale_px_per_mm, shape_factor)

    width_mm = arm_width_px / scale_px_per_mm
    assert result["width_mm"] == pytest.approx(width_mm)
    assert result["muac_estimate_mm"] == pytest.approx(math.pi * shape_factor * width_mm)
    assert result["shape_factor_used"] == shape_factor


def test_estimate_muac_invalid_scale() -> None:
    result = estimate_muac(100.0, 0.0)
    assert result["width_mm"] is None
    assert result["muac_estimate_mm"] is None


@pytest.mark.parametrize(
    ("muac_mm", "band", "color"),
    [
        (114.9, "SAM", "red"),
        (115.0, "MAM", "yellow"),
        (120.0, "MAM", "yellow"),
        (124.9, "MAM", "yellow"),
        (125.0, "Normal", "green"),
        (140.0, "Normal", "green"),
    ],
)
def test_classify_risk_band_boundaries(muac_mm: float, band: str, color: str) -> None:
    result = classify_risk_band(muac_mm)
    assert result["risk_band"] == band
    assert result["color"] == color
    assert result["muac_mm"] == pytest.approx(muac_mm)


def test_compute_confidence_score_weighted_average() -> None:
    score = compute_confidence_score(1.0, 0.0, 0.0)
    assert score == pytest.approx(WEIGHT_REFERENCE)
    mixed = compute_confidence_score(0.5, 0.5, 0.5)
    assert mixed == pytest.approx(
        WEIGHT_REFERENCE * 0.5 + WEIGHT_POSE * 0.5 + WEIGHT_SEGMENTATION * 0.5
    )
    assert 0.0 <= compute_confidence_score(2.0, -1.0, 0.3) <= 1.0


def test_calibration_corrector_passthrough_when_untrained() -> None:
    corrector = CalibrationCorrector()
    assert corrector.is_fitted is False
    assert corrector.predict(137.4) == pytest.approx(137.4)


def test_calibration_corrector_fit_predict_and_roundtrip(tmp_path) -> None:
    baseline = [100.0, 110.0, 120.0, 130.0]
    tape = [105.0, 115.0, 125.0, 135.0]
    corrector = CalibrationCorrector(alpha=0.01)
    corrector.fit(baseline, tape)
    corrected = corrector.predict(120.0)
    assert corrected == pytest.approx(125.0, abs=1.5)

    path = tmp_path / "calib.joblib"
    corrector.save(path)
    loaded = CalibrationCorrector.load(path)
    assert loaded.predict(120.0) == pytest.approx(corrected)
