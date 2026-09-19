"""Unit tests for evaluation metrics (no camera / MediaPipe required)."""

from __future__ import annotations

import math

from scripts.evaluate import compute_metrics


def test_compute_metrics_known_values() -> None:
    preds = [120.0, 130.0, 110.0]
    tapes = [125.0, 125.0, 110.0]
    metrics = compute_metrics(preds, tapes)

    errors = [120 - 125, 130 - 125, 110 - 110]  # -5, +5, 0
    assert metrics["n"] == 3
    assert metrics["mae"] == 10.0 / 3.0
    assert metrics["rmse"] == math.sqrt((25 + 25 + 0) / 3.0)
    assert metrics["bias"] == sum(errors) / 3.0
    # bands: 120 MAM vs 125 Normal; 130 Normal vs 125 Normal; 110 SAM vs 110 SAM
    assert metrics["band_agreement"] == 2 / 3


def test_compute_metrics_empty() -> None:
    metrics = compute_metrics([], [])
    assert metrics["n"] == 0
    assert math.isnan(metrics["mae"])
