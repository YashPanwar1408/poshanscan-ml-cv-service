"""Offline accuracy evaluation against tape-measured MUAC.

Reads a CSV of labelled photos, runs the CV pipeline (no HTTP), and reports
MAE, RMSE, bias, and WHO screening-band agreement.

Example
-------
    python scripts/evaluate.py --csv data/eval.csv --output results/eval.csv

    python scripts/evaluate.py --csv data/eval.csv --with-correction \\
        --model app/models/calibration.joblib --output results/eval_corrected.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline.classify import classify_risk_band  # noqa: E402
from app.pipeline.estimate import CalibrationCorrector, estimate_muac  # noqa: E402
from app.pipeline.pose_localize import locate_arm_midpoint  # noqa: E402
from app.pipeline.reference_detect import detect_reference_marker  # noqa: E402
from app.pipeline.segment import measure_width_at_row, segment_arm  # noqa: E402

REQUIRED_COLUMNS = ("image_path", "reference_size_mm", "tape_measurement_mm")


def load_image_bgr(path: Path) -> Optional[np.ndarray]:
    """Load a BGR image; ``fromfile``+``imdecode`` handles Unicode Windows paths."""
    if not path.is_file():
        return None
    payload = np.fromfile(str(path), dtype=np.uint8)
    if payload.size == 0:
        return None
    return cv2.imdecode(payload, cv2.IMREAD_COLOR)


def locate_either_arm(image: np.ndarray) -> dict[str, Any]:
    left = locate_arm_midpoint(image, side="left")
    if left.get("detected"):
        return left
    return locate_arm_midpoint(image, side="right")


def run_baseline_pipeline(
    image: np.ndarray,
    reference_size_mm: float,
) -> tuple[Optional[float], str]:
    """Return ``(baseline_muac_mm, status)``. status is ``ok`` or an error code."""
    reference = detect_reference_marker(image, float(reference_size_mm))
    if not reference.get("detected") or not reference.get("scale_px_per_mm"):
        return None, "reference_object_not_detected"

    pose = locate_either_arm(image)
    if not pose.get("detected") or pose.get("midpoint_px") is None:
        return None, "arm_not_detected"

    segmented = segment_arm(image, pose["midpoint_px"])
    mask = segmented["mask"]
    centroid = segmented.get("centroid_px")
    row = int(round(centroid[1])) if centroid is not None else mask.shape[0] // 2
    width_px = measure_width_at_row(mask, row)
    estimated = estimate_muac(width_px, float(reference["scale_px_per_mm"]))
    muac_mm = estimated.get("muac_estimate_mm")
    if muac_mm is None:
        return None, "estimation_failed"
    return float(muac_mm), "ok"


def compute_metrics(
    predictions_mm: Sequence[float],
    tape_mm: Sequence[float],
) -> dict[str, float]:
    """MAE, RMSE, bias (mean signed error: pred − tape), band agreement rate."""
    preds = np.asarray(predictions_mm, dtype=np.float64)
    tapes = np.asarray(tape_mm, dtype=np.float64)
    if preds.size == 0 or preds.size != tapes.size:
        return {
            "n": 0.0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "band_agreement": float("nan"),
        }

    errors = preds - tapes
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    bias = float(np.mean(errors))
    pred_bands = [classify_risk_band(float(p))["risk_band"] for p in preds]
    tape_bands = [classify_risk_band(float(t))["risk_band"] for t in tapes]
    agreement = float(sum(a == b for a, b in zip(pred_bands, tape_bands)) / len(pred_bands))
    return {
        "n": float(preds.size),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "band_agreement": agreement,
    }


def format_metrics(title: str, metrics: dict[str, float]) -> str:
    n = int(metrics["n"])
    if n == 0:
        return f"{title}\n  (no successful predictions)"
    return (
        f"{title}  (n={n})\n"
        f"  MAE               {metrics['mae']:.3f} mm\n"
        f"  RMSE              {metrics['rmse']:.3f} mm\n"
        f"  Bias (pred-tape)  {metrics['bias']:.3f} mm\n"
        f"  Band agreement    {100.0 * metrics['band_agreement']:.1f}%"
    )


def resolve_image_path(raw: str, csv_dir: Path) -> Path:
    path = Path(raw)
    if path.is_file():
        return path
    candidate = csv_dir / raw
    if candidate.is_file():
        return candidate
    return ROOT / raw


def read_eval_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV has no header: {csv_path}")
        columns = {name.strip() for name in reader.fieldnames}
        missing = [col for col in REQUIRED_COLUMNS if col not in columns]
        if missing:
            raise SystemExit(
                f"CSV is missing required columns {missing}. "
                f"Expected: {', '.join(REQUIRED_COLUMNS)}"
            )
        return [{key.strip(): (value or "").strip() for key, value in row.items()} for row in reader]


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MUAC pipeline accuracy against tape measurements.",
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="CSV with columns image_path, reference_size_mm, tape_measurement_mm.",
    )
    parser.add_argument(
        "--with-correction",
        action="store_true",
        help="Also score predictions after CalibrationCorrector (requires --model).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to a joblib CalibrationCorrector saved by CalibrationCorrector.save().",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_results.csv"),
        help="Where to write the per-image results CSV (default: evaluation_results.csv).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    corrector: Optional[CalibrationCorrector] = None
    if args.with_correction:
        if args.model is None:
            print("--with-correction requires --model <path>", file=sys.stderr)
            return 1
        model_path = args.model.resolve()
        if not model_path.is_file():
            print(f"Calibration model not found: {model_path}", file=sys.stderr)
            return 1
        corrector = CalibrationCorrector.load(model_path)
        if not corrector.is_fitted:
            print(
                f"Warning: model at {model_path} is not fitted; correction is pass-through.",
                file=sys.stderr,
            )

    labelled = read_eval_rows(csv_path)
    csv_dir = csv_path.parent
    result_rows: list[dict[str, Any]] = []
    baseline_preds: list[float] = []
    corrected_preds: list[float] = []
    tapes_ok: list[float] = []
    tapes_ok_corrected: list[float] = []

    print(f"Evaluating {len(labelled)} rows from {csv_path}")
    for index, row in enumerate(labelled, start=1):
        image_raw = row["image_path"]
        try:
            reference_size_mm = float(row["reference_size_mm"])
            tape_mm = float(row["tape_measurement_mm"])
        except ValueError:
            result_rows.append(
                _result_record(
                    image_raw, row.get("reference_size_mm"), row.get("tape_measurement_mm"),
                    status="invalid_row",
                    message="reference_size_mm or tape_measurement_mm is not a number",
                )
            )
            continue

        image_path = resolve_image_path(image_raw, csv_dir)
        image = load_image_bgr(image_path)
        if image is None:
            result_rows.append(
                _result_record(
                    str(image_path),
                    reference_size_mm,
                    tape_mm,
                    tape_band=classify_risk_band(tape_mm)["risk_band"],
                    status="image_not_found",
                    message=f"Could not read image: {image_path}",
                )
            )
            print(f"  [{index}/{len(labelled)}] FAIL image_not_found  {image_path}")
            continue

        baseline_mm, status = run_baseline_pipeline(image, reference_size_mm)
        tape_band = classify_risk_band(tape_mm)["risk_band"]
        if status != "ok" or baseline_mm is None:
            result_rows.append(
                _result_record(
                    str(image_path),
                    reference_size_mm,
                    tape_mm,
                    tape_band=tape_band,
                    status=status,
                    message=status,
                )
            )
            print(f"  [{index}/{len(labelled)}] FAIL {status}  {image_path.name}")
            continue

        baseline_band = classify_risk_band(baseline_mm)["risk_band"]
        baseline_error = baseline_mm - tape_mm
        baseline_preds.append(baseline_mm)
        tapes_ok.append(tape_mm)

        corrected_mm = None
        corrected_band = None
        corrected_error = None
        if corrector is not None:
            corrected_mm = corrector.predict(baseline_mm)
            corrected_band = classify_risk_band(corrected_mm)["risk_band"]
            corrected_error = corrected_mm - tape_mm
            corrected_preds.append(corrected_mm)
            tapes_ok_corrected.append(tape_mm)

        result_rows.append(
            _result_record(
                str(image_path),
                reference_size_mm,
                tape_mm,
                tape_band=tape_band,
                baseline_muac_mm=baseline_mm,
                baseline_band=baseline_band,
                baseline_error_mm=baseline_error,
                corrected_muac_mm=corrected_mm,
                corrected_band=corrected_band,
                corrected_error_mm=corrected_error,
                status="ok",
                message="",
            )
        )
        extra = ""
        if corrected_mm is not None:
            extra = f"  corrected={corrected_mm:.2f} mm"
        print(
            f"  [{index}/{len(labelled)}] ok  tape={tape_mm:.1f}  "
            f"baseline={baseline_mm:.2f} mm ({baseline_band}){extra}  {image_path.name}"
        )

    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    write_results_csv(output_path, result_rows)

    print()
    print(format_metrics("Baseline", compute_metrics(baseline_preds, tapes_ok)))
    if corrector is not None:
        print()
        print(format_metrics("Corrected", compute_metrics(corrected_preds, tapes_ok_corrected)))
    print()
    print(f"Wrote per-image results to {output_path}")
    n_fail = sum(1 for row in result_rows if row["status"] != "ok")
    print(f"Succeeded {len(baseline_preds)}/{len(labelled)}  failed {n_fail}")
    return 0


def _result_record(
    image_path: str,
    reference_size_mm: Any,
    tape_measurement_mm: Any,
    *,
    tape_band: Optional[str] = None,
    baseline_muac_mm: Optional[float] = None,
    baseline_band: Optional[str] = None,
    baseline_error_mm: Optional[float] = None,
    corrected_muac_mm: Optional[float] = None,
    corrected_band: Optional[str] = None,
    corrected_error_mm: Optional[float] = None,
    status: str,
    message: str,
) -> dict[str, Any]:
    return {
        "image_path": image_path,
        "reference_size_mm": reference_size_mm,
        "tape_measurement_mm": tape_measurement_mm,
        "tape_band": tape_band or "",
        "baseline_muac_mm": _fmt(baseline_muac_mm),
        "baseline_band": baseline_band or "",
        "baseline_error_mm": _fmt(baseline_error_mm),
        "corrected_muac_mm": _fmt(corrected_muac_mm),
        "corrected_band": corrected_band or "",
        "corrected_error_mm": _fmt(corrected_error_mm),
        "status": status,
        "message": message,
    }


def _fmt(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
