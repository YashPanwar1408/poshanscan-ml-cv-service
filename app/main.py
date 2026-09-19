"""FastAPI entrypoint for the PoshanScan computer-vision microservice."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from app.pipeline.classify import classify_risk_band, compute_confidence_score
from app.pipeline.estimate import CalibrationCorrector, estimate_muac
from app.pipeline.pose_localize import locate_arm_midpoint
from app.pipeline.reference_detect import detect_reference_marker
from app.pipeline.segment import measure_width_at_row, segment_arm
from app.schemas import (
    PIPELINE_VERSION,
    ErrorResponse,
    HealthResponse,
    InferResponse,
    QualityFlags,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("poshanscan-cv")

MIN_FOREGROUND_PX = 100
MIN_WIDTH_PX = 5.0

_calibrator = CalibrationCorrector()

app = FastAPI(
    title="PoshanScan CV Service",
    description=(
        "Estimates mid-upper arm circumference (MUAC) from a photo of a child's "
        "upper arm taken next to a reference marker, for malnutrition screening."
    ),
    version=PIPELINE_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error(status_code: int, error: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=error, message=message).model_dump(),
    )


def _decode_image(payload: bytes) -> np.ndarray | None:
    if not payload:
        return None
    array = np.frombuffer(payload, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def _segmentation_ok(mask: np.ndarray, width_px: float) -> bool:
    if mask is None or mask.size == 0:
        return False
    foreground = int(np.count_nonzero(mask))
    return foreground >= MIN_FOREGROUND_PX and width_px >= MIN_WIDTH_PX


def _segmentation_quality(mask: np.ndarray) -> float:
    if mask is None or mask.size == 0:
        return 0.0
    return float(np.clip(np.count_nonzero(mask) / float(mask.size), 0.0, 1.0))


def _locate_either_arm(image: np.ndarray) -> dict[str, Any]:
    left = locate_arm_midpoint(image, side="left")
    if left.get("detected"):
        logger.info(
            "pose_localize: side=left detected=%s midpoint=%s angle=%s conf=%.3f",
            left["detected"],
            left.get("midpoint_px"),
            left.get("arm_angle_degrees"),
            left.get("confidence") or 0.0,
        )
        return left
    right = locate_arm_midpoint(image, side="right")
    logger.info(
        "pose_localize: left missed; side=right detected=%s midpoint=%s angle=%s conf=%.3f",
        right["detected"],
        right.get("midpoint_px"),
        right.get("arm_angle_degrees"),
        right.get("confidence") or 0.0,
    )
    return right


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send browsers to the Swagger UI."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return a simple liveness check."""
    return HealthResponse(status="ok")


@app.post(
    "/infer",
    response_model=InferResponse,
    responses={
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def infer(
    image: UploadFile = File(..., description="Photo of the upper arm with a reference marker."),
    reference_type: str = Form("aruco", description="Reference object type (currently only aruco)."),
    reference_size_mm: float = Form(50.0, description="Printed marker edge length in millimetres."),
) -> InferResponse | JSONResponse:
    """Run the MUAC pipeline on an uploaded image."""
    try:
        payload = await image.read()
        bgr = _decode_image(payload)
        if bgr is None:
            return _error(
                422,
                "invalid_image",
                "Could not decode the uploaded file as an image.",
            )

        if (reference_type or "aruco").strip().lower() != "aruco":
            logger.info(
                "reference_type=%s is not implemented; falling back to aruco",
                reference_type,
            )

        reference = detect_reference_marker(bgr, float(reference_size_mm))
        logger.info(
            "reference_detect: detected=%s scale_px_per_mm=%s conf=%.3f marker_id=%s",
            reference["detected"],
            reference.get("scale_px_per_mm"),
            reference.get("confidence") or 0.0,
            reference.get("marker_id"),
        )
        if not reference["detected"] or not reference.get("scale_px_per_mm"):
            return _error(
                422,
                "reference_object_not_detected",
                "No ArUco reference marker was found in the image.",
            )

        pose = _locate_either_arm(bgr)
        if not pose.get("detected") or pose.get("midpoint_px") is None:
            return _error(
                422,
                "arm_not_detected",
                "Could not localise a shoulder–elbow pair for MUAC measurement.",
            )

        midpoint = pose["midpoint_px"]
        segmented = segment_arm(bgr, midpoint)
        mask = segmented["mask"]
        centroid = segmented.get("centroid_px")
        row = int(round(centroid[1])) if centroid is not None else mask.shape[0] // 2
        width_px = measure_width_at_row(mask, row)
        seg_ok = _segmentation_ok(mask, width_px)
        seg_quality = _segmentation_quality(mask)
        logger.info(
            "segment: centroid=%s width_px=%.2f foreground=%d ok=%s quality=%.3f",
            centroid,
            width_px,
            int(np.count_nonzero(mask)),
            seg_ok,
            seg_quality,
        )

        scale = float(reference["scale_px_per_mm"])
        estimated = estimate_muac(width_px, scale)
        baseline_mm = estimated.get("muac_estimate_mm")
        if baseline_mm is None:
            baseline_mm = 0.0
        muac_mm = _calibrator.predict(float(baseline_mm))
        logger.info(
            "estimate: width_mm=%s muac_baseline_mm=%s muac_corrected_mm=%.3f shape_factor=%s",
            estimated.get("width_mm"),
            baseline_mm,
            muac_mm,
            estimated.get("shape_factor_used"),
        )

        classified = classify_risk_band(muac_mm)
        confidence = compute_confidence_score(
            float(reference.get("confidence") or 0.0),
            float(pose.get("confidence") or 0.0),
            seg_quality,
        )
        logger.info(
            "classify: risk_band=%s color=%s confidence=%.3f",
            classified["risk_band"],
            classified["color"],
            confidence,
        )

        return InferResponse(
            muac_estimate_mm=float(muac_mm),
            risk_band=classified["risk_band"],
            confidence_score=confidence,
            quality_flags=QualityFlags(
                reference_detected=True,
                arm_detected=True,
                segmentation_ok=seg_ok,
            ),
            pipeline_version=PIPELINE_VERSION,
        )
    except Exception:
        logger.exception("Unexpected error during /infer")
        return _error(
            500,
            "internal_error",
            "An unexpected error occurred while processing the image.",
        )
