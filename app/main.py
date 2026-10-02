"""FastAPI entrypoint for the PoshanScan computer-vision microservice."""

from __future__ import annotations

import logging

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from app.pipeline.run import run_muac_pipeline
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

        result = run_muac_pipeline(bgr, float(reference_size_mm))
        if not result.get("ok"):
            return _error(
                422,
                str(result.get("error") or "pipeline_failed"),
                str(result.get("message") or "Pipeline failed."),
            )

        return InferResponse(
            muac_estimate_mm=float(result["muac_estimate_mm"]),
            risk_band=str(result["risk_band"]),
            confidence_score=float(result["confidence_score"]),
            quality_flags=QualityFlags(
                reference_detected=True,
                arm_detected=True,
                segmentation_ok=bool(result.get("segmentation_ok")),
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
