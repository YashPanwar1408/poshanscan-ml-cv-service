"""Pydantic request and response models for the MUAC estimation API."""

from pydantic import BaseModel, Field

from app.pipeline.classify import AgeCategory  # re-exported for API layer


PIPELINE_VERSION = "v1.0"


class QualityFlags(BaseModel):
    """Per-stage quality bits included in a successful inference."""

    reference_detected: bool
    arm_detected: bool
    segmentation_ok: bool


class InferResponse(BaseModel):
    """Successful POST /infer payload."""

    muac_estimate_mm: float = Field(..., description="Estimated MUAC in millimetres.")
    risk_band: str = Field(..., description='Classification band (e.g. "Normal", "MAM"/"Moderate", "SAM"/"Severe").')
    age_category: str = Field(..., description='Age category used for classification: "child_6_59m" or "adult".')
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    quality_flags: QualityFlags
    pipeline_version: str = Field(default=PIPELINE_VERSION)


class ErrorResponse(BaseModel):
    """Structured error body for 4xx/5xx responses."""

    error: str
    message: str


class HealthResponse(BaseModel):
    """Liveness payload returned by GET /health."""

    status: str = Field(default="ok")
