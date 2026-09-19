"""API tests for /health and /infer error paths."""

from __future__ import annotations

import io

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_redirects_to_docs() -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (307, 302)
    assert response.headers["location"] == "/docs"


def test_infer_rejects_undecodable_file() -> None:
    response = client.post(
        "/infer",
        files={"image": ("note.txt", b"not-an-image", "text/plain")},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "invalid_image"
    assert "message" in body


def test_infer_no_marker_returns_422() -> None:
    blank = np.full((240, 320, 3), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", blank)
    assert ok
    response = client.post(
        "/infer",
        files={"image": ("blank.png", io.BytesIO(encoded.tobytes()), "image/png")},
        data={"reference_type": "aruco", "reference_size_mm": "50.0"},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "reference_object_not_detected"
