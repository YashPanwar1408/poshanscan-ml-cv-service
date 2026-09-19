"""Generate a printable OpenCV ArUco DICT_4X4_50 marker (same family the API detects)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline.reference_detect import ARUCO_DICT_ID, _get_dictionary  # noqa: E402


def generate_marker_png(marker_id: int, side_px: int, border_px: int) -> np.ndarray:
    dictionary = _get_dictionary()
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
    else:
        marker = cv2.aruco.drawMarker(dictionary, marker_id, side_px)
    if border_px <= 0:
        return marker
    return cv2.copyMakeBorder(
        marker,
        border_px,
        border_px,
        border_px,
        border_px,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a printable DICT_4X4_50 ArUco PNG.")
    parser.add_argument("--id", type=int, default=0, help="Marker ID in 0–49 (default 0).")
    parser.add_argument("--size", type=int, default=800, help="Black square size in pixels.")
    parser.add_argument("--border", type=int, default=120, help="White quiet-zone in pixels.")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "sample_images" / "aruco_4x4_50_id0.png",
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    image = generate_marker_png(args.id, args.size, args.border)
    cv2.imwrite(str(args.out), image)
    print(f"Wrote {args.out}  (DICT_4X4_50 id={args.id})")
    print("Print it, measure the BLACK square edge in mm, and pass that as reference_size_mm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
