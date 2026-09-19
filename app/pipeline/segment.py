"""Segment the upper-arm region around the MUAC measurement site."""

from __future__ import annotations

from typing import Any, Optional, TypedDict

import cv2
import numpy as np

# HSV skin ranges in OpenCV (H: 0–180). Covers typical light/medium skin;
# a second band catches reddish hues that wrap around H=0.
_SKIN_HSV_LOWER_1 = np.array([0, 30, 60], dtype=np.uint8)
_SKIN_HSV_UPPER_1 = np.array([25, 255, 255], dtype=np.uint8)
_SKIN_HSV_LOWER_2 = np.array([160, 30, 60], dtype=np.uint8)
_SKIN_HSV_UPPER_2 = np.array([179, 255, 255], dtype=np.uint8)

DEFAULT_CROP_SIZE_PX = 200


class SegmentArmResult(TypedDict):
    """Result of :func:`segment_arm`. Keep this shape when swapping in YOLO."""

    mask: np.ndarray
    centroid_px: Optional[tuple[float, float]]
    crop_origin_px: tuple[int, int]
    crop_bgr: np.ndarray


def _empty_mask(crop_size_px: int) -> np.ndarray:
    return np.zeros((crop_size_px, crop_size_px), dtype=np.uint8)


def _empty_result(crop_size_px: int = DEFAULT_CROP_SIZE_PX) -> SegmentArmResult:
    return {
        "mask": _empty_mask(crop_size_px),
        "centroid_px": None,
        "crop_origin_px": (0, 0),
        "crop_bgr": np.zeros((crop_size_px, crop_size_px, 3), dtype=np.uint8),
    }


def _is_empty_image(image: Any) -> bool:
    if image is None:
        return True
    array = np.asarray(image)
    return array.size == 0 or array.ndim < 2 or array.shape[0] == 0 or array.shape[1] == 0


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _extract_crop(
    image: np.ndarray,
    midpoint_px: tuple[float, float],
    crop_size_px: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop ``crop_size_px`` square around midpoint; pad if it hits the border."""
    height, width = image.shape[:2]
    cx = int(round(float(midpoint_px[0])))
    cy = int(round(float(midpoint_px[1])))
    half = crop_size_px // 2
    x0 = cx - half
    y0 = cy - half
    x1 = x0 + crop_size_px
    y1 = y0 + crop_size_px

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - width)
    pad_bottom = max(0, y1 - height)

    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(width, x1), min(height, y1)
    crop = image[ya:yb, xa:xb]
    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    if crop.shape[0] != crop_size_px or crop.shape[1] != crop_size_px:
        crop = cv2.resize(crop, (crop_size_px, crop_size_px), interpolation=cv2.INTER_NEAREST)
    return crop, (x0, y0)


def _skin_mask(crop_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    band_a = cv2.inRange(hsv, _SKIN_HSV_LOWER_1, _SKIN_HSV_UPPER_1)
    band_b = cv2.inRange(hsv, _SKIN_HSV_LOWER_2, _SKIN_HSV_UPPER_2)
    skin = cv2.bitwise_or(band_a, band_b)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel)
    return skin


def _grabcut_refine(crop_bgr: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """Refine a skin-colour seed mask with GrabCut. Falls back to ``skin``."""
    if int(cv2.countNonZero(skin)) < 50:
        return skin

    gc_mask = np.full(skin.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[skin > 0] = cv2.GC_PR_FGD
    kernel = np.ones((5, 5), dtype=np.uint8)
    sure_fg = cv2.erode(skin, kernel, iterations=2)
    sure_bg = cv2.bitwise_not(cv2.dilate(skin, kernel, iterations=2))
    gc_mask[sure_bg > 0] = cv2.GC_BGD
    gc_mask[sure_fg > 0] = cv2.GC_FGD

    if not np.any(gc_mask == cv2.GC_FGD) and not np.any(gc_mask == cv2.GC_PR_FGD):
        return skin

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            crop_bgr,
            gc_mask,
            None,
            bgd_model,
            fgd_model,
            5,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return skin

    refined = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    if int(cv2.countNonZero(refined)) == 0:
        return skin
    return refined


def _mask_centroid(mask: np.ndarray) -> Optional[tuple[float, float]]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def segment_arm(
    image: np.ndarray,
    midpoint_px: tuple[float, float],
    crop_size_px: int = DEFAULT_CROP_SIZE_PX,
) -> SegmentArmResult:
    """Segment the arm in a crop centred on the MUAC midpoint.

    Baseline implementation: HSV skin-colour threshold + GrabCut refinement.
    The returned dict shape is the contract for later model swaps.

    TODO(yolov8-seg): replace the body of this function with a YOLOv8-seg
    (or equivalent instance-segmentation) forward pass, for example::

        # from ultralytics import YOLO
        # model = YOLO("app/models/arm_seg.pt")
        # preds = model.predict(crop_bgr, verbose=False)
        # mask = preds[0].masks.data[0].cpu().numpy()  # then resize to crop

    Keep this signature stable so estimate/classify do not change::

        segment_arm(image: np.ndarray, midpoint_px: tuple, crop_size_px: int = 200) -> dict
        # dict keys: mask (H=W=crop_size_px, uint8 0/255), centroid_px, crop_origin_px, crop_bgr

    Args:
        image: Full BGR (or grayscale) photo.
        midpoint_px: ``(x, y)`` MUAC site in full-image pixel coordinates.
        crop_size_px: Square crop side length.

    Returns:
        Segmentation dict. ``mask`` is always ``crop_size_px x crop_size_px``.
        ``centroid_px`` is in crop coordinates, or None if nothing was found.
    """
    if _is_empty_image(image) or midpoint_px is None or crop_size_px <= 0:
        return _empty_result(max(int(crop_size_px), 1) if crop_size_px else DEFAULT_CROP_SIZE_PX)

    bgr = _to_bgr(np.asarray(image))
    crop_bgr, origin = _extract_crop(bgr, midpoint_px, crop_size_px)

    # --- classical baseline (swap this block for YOLOv8-seg; see TODO above) ---
    skin = _skin_mask(crop_bgr)
    mask = _grabcut_refine(crop_bgr, skin)
    # --- end classical baseline ------------------------------------------------

    return {
        "mask": mask,
        "centroid_px": _mask_centroid(mask),
        "crop_origin_px": origin,
        "crop_bgr": crop_bgr,
    }


def measure_width_at_row(mask: np.ndarray, row: int) -> float:
    """Pixel width of the arm on a horizontal slice of the binary mask.

    Uses the longest contiguous foreground run so small holes/speckles do
    not inflate the width. Returns 0.0 if ``row`` is out of bounds or empty.
    """
    if mask is None or mask.size == 0:
        return 0.0
    if row < 0 or row >= mask.shape[0]:
        return 0.0
    line = (mask[row] > 0).astype(np.uint8)
    if int(line.sum()) == 0:
        return 0.0

    padded = np.concatenate(([0], line, [0]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    run_lengths = ends - starts
    return float(run_lengths.max()) if run_lengths.size else 0.0
