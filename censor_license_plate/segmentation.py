"""Contour based segmentation of plate / traffic-sign (rambu) regions."""

from __future__ import annotations

import cv2
import numpy as np

TARGET_WIDTH = 1000
# The largest contour of these photos is the image outline, so the original
# experiment picked the second largest one.
DEFAULT_RANK = -2


def resize_to_width(image: np.ndarray, width: int = TARGET_WIDTH) -> np.ndarray:
    """Scale the image to a fixed width, keeping the aspect ratio."""
    if image is None:
        raise ValueError("image could not be read")
    scale = width / float(image.shape[1])
    if scale == 1.0:
        return image
    return cv2.resize(image, (width, int(round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def threshold_edges(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the Otsu threshold and its dilated Canny edges."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray, float(np.mean(gray)), 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return thresh, cv2.dilate(cv2.Canny(thresh, 0, 255), None)


def contour_by_area(edges: np.ndarray, rank: int = DEFAULT_RANK) -> np.ndarray:
    """Pick a contour by area rank, counted from the end of the sorted list."""
    contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(contours) < abs(rank):
        raise ValueError(f"only {len(contours)} contours found, cannot pick rank {rank}")
    return sorted(contours, key=cv2.contourArea)[rank]


def filled_mask(shape, contour: np.ndarray) -> np.ndarray:
    """Binary mask of the contour region."""
    mask = np.zeros(shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask


def crop_to_mask(mask: np.ndarray, source: np.ndarray, pad: int = 1) -> np.ndarray:
    """Crop `source` to the contour bounding box, grown by `pad` pixels."""
    x, y, w, h = cv2.boundingRect(mask)
    y0, x0 = max(0, y - pad), max(0, x - pad)
    return source[y0 : y + h + pad, x0 : x + w + pad]


def segment(
    image: np.ndarray, rank: int = DEFAULT_RANK, width: int = TARGET_WIDTH
) -> tuple[np.ndarray, np.ndarray]:
    """Segment the plate/sign region.

    Returns the RGB crop of the ranked contour and that contour, both taken
    from the resized image.
    """
    image = resize_to_width(image, width)
    _, edges = threshold_edges(image)
    contour = contour_by_area(edges, rank)
    mask = filled_mask(image.shape, contour)
    rgb = cv2.cvtColor(cv2.bitwise_and(image, image, mask=mask), cv2.COLOR_BGR2RGB)
    return crop_to_mask(mask, rgb), contour


def _region_and_blur(
    image: np.ndarray, rank: int, width: int, blur_divisor: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the resized image, its region mask and a blurred copy."""
    image = resize_to_width(image, width)
    _, edges = threshold_edges(image)
    mask = filled_mask(image.shape, contour_by_area(edges, rank))
    kernel = max(1, int(image.shape[0] / blur_divisor))
    return image, mask, cv2.blur(image, (kernel, kernel))


def highlight(
    image: np.ndarray,
    rank: int = DEFAULT_RANK,
    width: int = TARGET_WIDTH,
    blur_divisor: int = 7,
) -> np.ndarray:
    """Mode `highlight`: keep the ranked region sharp, blur the rest."""
    image, mask, blurred = _region_and_blur(image, rank, width, blur_divisor)
    out = np.where(mask[:, :, None] == 255, image, blurred)
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2RGB)


def censor(
    image: np.ndarray,
    rank: int = DEFAULT_RANK,
    width: int = TARGET_WIDTH,
    blur_divisor: int = 7,
) -> np.ndarray:
    """Mode `censor`: blur the ranked region, keep the rest sharp."""
    image, mask, blurred = _region_and_blur(image, rank, width, blur_divisor)
    out = np.where(mask[:, :, None] == 255, blurred, image)
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2RGB)
