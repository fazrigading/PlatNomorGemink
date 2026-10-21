"""Fixed size feature vector of a segmented sign: 32x32 occupancy profile."""

from __future__ import annotations

import cv2
import numpy as np

SIZE = 32


def square(image: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Resize to `size` x `size`."""
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def occupancy_vector(image: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Return the 64 element feature vector of an image.

    The image is resized to `size` x `size`, binarised, then summed per column
    and per row; the two profiles are concatenated. This is the representation
    stored in ``data/rambu_dataset.csv``.
    """
    if image is None or image.size == 0:
        raise ValueError("empty image")
    gray = cv2.cvtColor(square(image, size), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, float(np.mean(gray)), 1, cv2.THRESH_BINARY)
    return np.concatenate((np.sum(thresh, axis=0), np.sum(thresh, axis=1)), axis=None)


def vertex_count(contour: np.ndarray, epsilon_ratio: float = 0.01) -> int:
    """Number of polygon vertices of a contour, the `segi` column of the dataset."""
    perimeter = cv2.arcLength(contour, True)
    return len(cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True))


def grayscale_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
    """Histogram of the grayscale image, one count per intensity value."""
    if image is None or image.size == 0:
        raise ValueError("empty image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.calcHist([gray], [0], None, [bins], [0, 256]).ravel()


def color_histograms(image: np.ndarray, bins: int = 256) -> np.ndarray:
    """Histogram of every BGR channel, shape (3, bins)."""
    if image is None or image.ndim != 3:
        raise ValueError("expected a colour image")
    return np.array(
        [cv2.calcHist([image], [channel], None, [bins], [0, 256]).ravel() for channel in range(3)]
    )
