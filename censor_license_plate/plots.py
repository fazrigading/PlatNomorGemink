"""Histogram plots drawn with OpenCV, so numpy and cv2 stay the only runtime deps."""

from __future__ import annotations

import cv2
import numpy as np

from .features import color_histograms, grayscale_histogram

PANEL_WIDTH = 560
PANEL_HEIGHT = 420
MARGIN = 40
BG = (255, 255, 255)
AXIS = (200, 200, 200)
# OpenCV channel order: blue, green, red.
BGR_KEYS = ((255, 0, 0), (0, 255, 0), (0, 0, 255))


def _panel(histograms: list[np.ndarray], colors, title: str) -> np.ndarray:
    canvas = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), BG, np.uint8)
    plot_width = PANEL_WIDTH - 2 * MARGIN
    plot_height = PANEL_HEIGHT - 2 * MARGIN
    peak = max(float(hist.max()) for hist in histograms) or 1.0

    cv2.line(canvas, (MARGIN, MARGIN), (MARGIN, MARGIN + plot_height), AXIS, 1)
    cv2.line(canvas, (MARGIN, MARGIN + plot_height), (MARGIN + plot_width, MARGIN + plot_height), AXIS, 1)

    for hist, color in zip(histograms, colors):
        points = [
            (
                MARGIN + int(i * plot_width / (len(hist) - 1)),
                MARGIN + plot_height - int(float(count) / peak * plot_height),
            )
            for i, count in enumerate(hist)
        ]
        cv2.polylines(canvas, [np.array(points, np.int32)], False, color, 2)

    cv2.putText(canvas, title, (MARGIN, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(
        canvas,
        f"peak {int(peak)}",
        (MARGIN, PANEL_HEIGHT - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )
    return canvas


def histogram_figure(image: np.ndarray) -> np.ndarray:
    """Side by side grayscale and RGB histogram figure of a BGR image."""
    grayscale = _panel([grayscale_histogram(image)], [(0, 0, 0)], "Histogram Grayscale")
    rgb = _panel(list(color_histograms(image)), BGR_KEYS, "Histogram RGB")
    return np.hstack([grayscale, rgb])
