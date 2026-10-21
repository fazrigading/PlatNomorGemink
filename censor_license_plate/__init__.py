"""Plate / traffic-sign (plat nomor, rambu) region segmentation."""

from .features import (
    color_histograms,
    grayscale_histogram,
    occupancy_vector,
    square,
    vertex_count,
)
from .plots import histogram_figure
from .segmentation import (
    censor,
    contour_by_area,
    crop_to_mask,
    filled_mask,
    highlight,
    resize_to_width,
    segment,
    threshold_edges,
)

__all__ = [
    "censor",
    "color_histograms",
    "contour_by_area",
    "crop_to_mask",
    "filled_mask",
    "grayscale_histogram",
    "highlight",
    "histogram_figure",
    "occupancy_vector",
    "resize_to_width",
    "segment",
    "square",
    "threshold_edges",
    "vertex_count",
]
