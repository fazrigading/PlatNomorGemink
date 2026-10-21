"""Build the rambu feature dataset from a folder of images."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2

from .features import occupancy_vector, vertex_count
from .segmentation import segment

COLUMNS = ("nama", "fitur", "segi")


def build_rows(image_dir: str | Path, pattern: str = "*"):
    """Yield one row per readable image: name, 64 element vector, vertex count."""
    for path in sorted(p for p in Path(image_dir).glob(pattern) if p.is_file()):
        image = cv2.imread(str(path))
        if image is None:
            continue
        crop, contour = segment(image)
        yield {
            "nama": path.name,
            "fitur": occupancy_vector(crop).tolist(),
            "segi": vertex_count(contour),
        }


def write_dataset(image_dir: str | Path, csv_path: str | Path, pattern: str = "*") -> int:
    """Write the dataset CSV and return the number of rows written."""
    rows = list(build_rows(image_dir, pattern))
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
