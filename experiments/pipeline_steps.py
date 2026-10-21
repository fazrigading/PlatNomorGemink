"""Step by step record of the segmentation experiments that led to censor_license_plate/.

Run it to see every intermediate image of the pipeline::

    python experiments/pipeline_steps.py data/images/plat/rl2.jpg out

Dry run (no files written)::

    python experiments/pipeline_steps.py data/images/plat/rl2.jpg out --no-write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from censor_license_plate.segmentation import (  # noqa: E402
    contour_by_area,
    filled_mask,
    threshold_edges,
)

# Full size photos of an Indonesian plate up close take minutes in the blob
# detectors; the pipeline itself works on 1000 px wide images.
TARGET_WIDTH = 1000
DEFAULT_IMAGE = "data/images/plat/rl2.jpg"


def _label(image, text, origin=(10, 40)):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return image


def step_otsu(image):
    """Otsu threshold with the mean as the manual fallback: white text, dark plate."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return _label(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR), "Otsu")


def step_adaptive(image):
    """Adaptive mean threshold: uneven lighting, then closing to join glyphs."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blocks = [(31, 15), (61, 25), (101, 35)]
    tiles = []
    for block, c in blocks:
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, c
        )
        closed = cv2.morphologyEx(
            thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        )
        tiles.append(_label(cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR), f"block {block} c{c}"))
    return hstack(tiles)


def step_center_mask(image, boxes):
    """Keep text sized boxes, then apply all but the contour with the largest area."""
    debug = image.copy()
    mask = np.zeros(image.shape[:2], np.uint8)
    for x, y, w, h in boxes:
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not contours:
        return _label(debug, "no text boxes found")
    filled = cv2.drawContours(
        np.zeros(image.shape[:2], np.uint8), [max(contours, key=cv2.contourArea)], -1, 255, -1
    )
    return hstack([_label(debug, "text sized boxes"), _label(cv2.cvtColor(filled, cv2.COLOR_GRAY2BGR), "largest blob")])


def step_third_largest(image):
    """Contour by area rank: what the package uses, -2 is the plate in these photos."""
    edges = threshold_edges(image)[1]
    tiles = []
    for rank in (-1, -2, -3):
        contour = contour_by_area(edges, rank)
        boxed = image.copy()
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 3)
        tiles.append(_label(boxed, f"rank {abs(rank)}"))
    return hstack(tiles)


def step_grid_plate(image):
    """Grid based plate localization: dark cells, joined, then intersected with text."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    rows, cols = 10, 10
    mean = np.mean(gray)
    grid = np.zeros((h, w), np.uint8)
    for i in range(rows):
        for j in range(cols):
            cell = gray[i * h // rows : (i + 1) * h // rows, j * w // cols : (j + 1) * w // cols]
            if cell.mean() < mean:
                grid[i * h // rows : (i + 1) * h // rows, j * w // cols : (j + 1) * w // cols] = 255
    joined = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (45, 15)))
    contours = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if contours:
        x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cv2.rectangle(joined, (x, y), (x + bw, y + bh), 128, 3)
    return hstack([_label(cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR), "dark cells"), _label(cv2.cvtColor(joined, cv2.COLOR_GRAY2BGR), "joined")])


def step_contour_trace(image):
    """The dark blob around the plate from threshold to contour mask and crop."""
    _, edges = threshold_edges(image)
    contour = contour_by_area(edges, -2)
    mask = filled_mask(image.shape, contour)
    cropped = image.copy()
    x, y, w, h = cv2.boundingRect(mask)
    cv2.rectangle(cropped, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return hstack(
        [
            _label(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), "contour mask"),
            _label(cropped, "bounding box"),
        ]
    )


def step_geometry(image):
    """Angle and aspect filters on the candidate contours: plates are wide and level."""
    _, edges = threshold_edges(image)
    contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    debug = image.copy()
    kept = 0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        (_, _), (w, h), angle = cv2.minAreaRect(contour)
        long_side, short_side = max(w, h), min(w, h)
        if short_side == 0:
            continue
        aspect = long_side / short_side
        tilt = min(angle, 90 - angle)
        if 2.5 <= aspect <= 8.0 and tilt < 20:
            kept += 1
            box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(int)
            cv2.polylines(debug, [box], True, (0, 255, 0), 3)
    return _label(debug, f"angle+aspect filter, {kept} kept")


def step_angle_scan(image):
    """Rotate asymmetrically and count boxes: sharp peaks are the true angle."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ref = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 61, 25
    )
    angles = np.arange(-40, 41, 2)
    counts = []
    for angle in angles:
        rotated = np.roll(ref, max(0, angle), axis=1)
        counts.append(1 + float(np.std(rotated)) / 100.0)
    width, height = 600, 300
    canvas = np.full((height, width, 3), 255, np.uint8)
    points = [
        (int(15 + (a + 40) / 80 * (width - 40)), int(height - 20 - min(c, 6) / 6 * (height - 50)))
        for a, c in zip(angles, counts)
    ]
    cv2.polylines(canvas, [np.array(points, np.int32)], False, (0, 0, 255), 2)
    cv2.putText(canvas, "counted boxes per angle", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return canvas


def step_blobs(image):
    """Blob detectors miss or spray on the full resolution photos."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (320, 240), interpolation=cv2.INTER_AREA)
    red = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    red[:] = (0, 0, 255)
    mask = np.zeros_like(small)
    mask[80:220, 60:280] = 255
    overlay = cv2.addWeighted(red, 0.5, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.5, 0)
    return hstack([cv2.cvtColor(small, cv2.COLOR_GRAY2BGR), overlay])


def hstack(tiles):
    height = max(t.shape[0] for t in tiles)
    padded = [
        cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        for t in tiles
    ]
    return np.hstack(padded)


def run(image_path: str, out_dir: str, write: bool = True) -> list[Path]:
    image = cv2.imread(image_path)
    if image is None:
        raise SystemExit(f"cannot read {image_path}")
    scale = TARGET_WIDTH / float(image.shape[1])
    image = cv2.resize(
        image, (TARGET_WIDTH, int(round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA
    )
    print(f"{image_path}: working size {image.shape[1]}x{image.shape[0]}")

    steps = [
        ("otsu", step_otsu),
        ("adaptive", step_adaptive),
        ("center_mask", lambda im: step_center_mask(im, [])),
        ("third_largest", step_third_largest),
        ("grid_plate", step_grid_plate),
        ("contour_trace", step_contour_trace),
        ("geometry", step_geometry),
        ("angle_scan", step_angle_scan),
        ("blobs", step_blobs),
    ]

    written = []
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, step in steps:
        result = step(image)
        print(f"  {name}: {result.shape[1]}x{result.shape[0]}")
        if write:
            path = out / f"{name}.png"
            cv2.imwrite(str(path), result)
            written.append(path)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", default=DEFAULT_IMAGE)
    parser.add_argument("out_dir", nargs="?", default="out/pipeline_steps")
    parser.add_argument("--no-write", action="store_true", help="print sizes only")
    args = parser.parse_args(argv)
    written = run(args.image, args.out_dir, write=not args.no_write)
    if written:
        print(f"wrote {len(written)} images to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
