"""Command line entry point: python -m censor_license_plate <command>."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .dataset import write_dataset
from .plots import histogram_figure
from .segmentation import censor, highlight, segment

DEFAULT_IMAGES = "data/images/rambu"
DEFAULT_DATASET = "data/rambu_dataset.csv"


def cmd_dataset(args: argparse.Namespace) -> int:
    count = write_dataset(args.images, args.csv, args.pattern)
    print(f"wrote {count} rows to {args.csv}")
    return 0


def cmd_segment(args: argparse.Namespace) -> int:
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"cannot read {args.image}")
    crop, contour = segment(image)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    print(f"{args.image}: crop {crop.shape[1]}x{crop.shape[0]}, {len(contour)} contour points -> {out}")
    return 0


MODES = {"highlight": highlight, "censor": censor}


def cmd_mode(args: argparse.Namespace) -> int:
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"cannot read {args.image}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = MODES[args.mode](image)
    cv2.imwrite(str(out), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"{args.image}: {args.mode} -> {out}")
    return 0


def cmd_histogram(args: argparse.Namespace) -> int:
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"cannot read {args.image}")
    figure = histogram_figure(image)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), figure)
    print(f"{args.image}: grayscale and RGB histograms -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="censor_license_plate", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    dataset = sub.add_parser("dataset", help="rebuild the feature dataset CSV")
    dataset.add_argument("--images", default=DEFAULT_IMAGES)
    dataset.add_argument("--csv", default=DEFAULT_DATASET)
    dataset.add_argument("--pattern", default="*")
    dataset.set_defaults(func=cmd_dataset)

    segment_cmd = sub.add_parser("segment", help="write the segmented region of one image")
    segment_cmd.add_argument("image")
    segment_cmd.add_argument("--out", default="out/segment.png")
    segment_cmd.set_defaults(func=cmd_segment)

    for mode, help_text, default_out in (
        ("censor", "blur the plate/sign region, keep the rest sharp", "out/censor.png"),
        ("highlight", "keep the plate/sign region sharp, blur the rest", "out/highlight.png"),
    ):
        mode_cmd = sub.add_parser(mode, help=help_text)
        mode_cmd.add_argument("image")
        mode_cmd.add_argument("--out", default=default_out)
        mode_cmd.set_defaults(func=cmd_mode, mode=mode)

    histogram_cmd = sub.add_parser(
        "histogram", help="draw the grayscale and RGB histograms of one image"
    )
    histogram_cmd.add_argument("image")
    histogram_cmd.add_argument("--out", default="out/histogram.png")
    histogram_cmd.set_defaults(func=cmd_histogram)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
