"""Command line entry point: python -m platnomor <command>."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .dataset import write_dataset
from .segmentation import highlight, segment

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


def cmd_highlight(args: argparse.Namespace) -> int:
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"cannot read {args.image}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(highlight(image), cv2.COLOR_RGB2BGR))
    print(f"{args.image}: highlighted region -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="platnomor", description=__doc__)
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

    highlight_cmd = sub.add_parser("highlight", help="blur everything but the sign region")
    highlight_cmd.add_argument("image")
    highlight_cmd.add_argument("--out", default="out/highlight.png")
    highlight_cmd.set_defaults(func=cmd_highlight)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
