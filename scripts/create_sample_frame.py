"""Create a deterministic synthetic frame for offline inference smoke tests.

This image is not training data and should not be used for evaluation. It only
gives the inference CLI a stable local input before real datasets or AirSim
frames are connected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def create_sample_frame(width: int = 320, height: int = 240) -> np.ndarray:
    """Return a simple road-like BGR frame."""

    image = np.full((height, width, 3), fill_value=175, dtype=np.uint8)

    horizon_y = int(height * 0.42)
    road = np.array(
        [
            [int(width * 0.10), height],
            [int(width * 0.42), horizon_y],
            [int(width * 0.58), horizon_y],
            [int(width * 0.90), height],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [road], color=(70, 70, 70))

    cv2.line(
        image,
        (int(width * 0.49), height),
        (int(width * 0.50), horizon_y),
        color=(230, 230, 230),
        thickness=2,
    )
    cv2.line(
        image,
        (int(width * 0.25), height),
        (int(width * 0.43), horizon_y),
        color=(210, 210, 210),
        thickness=3,
    )
    cv2.line(
        image,
        (int(width * 0.75), height),
        (int(width * 0.57), horizon_y),
        color=(210, 210, 210),
        thickness=3,
    )

    obstacle_top_left = (int(width * 0.62), int(height * 0.56))
    obstacle_bottom_right = (int(width * 0.73), int(height * 0.78))
    cv2.rectangle(image, obstacle_top_left, obstacle_bottom_right, color=(35, 35, 35), thickness=-1)

    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic sample frame.")
    parser.add_argument(
        "--output",
        default="data/samples/synthetic_road_frame.png",
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = create_sample_frame()

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to write sample image: {output_path}")

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

