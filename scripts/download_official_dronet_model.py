"""Download and extract the official DroNet trained model archive."""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

OFFICIAL_MODEL_URL = "http://rpg.ifi.uzh.ch/data/dronet_model.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the official DroNet model archive.")
    parser.add_argument("--url", default=OFFICIAL_MODEL_URL, help="Model archive URL.")
    parser.add_argument(
        "--output-dir",
        default="checkpoints/official",
        help="Directory for the downloaded and extracted archive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "dronet_model.zip"

    if not zip_path.exists():
        print(f"Downloading {args.url}")
        urllib.request.urlretrieve(args.url, zip_path)
    else:
        print(f"Using existing {zip_path}")

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)

    print(f"Extracted model archive to {output_dir}")


if __name__ == "__main__":
    main()

