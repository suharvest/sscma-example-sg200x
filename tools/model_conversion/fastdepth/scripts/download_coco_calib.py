#!/usr/bin/env python3
"""
Download COCO val2017 images for INT8 calibration.

FastDepth's official calibration/eval set is NYU Depth V2, which is not
reachable from this network (datasets.lids.mit.edu and most NYU/DIODE
mirrors timed out, see README "校准数据" section). COCO val2017 general
scene photos are used instead, following the same pattern already used by
model_conversion/recamera_yolo_detection/scripts/download_coco_calib.py.
INT8 calibration only needs a representative distribution of natural-image
activations, not depth labels, so this is an acceptable substitute -- see
README for the tradeoff.

Usage:
    uv run python scripts/download_coco_calib.py --count 500
    uv run python scripts/download_coco_calib.py --count 500 --output-dir calib_set
"""

import argparse
import zipfile
import urllib.request
import random
import shutil
from pathlib import Path


COCO_VAL2017_URL = "http://images.cocodataset.org/zips/val2017.zip"


def download_coco_val2017(output_dir: Path, count: int = 500, seed: int = 42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.glob("*.jpg"))
    if len(existing) >= count:
        print(f"Already have {len(existing)} images in {output_dir}, skipping download.")
        return

    zip_path = output_dir.parent / "val2017.zip"
    extract_dir = output_dir.parent / "val2017_full"

    try:
        if not zip_path.exists():
            print(f"Downloading COCO val2017 ({COCO_VAL2017_URL})...")
            print("This is ~1GB, may take a few minutes...")

            def progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    pct = min(100, downloaded * 100 / total_size)
                    mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct:.0f}%)", end="", flush=True)

            urllib.request.urlretrieve(COCO_VAL2017_URL, zip_path, progress_hook)
            print("\nDownload complete.")
        else:
            print(f"Using cached zip: {zip_path}")

        print("Extracting images...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            jpg_files = [f for f in zf.namelist() if f.endswith(".jpg")]
            print(f"  Found {len(jpg_files)} images in archive")

            random.seed(seed)
            selected = random.sample(jpg_files, min(count, len(jpg_files)))
            print(f"  Selected {len(selected)} random images for calibration")

            for i, fname in enumerate(selected):
                zf.extract(fname, extract_dir)
                if (i + 1) % 50 == 0:
                    print(f"  Extracted {i + 1}/{len(selected)}")

        for f in output_dir.iterdir():
            if not f.name.endswith(".jpg"):
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)

        for f in output_dir.glob("*.jpg"):
            if f.stat().st_size < 10000:
                f.unlink()

        moved = 0
        for jpg in extract_dir.rglob("*.jpg"):
            dest = output_dir / jpg.name
            shutil.move(str(jpg), str(dest))
            moved += 1

        print(f"  Moved {moved} images to {output_dir}")

    finally:
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)

    final_count = len(list(output_dir.glob("*.jpg")))
    print(f"\nCalibration dataset ready: {final_count} images in {output_dir}")
    print(f"Zip cached at: {zip_path} (delete to free ~1GB)")


def main():
    parser = argparse.ArgumentParser(description="Download COCO val2017 images for INT8 calibration")
    parser.add_argument("--count", type=int, default=500, help="Number of images to extract (default: 500)")
    parser.add_argument("--output-dir", type=Path, default=Path("calib_set"), help="Output directory (default: calib_set)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for image selection (default: 42)")

    args = parser.parse_args()
    download_coco_val2017(args.output_dir, args.count, args.seed)


if __name__ == "__main__":
    main()
