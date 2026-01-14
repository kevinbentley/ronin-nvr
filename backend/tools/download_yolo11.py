#!/usr/bin/env python3
"""Download YOLO11 and export to ONNX with dynamic batch size."""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Download YOLO11 and export to ONNX")
    parser.add_argument(
        "--model",
        default="yolo11l",
        choices=["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
        help="YOLO11 model variant (default: yolo11l)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/sas1/ronin/ml_models"),
        help="Output directory for ONNX model",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        default=True,
        help="Export with dynamic batch size (default: True)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17 for CUDA compatibility)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.model}...")
    model = YOLO(f"{args.model}.pt")

    suffix = "_dynamic" if args.dynamic else ""
    output_path = args.output_dir / f"{args.model}{suffix}.onnx"

    print(f"Exporting to ONNX: {output_path} (opset {args.opset})")
    model.export(
        format="onnx",
        dynamic=args.dynamic,
        imgsz=args.imgsz,
        simplify=True,
        opset=args.opset,
    )

    # Move from default location to output directory
    default_export = Path(f"{args.model}.onnx")
    if default_export.exists():
        shutil.move(str(default_export), str(output_path))
        print(f"Model saved to: {output_path}")
    else:
        print(f"Warning: Expected export at {default_export} not found")
        print("Check current directory for exported model")

    # Clean up the .pt file if downloaded to current directory
    pt_file = Path(f"{args.model}.pt")
    if pt_file.exists():
        pt_file.unlink()
        print(f"Cleaned up: {pt_file}")


if __name__ == "__main__":
    main()
