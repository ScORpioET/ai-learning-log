"""
Export YOLO models to ONNX in FP32 and/or FP16.

Usage:
    python export_yolo.py yolov8x
    python export_yolo.py yolov8n yolov8s yolov8m yolov8x
    python export_yolo.py yolov8x --precision fp16
    python export_yolo.py yolov8x --output-dir ~/ai-transition-2026/model
"""
import argparse
import os
import shutil
from pathlib import Path
from ultralytics import YOLO

PRECISION_MAP = {
    "fp32": False,
    "fp16": True,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO models to ONNX")
    parser.add_argument(
        "models",
        nargs="+",
        help="Model names (e.g. yolov8n yolov8m yolov8x). '.pt' suffix optional.",
    )
    parser.add_argument(
        "--precision",
        nargs="+",
        choices=["fp32", "fp16"],
        default=["fp32", "fp16"],
        help="Precision(s) to export. Default: both.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Where to move the exported ONNX files. Default: current dir.",
    )
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version. Default: 12.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size. Default: 640.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in args.models:
        # 容錯：吃 "yolov8x" 或 "yolov8x.pt" 都行
        model_file = name if name.endswith(".pt") else f"{name}.pt"
        base = model_file.replace(".pt", "")

        for suffix in args.precision:
            half_flag = PRECISION_MAP[suffix]
            print(f"\n=== Exporting {base} @ {suffix} ===")

            model = YOLO(model_file)
            model.export(
                format="onnx",
                opset=args.opset,
                simplify=True,
                dynamic=False,
                imgsz=args.imgsz,
                half=half_flag,
            )

            # Ultralytics 固定輸出 <basename>.onnx、rename + move
            src = Path(f"{base}.onnx")
            dst = output_dir / f"{base}_{suffix}.onnx"

            if src.exists():
                shutil.move(str(src), str(dst))
                size_mb = dst.stat().st_size / 1e6
                print(f">>> {dst}  ({size_mb:.1f} MB)")
            else:
                print(f">>> ERROR: {src} not found")


if __name__ == "__main__":
    main()