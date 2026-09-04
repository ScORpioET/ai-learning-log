"""
對齊 test split(RGB/thermal 各 3749 張,100% frame 對應)全量跑 YOLO 偵測。
直接 import Day35 run_yolo_inference.py 的 main(),不重寫偵測邏輯——
KEEP_CLASSES、CONF_THRESH(0.25)、MODEL_PATH(yolov8m,COCO pretrained)
全部沿用。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "caption_fusion" / ".pylibs"))
sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "Day35"))

from run_yolo_inference import main as yolo_main  # noqa: E402

TD = Path.home() / "ai-transition-2026" / "thermal_dataset"
HERE = Path(__file__).parent

if __name__ == "__main__":
    print("[step 1/2] RGB test full scan...")
    yolo_main(TD / "video_rgb_test" / "data", HERE / "detections_rgb_test.jsonl")
    print("\n[step 2/2] thermal test full scan...")
    yolo_main(TD / "video_thermal_test" / "data", HERE / "detections_thermal_test.jsonl")
    print("\n[all done]")
