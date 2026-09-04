"""對 RGB/thermal val 全量圖片跑 YOLO,重用 Day35 run_yolo_inference.py 的 main(),不重寫。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "caption_fusion" / ".pylibs"))
sys.path.insert(0, str(Path.home() / "ai-transition-2026" / "Phase3" / "Day35"))
from run_yolo_inference import main as yolo_main  # noqa: E402

TD = Path.home() / "ai-transition-2026" / "thermal_dataset"
HERE = Path(__file__).parent

if __name__ == "__main__":
    print("[1/2] RGB val...")
    yolo_main(TD / "images_rgb_val" / "data", HERE / "detections_rgb_val.jsonl")
    print("\n[2/2] thermal val...")
    yolo_main(TD / "images_thermal_val" / "data", HERE / "detections_thermal_val.jsonl")
    print("\n[all done]")
