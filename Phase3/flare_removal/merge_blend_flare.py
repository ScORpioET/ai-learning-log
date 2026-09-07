"""
把 blend/(去光斑後的圖,原始解析度)跟 flare/(抽出的光斑成分,已重新上色成
暖色熱圖,解析度較低)左右合併成一張圖,方便對照。flare 圖 resize 成跟 blend
一樣大小再左右拼接,同檔名輸出到各 split 底下新的 merged/ 資料夾。
"""
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
SPLITS = ["out_train", "out_val", "out_test"]


def merge_one(blend_path, flare_path, out_path):
    blend = Image.open(blend_path).convert("RGB")
    flare = Image.open(flare_path).convert("RGB").resize(blend.size)
    w, h = blend.size
    canvas = Image.new("RGB", (w * 2, h))
    canvas.paste(blend, (0, 0))
    canvas.paste(flare, (w, 0))
    canvas.save(out_path, format="JPEG", quality=90)


def main():
    total = 0
    for split in SPLITS:
        blend_dir = HERE / split / "blend"
        flare_dir = HERE / split / "flare"
        out_dir = HERE / split / "merged"
        out_dir.mkdir(exist_ok=True)

        files = sorted(blend_dir.glob("*.jpg"))
        print(f"[{split}] {len(files)} 張")
        for i, bpath in enumerate(files):
            fpath = flare_dir / bpath.name
            if not fpath.exists():
                continue
            merge_one(bpath, fpath, out_dir / bpath.name)
            total += 1
            if (i + 1) % 1000 == 0:
                print(f"  ...{i+1}/{len(files)}", end="\r")
        print()
    print(f"[done] 共合併 {total} 張圖片")


if __name__ == "__main__":
    main()
