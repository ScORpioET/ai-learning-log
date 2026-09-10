"""
畫thermal vs RGB(含night子集標出來)的channel_diff分布圖。

thermal這邊全部精確等於0.0(檔案格式層級的單通道JPEG,不是統計上接近0),
畫在同一張線性刻度的圖上會變成一根貼著y軸的尖峰,看不出RGB那邊的分布形狀,
所以拆成兩個panel:左邊全範圍(log-y軸,才看得到thermal的尖峰跟RGB的分布
同時存在)、右邊放大0~15的範圍(這才是實際判斷門檻要看的區域,清楚show出
thermal的0跟RGB最小值之間有多少間隙)。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("sampled_measurements.csv")
thermal = df[df.domain == "thermal"]["channel_diff"].values
rgb_day = df[(df.domain == "rgb") & (~df.is_night_tag)]["channel_diff"].values
rgb_night = df[(df.domain == "rgb") & (df.is_night_tag)]["channel_diff"].values

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

bins_full = np.linspace(0, 95, 96)
ax = axes[0]
ax.hist(thermal, bins=bins_full, alpha=0.7, label=f"thermal (n={len(thermal)}, all exactly 0.0)", color="#4c72b0")
ax.hist(rgb_day, bins=bins_full, alpha=0.6, label=f"rgb - day/no night-tag (n={len(rgb_day)})", color="#dd8452")
ax.hist(rgb_night, bins=bins_full, alpha=0.6, label=f"rgb - night-tagged (n={len(rgb_night)})", color="#c44e52")
ax.set_yscale("log")
ax.set_xlabel("channel_diff = max(mean|R-G|, mean|G-B|, mean|R-B|)")
ax.set_ylabel("count (log scale)")
ax.set_title("Full range (log y-axis)")
ax.legend(fontsize=9)

bins_zoom = np.linspace(0, 15, 76)
ax = axes[1]
ax.hist(thermal, bins=bins_zoom, alpha=0.7, label="thermal", color="#4c72b0")
ax.hist(rgb_day, bins=bins_zoom, alpha=0.6, label="rgb - day/no night-tag", color="#dd8452")
ax.hist(rgb_night, bins=bins_zoom, alpha=0.6, label="rgb - night-tagged", color="#c44e52")
ax.axvline(rgb_day.min() if len(rgb_day) else rgb_night.min(), color="gray", linestyle="--", linewidth=1)
rgb_all_min = min(rgb_day.min(), rgb_night.min())
ax.axvline(rgb_all_min, color="black", linestyle="--", linewidth=1.2,
           label=f"RGB overall min={rgb_all_min:.2f}")
ax.set_xlabel("channel_diff (zoomed to 0-15)")
ax.set_ylabel("count")
ax.set_title("Threshold decision zone (zoomed)\n(thermal always = 0.0, RGB min stays well above 0)")
ax.legend(fontsize=9)

plt.tight_layout()
out_path = "channel_diff_distribution.png"
plt.savefig(out_path, dpi=150)
print(f"[done] saved {out_path}")
