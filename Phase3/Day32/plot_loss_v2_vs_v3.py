"""
plot_loss_v2_vs_v3.py — v2(captions_train.jsonl)vs v3(captions_train_v3.jsonl)
訓練/驗證 loss 曲線疊圖比對。

train_vlm.py 本身沒把 per-epoch train loss 寫進 wandb 或 log 檔(只有 val_loss
有 wandb.log,train loss 只 print 到 stdout)——這裡直接從兩次訓練跑的完整
stdout 記錄(v2 是 wandb 幫忙存的 output.log,v3 是自己 nohup 重導向存的)
用 regex 撈 "validation loss: X" / "epoch N, loss: Y" 這兩行組成的 pair。
"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V2_LOG = "log/v2_train_stdout.log"
V3_LOG = "log/v3_train_stdout.log"
OUT_PNG = "log/loss_curve_v2_vs_v3.png"


def parse_log(path):
    train_losses, val_losses = [], []
    val_pending = None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"validation loss: ([\d.]+)", line)
        if m:
            val_pending = float(m.group(1))
            continue
        m = re.match(r"epoch (\d+), loss: ([\d.]+),", line)
        if m:
            epoch = int(m.group(1))
            train_losses.append((epoch, float(m.group(2))))
            val_losses.append((epoch, val_pending))
    return train_losses, val_losses


def main():
    v2_train, v2_val = parse_log(V2_LOG)
    v3_train, v3_val = parse_log(V3_LOG)

    print(f"v2: {len(v2_train)} epochs parsed")
    print(f"v3: {len(v3_train)} epochs parsed")

    fig, ax = plt.subplots(figsize=(9, 6))

    e2, t2 = zip(*v2_train)
    _, va2 = zip(*v2_val)
    e3, t3 = zip(*v3_train)
    _, va3 = zip(*v3_val)

    ax.plot(e2, t2, "o-", color="#4c72b0", label="v2 train loss")
    ax.plot(e2, va2, "o--", color="#4c72b0", alpha=0.6, label="v2 val loss")
    ax.plot(e3, t3, "s-", color="#dd8452", label="v3 train loss")
    ax.plot(e3, va3, "s--", color="#dd8452", alpha=0.6, label="v3 val loss")

    best_v2_epoch = min(v2_val, key=lambda p: p[1])
    best_v3_epoch = min(v3_val, key=lambda p: p[1])
    ax.scatter([best_v2_epoch[0]], [best_v2_epoch[1]], marker="*", s=250, color="#4c72b0", zorder=5,
               label=f"v2 best_model (epoch {best_v2_epoch[0]}, val={best_v2_epoch[1]:.4f})")
    ax.scatter([best_v3_epoch[0]], [best_v3_epoch[1]], marker="*", s=250, color="#dd8452", zorder=5,
               label=f"v3 best_model (epoch {best_v3_epoch[0]}, val={best_v3_epoch[1]:.4f})")

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("VLM training: v2 (v0.6 captions) vs v3 (v0.7 captions)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"寫出 {OUT_PNG}")
    print(f"v2 best: epoch={best_v2_epoch[0]} val_loss={best_v2_epoch[1]:.4f}")
    print(f"v3 best: epoch={best_v3_epoch[0]} val_loss={best_v3_epoch[1]:.4f}")


if __name__ == "__main__":
    main()
