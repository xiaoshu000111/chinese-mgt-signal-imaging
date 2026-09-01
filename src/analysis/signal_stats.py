"""D2 信号质量自查：各通道类间 KS 统计量 + 人工/机器 logp 曲线叠加图。

产出：
- experiments/analysis/ks_{split}.csv         5 通道类间 KS，降序
- experiments/analysis/logp_curves_{split}.png  前 3 条人工 vs 机器的 logp 曲线

KS 越大说明该通道人机分布分得越开，是「信号可分性」的量化依据（决策门 KS >= 0.3）。
"""
import numpy as np
import pandas as pd

import config

# 曲线图最多各画几条样本
N_CURVES_PER_CLASS = 3


def ks_2samp(a, b) -> float:
    """双样本 KS 统计量 sup|F_a - F_b|（手写避免引入 scipy 依赖）。"""
    a, b = np.sort(a), np.sort(b)
    all_v = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_v, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_v, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def run(split: str = "train", max_n: int = 3000):
    sig_dir = config.SIGNALS_DIR / split
    npz_files = sorted(sig_dir.glob("*.npz"))[:max_n]
    if not npz_files:
        raise FileNotFoundError(f"{sig_dir} 下没有 .npz，请先运行 python run.py extract")

    per_channel = {c: {"human": [], "machine": []} for c in config.SIGNAL_CHANNELS}
    curves = {"human": [], "machine": []}
    for p in npz_files:
        d = np.load(p)
        lab = int(d["label"])
        if lab not in (0, 1):
            continue    # 无标签样本不参与
        key = "machine" if lab == 1 else "human"
        for c in config.SIGNAL_CHANNELS:
            per_channel[c][key].append(d[c].astype(np.float32))
        if len(curves[key]) < N_CURVES_PER_CLASS:
            curves[key].append(d["logp"].astype(np.float32))

    rows = []
    for c in config.SIGNAL_CHANNELS:
        h, m = per_channel[c]["human"], per_channel[c]["machine"]
        if not h or not m:
            print(f"[ks] 跳过 {c}：某一类无样本")
            continue
        rows.append({"channel": c, "ks": ks_2samp(np.concatenate(h), np.concatenate(m)),
                     "n_human": len(h), "n_machine": len(m)})

    out_dir = config.PROJECT_ROOT / "experiments" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("ks", ascending=False)
    csv_path = out_dir / f"ks_{split}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n===== KS 统计量（越大越可分，决策门 >= 0.3）=====")
    print(df.round(4).to_string(index=False))
    print(f"已保存 -> {csv_path}")

    # logp 曲线叠加图：人工应出现更明显的下探（意外词）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过曲线图（pip install matplotlib）")
        return csv_path

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, s in enumerate(curves["human"]):
        ax.plot(s, color="tab:blue", alpha=0.7, lw=1,
                label="human" if i == 0 else None)
    for i, s in enumerate(curves["machine"]):
        ax.plot(s, color="tab:orange", alpha=0.7, lw=1,
                label="machine" if i == 0 else None)
    # 标签用英文：matplotlib 默认字体无 CJK glyph，中文会渲染成方框
    ax.set_xlabel("token position")
    ax.set_ylabel("logp")
    ax.set_title(f"{split}: human vs machine logp curves (first {N_CURVES_PER_CLASS} each)")
    ax.legend()
    png_path = out_dir / f"logp_curves_{split}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"曲线图 -> {png_path}")
    return csv_path


if __name__ == "__main__":
    run()
