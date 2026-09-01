"""② 读取 .npz 信号 -> 长度统一 -> 跨样本一致缩放 -> 转图 -> 存 png，并生成带路径的清单 CSV。

缩放映射在 train 上用分位数拟合并落盘 JSON，dev/test 复用同一份映射（docs/01 §2），
避免逐样本 min-max 把「人机整体水平差异」这一判别信息洗掉。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import config
from src.imaging.signal_to_image import signal_to_image

# 分位数缩放超参：用 1%/99% 分位做线性映射端点，截断离群值避免单样本极端值主导
SCALING_LO_Q = 0.01
SCALING_HI_Q = 0.99
# 拟合时最多读取的样本数（分位数估计无需全量，控制耗时）
SCALING_MAX_FILES = 20000


def pad_or_truncate(x: np.ndarray, T: int) -> np.ndarray:
    """统一到固定长度 T：过长截断，过短边缘反射补齐（不引入人为零纹理）。"""
    if len(x) >= T:
        return x[:T]
    return np.pad(x, (0, T - len(x)), mode="edge")


def load_signal(npz_path: Path, channel: str) -> np.ndarray:
    """读 .npz 里某条通道，返回一维 numpy 信号。"""
    data = np.load(npz_path)
    return data[channel]


def scaling_path(channel: str) -> Path:
    return config.DATA_PROCESSED / f"scaling_{channel}.json"


def fit_scaling(split: str, channel: str) -> tuple[float, float]:
    """在某个 split（应为 train）的信号上估计分位数端点，落盘 JSON 供所有 split 复用。"""
    npz_files = sorted((config.SIGNALS_DIR / split).glob("*.npz"))[:SCALING_MAX_FILES]
    if not npz_files:
        raise FileNotFoundError(f"{config.SIGNALS_DIR / split} 下没有 .npz，请先运行 extract")
    vals = [np.load(p)[channel].astype(np.float32) for p in npz_files]
    x = np.concatenate(vals)
    lo, hi = (float(v) for v in np.quantile(x, [SCALING_LO_Q, SCALING_HI_Q]))
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(scaling_path(channel), "w", encoding="utf-8") as f:
        json.dump({"channel": channel, "base_model": config.BASE_MODEL_NAME,
                   "fit_split": split,
                   "lo": lo, "hi": hi,
                   "lo_q": SCALING_LO_Q, "hi_q": SCALING_HI_Q,
                   "n_files": len(npz_files)}, f, ensure_ascii=False, indent=2)
    print(f"[to_image] 在 {split} 上拟合 {channel} 缩放: lo={lo:.6f} hi={hi:.6f} -> {scaling_path(channel)}")
    return lo, hi


def load_scaling(channel: str):
    p = scaling_path(channel)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        sc = json.load(f)
    # 换基础模型后旧映射必须重拟合，否则新信号被旧尺度错误压缩
    fit_model = sc.get("base_model")
    if fit_model and fit_model != config.BASE_MODEL_NAME:
        raise RuntimeError(
            f"scaling_{channel}.json 是用 {fit_model} 拟合的，与当前 {config.BASE_MODEL_NAME} 不符。"
            f"请删除 {p} 后重跑（train 的 to_image 会重新拟合）")
    return float(sc["lo"]), float(sc["hi"])


def apply_scaling(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """分位数线性映射到 [-1,1]（跨样本一致，离群值截断）。"""
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo) * 2.0 - 1.0, -1.0, 1.0)


def path_for_image(split: str, channel: str, method: str, sample_id: str) -> Path:
    return config.IMAGES_DIR / f"{channel}_{method}" / split / f"{sample_id}.png"


def run_split(split: str, channel: str, method: str):
    sig_dir = config.SIGNALS_DIR / split
    npz_files = sorted(sig_dir.glob("*.npz"))

    # 跨样本一致缩放：优先加载 train 拟合的映射；不存在则在当前 split 拟合（正常流程首次跑 train 时生成）
    sc = load_scaling(channel)
    if sc is None:
        print(f"[to_image] 未找到 {channel} 的缩放映射（首次运行？），在 {split} 上拟合")
        lo, hi = fit_scaling(split, channel)
    else:
        lo, hi = sc

    rows = []
    for npz_path in tqdm(npz_files, desc=f"to_image [{channel}/{method}] {split}"):
        sample_id = npz_path.stem
        data = np.load(npz_path)
        label = int(data["label"])
        model = str(data["model"]) if "model" in data.files else ""

        x = load_signal(npz_path, channel).astype(np.float32)
        x = pad_or_truncate(x, config.TARGET_LEN)
        x = apply_scaling(x, lo, hi)
        img = signal_to_image(x, method=method, scaled=True,
                              n_bins=config.MTF_N_BINS,
                              percentile=config.RP_PERCENTILE)   # (T, T) float [0,1]

        out_path = path_for_image(split, channel, method, sample_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((img * 255).astype(np.uint8), mode="L").save(out_path)

        rows.append({
            "sample_id": sample_id,
            "label": label,
            "model": model,
            "signal_path": str(npz_path),
            "image_path": str(out_path),
        })

    df = pd.DataFrame(rows)
    manifest = config.SPLITS_DIR / f"{split}_{channel}_{method}.csv"
    df.to_csv(manifest, index=False)
    print(f"[to_image] {split}: {len(df)} 张 -> {manifest}")


if __name__ == "__main__":
    channel = config.SIGNAL_CHANNELS[config.ACTIVE_CHANNEL_IDX]
    method = config.METHOD
    for split in ["train", "dev", "test_with_label"]:
        run_split(split, channel, method)