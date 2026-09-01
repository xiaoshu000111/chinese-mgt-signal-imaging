"""② 一维信号转二维图像：GASF / GADF / MTF / RecurrencePlot。

纯 numpy 实现，不依赖 pyts，便于看懂每一步、方便答辩讲原理。
"""
import numpy as np


def scale_minmax(x: np.ndarray, lower: float = -1.0, upper: float = 1.0) -> np.ndarray:
    """线性缩放到 [lower, upper]。常数序列会退化为全 0，避免除零。"""
    xmin, xmax = x.min(), x.max()
    if xmax - xmin < 1e-12:
        return np.zeros_like(x)
    return lower + (x - xmin) / (xmax - xmin) * (upper - lower)


def _phi(x_scaled: np.ndarray) -> np.ndarray:
    """把 [-1,1] 映射到 [0, pi]，用于 GAF。"""
    return np.arccos(np.clip(x_scaled, -1.0, 1.0))


def gasf(x: np.ndarray, scaled: bool = False) -> np.ndarray:
    """Gramian Angular Summation Field，返回 (T, T)。

    scaled=True 时认为输入已在 [-1,1]（跨样本一致的缩放，见 to_image.apply_scaling），
    仅做 clip 防越界；否则退回逐样本 min-max（仅测试/临时调试用）。
    """
    if not scaled:
        x = scale_minmax(x)
    phi = _phi(np.clip(x, -1.0, 1.0))
    return np.cos(np.add.outer(phi, phi))


def gadf(x: np.ndarray, scaled: bool = False) -> np.ndarray:
    """Gramian Angular Difference Field，返回 (T, T)。"""
    if not scaled:
        x = scale_minmax(x)
    phi = _phi(np.clip(x, -1.0, 1.0))
    return np.sin(np.subtract.outer(phi, phi))


def _quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """分位数分箱，返回每个时刻的 bin 索引 0..n_bins-1。"""
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    q = np.digitize(x, edges[1:-1])
    return np.clip(q, 0, n_bins - 1)


def mtf(x: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Markov Transition Field，返回 (T, T)。

    先估计状态转移矩阵 W，再把网格点 (i,j) 填成 W[q_i][q_j]。
    """
    q = _quantile_bins(x, n_bins)
    W = np.zeros((n_bins, n_bins))
    for i in range(len(q) - 1):
        W[q[i], q[i + 1]] += 1
    row_sum = W.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    W = W / row_sum
    return W[np.ix_(q, q)]


def recurrence_plot(x: np.ndarray, percentile: float = 20.0) -> np.ndarray:
    """RecurrencePlot，返回 (T, T) 二值矩阵。阈值按成对距离的百分位数确定。"""
    D = np.abs(np.subtract.outer(x, x))
    eps = np.percentile(D, percentile)
    return (D <= eps).astype(np.float32)


def signal_to_image(x: np.ndarray, method: str = "gasf", scaled: bool = False, **kwargs) -> np.ndarray:
    """单条一维信号 -> (T, T) 图像矩阵（float，范围 [0,1]）。

    scaled=True 时输入应已在 [-1,1]（跨样本一致缩放）。输出用固定映射而非逐图
    min-max 拉伸，保证同一数值在所有图像中灰度一致（docs/01 §2）。
    """
    if method == "gasf":
        img = gasf(x, scaled=scaled)
    elif method == "gadf":
        img = gadf(x, scaled=scaled)
    elif method == "mtf":
        img = mtf(x, n_bins=kwargs.get("n_bins", 8))       # 分位数分箱，对单调缩放不变
    elif method == "rp":
        img = recurrence_plot(x, percentile=kwargs.get("percentile", 20.0))
    else:
        raise ValueError(f"未知转图方法: {method}")
    # 统一到 [0,1]：GAF 天然落在 [-1,1]，用固定 (x+1)/2；MTF/RP 本就在 [0,1]，仅截断防越界
    if method in ("gasf", "gadf"):
        img = (img + 1.0) / 2.0
    else:
        img = np.clip(img, 0.0, 1.0)
    return img


def stack_rgb(channel_imgs: list[np.ndarray]) -> np.ndarray:
    """把 3 个 (T,T) 矩阵拼成 (T,T,3) RGB 图像。用于多通道消融。"""
    assert len(channel_imgs) == 3, "RGB 需要恰好 3 个通道"
    return np.stack(channel_imgs, axis=-1)