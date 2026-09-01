"""转图与长度处理单元测试。"""
import numpy as np

from src.imaging.signal_to_image import gasf, gadf, mtf, recurrence_plot, signal_to_image
from src.imaging.to_image import pad_or_truncate


def test_pad_or_truncate():
    x = np.arange(10, dtype=np.float32)
    assert pad_or_truncate(x, 5).shape == (5,)
    padded = pad_or_truncate(x, 20)
    assert padded.shape == (20,)
    assert padded[10] == x[-1]   # 边缘反射：末值延展


def test_gaf_shapes_and_diag():
    x = np.random.randn(64).astype(np.float32)
    g = gasf(x)
    assert g.shape == (64, 64)
    assert np.allclose(g, g.T)   # GASF 对称


def test_mtf_shape():
    x = np.random.randn(64).astype(np.float32)
    m = mtf(x, n_bins=8)
    assert m.shape == (64, 64)
    assert m.min() >= 0 and m.max() <= 1


def test_recurrence_plot_binary():
    x = np.random.randn(64).astype(np.float32)
    r = recurrence_plot(x, percentile=20.0)
    assert set(np.unique(r)).issubset({0.0, 1.0})


def test_signal_to_image_range():
    x = np.random.randn(64).astype(np.float32)
    img = signal_to_image(x, method="gasf")
    assert img.min() >= 0 and img.max() <= 1


if __name__ == "__main__":
    test_pad_or_truncate()
    test_gaf_shapes_and_diag()
    test_mtf_shape()
    test_recurrence_plot_binary()
    test_signal_to_image_range()
    print("全部测试通过")