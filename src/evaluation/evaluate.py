"""④ 评估 + 消融：加载权重，在带标签测试集上算指标；可遍历通道×方法输出消融表。

F1 有两个口径（类别 3:1 不平衡，固定阈值 0.5 常常不是最优）：
- f1：固定阈值 0.5，与训练期观察口径一致
- f1_tuned：阈值在 dev 上按 F1 扫描调优后应用到 test（主报告口径）
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

import config
from src.models.cnn import SignalClassifier
from src.training.train import SignalImageDataset, collect_preds


def _make_loader(manifest: pd.DataFrame) -> DataLoader:
    return DataLoader(SignalImageDataset(manifest, config.N_CHANNELS),
                      batch_size=config.VAL_BATCH, shuffle=False, num_workers=config.NUM_WORKERS)


def _metrics_at(preds, labels, t):
    pred_label = (preds > t).astype(int)
    acc = (pred_label == labels).mean()
    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else float("nan")
    f1 = f1_score(labels, pred_label, zero_division=0)
    return acc, f1, auc


def _best_f1_threshold(preds, labels):
    """在 dev 上扫阈值 [0.05, 0.95]，返回 F1 最优的阈值。"""
    ts = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(labels, (preds > t).astype(int), zero_division=0) for t in ts]
    i = int(np.argmax(f1s))
    return float(ts[i])


def evaluate_ckpt(channel: str, method: str) -> dict:
    ckpt = config.CKPT_DIR / f"{channel}_{method}.pt"
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location=device)
    # in_channels 从 ckpt 里读而不是 config：防止「训练 3 通道、评估时 config 忘改」的静默错配
    in_ch = int(ck.get("config", {}).get("in_channels", config.N_CHANNELS))
    model = SignalClassifier(in_channels=in_ch).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    manifest = pd.read_csv(config.SPLITS_DIR / f"test_with_label_{channel}_{method}.csv")
    preds, labels = collect_preds(model, _make_loader(manifest), device)
    acc, f1, auc = _metrics_at(preds, labels, 0.5)

    # 阈值在 dev 上调优（test 不参与任何调参）
    threshold = 0.5
    dev_csv = config.SPLITS_DIR / f"dev_{channel}_{method}.csv"
    if dev_csv.exists():
        dev_preds, dev_labels = collect_preds(model, _make_loader(pd.read_csv(dev_csv)), device)
        threshold = _best_f1_threshold(dev_preds, dev_labels)
    _, f1_tuned, _ = _metrics_at(preds, labels, threshold)

    return {"channel": channel, "method": method, "acc": acc, "f1": f1, "auc": auc,
            "f1_tuned": f1_tuned, "threshold": threshold}


def ablation():
    """遍历所有通道 × 方法，输出消融表。前提：对应转图和权重都已生成。"""
    rows = []
    for channel in config.SIGNAL_CHANNELS:
        for method in ["gasf", "gadf", "mtf", "rp"]:
            ckpt = config.CKPT_DIR / f"{channel}_{method}.pt"
            if not ckpt.exists():
                rows.append({"channel": channel, "method": method, "acc": "N/A", "f1": "N/A",
                             "auc": "N/A", "f1_tuned": "N/A", "threshold": "N/A"})
                continue
            r = evaluate_ckpt(channel, method)
            rows.append(r)
    table = pd.DataFrame(rows)
    print("\n===== 消融表 (acc/f1/auc/f1_tuned@dev阈值) =====")
    print(table.round(4).to_string(index=False))
    out = config.OUTPUT_ROOT / "ablation.csv"
    table.to_csv(out, index=False)
    print(f"已保存 -> {out}")


if __name__ == "__main__":
    channel = config.SIGNAL_CHANNELS[config.ACTIVE_CHANNEL_IDX]
    method = config.METHOD
    r = evaluate_ckpt(channel, method)
    print(f"{channel}/{method}: acc={r['acc']:.4f} f1={r['f1']:.4f} auc={r['auc']:.4f} "
          f"f1_tuned={r['f1_tuned']:.4f}(阈值={r['threshold']:.2f})")
