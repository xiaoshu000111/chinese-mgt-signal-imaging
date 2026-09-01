"""数据加载与 manifest 构建。

把原始 json 读成统一结构，并导出 CSV 清单供训练/评估使用。
"""
import json
from pathlib import Path

import pandas as pd

import config


def load_split(split: str) -> list[dict]:
    """读取一个 split 的原始 json，返回样本列表（字段随文件不同略有差异）。"""
    path = config.DATA_RAW / config.RAW_FILES[split]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for i, item in enumerate(data):
        rec = {
            "sample_id": f"{split}_{i}",
            "split": split,
            "text": item["text"],
            "label": item.get("label"),            # test.json 无 label
            "model": item.get("model"),            # 仅 train.json 有
            "source": item.get("source"),          # 仅 train.json 有
            "external_id": item.get("id"),         # 仅 test* 有
        }
        out.append(rec)
    return out


def build_manifests() -> dict[str, pd.DataFrame]:
    """把所有 split 读入并落 CSV 清单（训练阶段只读 CSV，不扫 json）。"""
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for split in config.RAW_FILES:
        recs = load_split(split)
        df = pd.DataFrame(recs)
        csv_path = config.SPLITS_DIR / f"{split}.csv"
        df.to_csv(csv_path, index=False)
        manifests[split] = df
        print(f"[load_data] {split}: {len(df)} 条 -> {csv_path}")
    return manifests


if __name__ == "__main__":
    build_manifests()