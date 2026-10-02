"""Run a cheap, read-only environment/data check before an experiment."""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path

import config


REQUIRED_RAW = tuple(config.RAW_FILES.values())


def _print_path(label: str, path: Path) -> None:
    exists = path.exists()
    marker = "OK" if exists else "MISSING"
    print(f"[{marker:7}] {label}: {path}")


def _check_packages() -> list[str]:
    packages = ["numpy", "pandas", "PIL", "sklearn", "torch", "transformers"]
    missing = []
    for name in packages:
        if importlib.util.find_spec(name) is None:
            missing.append(name)
    if missing:
        print(f"[MISSING] Python packages: {', '.join(missing)}")
    else:
        print("[OK     ] Python packages: numpy/pandas/Pillow/sklearn/torch/transformers")
    return missing


def _check_torch() -> bool:
    try:
        import torch

        print(f"[INFO   ] torch={torch.__version__}")
        print(f"[INFO   ] CUDA available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[INFO   ] CUDA device={torch.cuda.get_device_name(0)}")
        if hasattr(torch.backends, "mps"):
            print(f"[INFO   ] MPS available={torch.backends.mps.is_available()}")
    except Exception as exc:
        print(f"[MISSING] torch check failed: {exc}")
        return False

    if config.DEVICE == "cuda" and not torch.cuda.is_available():
        print("[ERROR  ] MGT_DEVICE=cuda, but CUDA is unavailable")
        return False
    return True


def main() -> None:
    print("=" * 78)
    print("Chinese MGT signal-imaging preflight (read-only)")
    print(f"Python: {sys.version.split()[0]} | platform: {platform.platform()}")
    print(f"run_id: {config.RUN_ID}")
    print(f"base_model: {config.BASE_MODEL_NAME}")
    print(f"channel/method: {config.SIGNAL_CHANNELS[config.ACTIVE_CHANNEL_IDX]}/{config.METHOD}")
    print(f"target_len/batch/max_samples: {config.TARGET_LEN}/{config.BATCH_SIZE}/{config.MAX_SAMPLES}")
    print("-" * 78)

    for name, path in (
        ("raw data", config.DATA_RAW),
        ("processed", config.DATA_PROCESSED),
        ("splits", config.SPLITS_DIR),
        ("output", config.OUTPUT_ROOT),
        ("logs", config.LOG_DIR),
    ):
        _print_path(name, path)

    missing_raw = []
    for filename in REQUIRED_RAW:
        path = config.DATA_RAW / filename
        if not path.is_file() or path.stat().st_size == 0:
            missing_raw.append(filename)
        else:
            print(f"[OK     ] raw/{filename}: {path.stat().st_size / 1024**2:.1f} MiB")

    missing_packages = _check_packages()
    torch_ok = _check_torch()

    for path in (config.DATA_PROCESSED, config.OUTPUT_ROOT, config.LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)
        total, used, free = shutil.disk_usage(path)
        print(f"[INFO   ] free space at {path}: {free / 1024**3:.1f} GiB")

    print("-" * 78)
    if missing_raw:
        print(f"[ERROR  ] missing raw files: {', '.join(missing_raw)}")
    if missing_packages:
        print("[ERROR  ] install requirements before running experiments")
    if not torch_ok:
        print("[ERROR  ] device check failed")

    if missing_raw or missing_packages or not torch_ok:
        raise SystemExit(1)
    print("[PASS   ] preflight passed")


if __name__ == "__main__":
    main()
