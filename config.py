"""全局配置：路径、基础模型、信号通道、转图与训练超参。

所有模块统一从这里读配置，避免参数在多个文件里漂移。

路径支持通过环境变量覆盖，供 Colab + Google Drive 使用。默认值仍然是
仓库内的本地目录，因此本地运行方式不变。推荐的 Colab 做法是让每个实验
设置独立的 ``MGT_RUN_ID``，并让各个 Drive 目录环境变量指向该 run，避免不同基础模型或转图配置混用缓存。
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _path_env(name: str, default: Path) -> Path:
    """读取路径环境变量，并展开用户目录/绝对路径。"""
    return Path(os.environ.get(name, str(default))).expanduser()


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _float_env(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


# 数据与产物根目录可以指向 Google Drive；本地默认路径保持兼容。
DATA_RAW = _path_env("MGT_DATA_RAW", PROJECT_ROOT / "data" / "raw")
DATA_PROCESSED = _path_env("MGT_PROCESSED_ROOT", PROJECT_ROOT / "data" / "processed")
SIGNALS_DIR = DATA_PROCESSED / "signals"
IMAGES_DIR = DATA_PROCESSED / "images"
SPLITS_DIR = _path_env("MGT_SPLITS_ROOT", PROJECT_ROOT / "data" / "splits")

# 结果、权重、日志也可独立写入 Drive 的当前 run 目录。
OUTPUT_ROOT = _path_env("MGT_OUTPUT_ROOT", PROJECT_ROOT / "experiments")
ANALYSIS_DIR = _path_env("MGT_ANALYSIS_ROOT", OUTPUT_ROOT / "analysis")
LOG_DIR = _path_env("MGT_LOG_ROOT", PROJECT_ROOT / "logs")
CKPT_DIR = _path_env("MGT_CHECKPOINT_ROOT", OUTPUT_ROOT / "checkpoints")
RUN_ID = os.environ.get("MGT_RUN_ID", "local")

# ---- 数据集文件名（保持你原始文件名不变）----
RAW_FILES = {
    "train": "train.json",
    "dev": "dev.json",
    "test": "test.json",
    "test_with_label": "test_with_label.json",
}

# ---- 基础模型（用于提取概率信号）----
BASE_MODEL_NAME = os.environ.get("MGT_BASE_MODEL", "Qwen/Qwen2.5-0.5B")
MAX_SEQ_LEN = _int_env("MGT_MAX_SEQ_LEN", 512)
DEVICE = os.environ.get("MGT_DEVICE", "cuda")
BATCH_SIZE = _int_env("MGT_BATCH_SIZE", 8)
_max_samples = os.environ.get("MGT_MAX_SAMPLES", "none").strip().lower()
MAX_SAMPLES = None if _max_samples in {"", "none", "null", "all"} else int(_max_samples)

# ---- 信号通道（一维信号）----
SIGNAL_CHANNELS = ["logp", "rank", "rank_norm", "entropy", "top_prob"]
MIN_LOGIT = 1e-9                            # 防止 log(0)

# ---- 转图参数 ----
# 目标信号长度 T（图像边长）。长文截断、短文边缘补齐到该值。
TARGET_LEN = _int_env("MGT_TARGET_LEN", 512)
# 默认转图方法：gasf | gadf | mtf | rp
METHOD = os.environ.get("MGT_METHOD", "gasf")
MTF_N_BINS = _int_env("MGT_MTF_BINS", 8)
RP_PERCENTILE = _float_env("MGT_RP_PERCENTILE", 20.0)

# ---- 训练超参 ----
IMG_SIZE = _int_env("MGT_IMG_SIZE", TARGET_LEN)
N_CHANNELS = _int_env("MGT_N_CHANNELS", 1)
EPOCHS = _int_env("MGT_EPOCHS", 20)
LR = _float_env("MGT_LR", 1e-3)
WEIGHT_DECAY = _float_env("MGT_WEIGHT_DECAY", 1e-4)
TRAIN_BATCH = _int_env("MGT_TRAIN_BATCH", 32)
VAL_BATCH = _int_env("MGT_VAL_BATCH", 64)
NUM_WORKERS = _int_env("MGT_NUM_WORKERS", 2)
SEED = _int_env("MGT_SEED", 42)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# 简单开关：用于消融时快速切换信号通道编号。
# 0=logp 1=rank 2=rank_norm 3=entropy 4=top_prob
ACTIVE_CHANNEL_IDX = _int_env("MGT_CHANNEL_INDEX", 0)
