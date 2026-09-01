"""全局配置：路径、基础模型、信号通道、转图与训练超参。

所有模块统一从这里读配置，避免参数在多个文件里漂移。
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SIGNALS_DIR = DATA_PROCESSED / "signals"
IMAGES_DIR = DATA_PROCESSED / "images"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

# ---- 数据集文件名（保持你原始文件名不变）----
RAW_FILES = {
    "train": "train.json",
    "dev": "dev.json",
    "test": "test.json",
    "test_with_label": "test_with_label.json",
}

# ---- 基础模型（用于提取概率信号）----
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B"      # 中文因果 LM，约 1GB
MAX_SEQ_LEN = 512                           # 一次前向最多 token 数
DEVICE = "cuda"                             # 无 GPU 改 "cpu"
BATCH_SIZE = 8    # 提取批大小；后处理为 fp32 的 [B,T,V] 大张量，批太大会撑爆显存
MAX_SAMPLES = None                          # 调试时设 None=全部；可设 200 快速跑通

# ---- 信号通道（一维信号）----
SIGNAL_CHANNELS = ["logp", "rank", "rank_norm", "entropy", "top_prob"]
MIN_LOGIT = 1e-9                            # 防止 log(0)

# ---- 转图参数 ----
# 目标信号长度 T（图像边长）。长文截断、短文边缘补齐到该值。
TARGET_LEN = 512
# 默认转图方法：gasf | gadf | mtf | rp
METHOD = "gasf"
MTF_N_BINS = 8                              # MTF 分位数分箱数
RP_PERCENTILE = 20                          # 递归图阈值使用的距离百分位数

# ---- 训练超参 ----
IMG_SIZE = TARGET_LEN                       # 图像边长（与 TARGET_LEN 一致）
N_CHANNELS = 1                              # 单通道起步；三通道拼 RGB 时改 3
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_BATCH = 32
VAL_BATCH = 64
NUM_WORKERS = 4
SEED = 42
CKPT_DIR = PROJECT_ROOT / "experiments" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# 简单开关：用于消融时快速切换信号通道编号。
# 0=logp 1=rank 2=rank_norm 3=entropy 4=top_prob
ACTIVE_CHANNEL_IDX = 0