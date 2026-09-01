# 中文机器生成文本检测：概率信号 → 图像 → CNN

本仓库是实现「一维概率信号转二维图像（GAF/MTF/递归图）+ CNN 分类」路线的完整脚手架。
核心假设：**人工与机器生成的中文文本，在预训练语言模型下的逐 token 概率信号存在可分离的分布差异；把这条一维信号编码成二维图像后，CNN 能学到人机之间的局部与全局纹理差异。**

## 四步路线

| 步骤 | 目标 | 核心产出 | 对应代码 |
|------|------|----------|----------|
| ① 信号提取 | 用中文因果 LM 得到逐 token 概率信号（多通道一维序列） | `.npz` 信号文件 | `src/signals/extract_signals.py` |
| ② 转图 | 把一维信号编码为 GAF / MTF / 递归图 | `.png` 图像 | `src/imaging/signal_to_image.py` |
| ③ 分类 | 用 CNN 对图像做二分类（人工 / 机器） | 模型权重、评测指标 | `src/models/cnn.py`、`src/training/train.py` |
| ④ 评估消融 | 对比不同信号通道、不同转图方法的贡献 | 消融表、混淆矩阵 | `src/evaluation/evaluate.py` |

## 目录结构

```
chinese-mgt-signal-imaging/
├── README.md                 # 本文件：规划总览与快速开始
├── requirements.txt          # 依赖
├── config.py                 # 全局配置入口（路径、模型、超参）
├── run.py                    # 命令行主入口（extract/to_image/train/eval）
├── data/
│   ├── raw/                  # 原始数据集（把 4 个 json 放到这里）
│   ├── processed/
│   │   ├── signals/          # ① 阶段产出的 .npz 信号
│   │   └── images/           # ② 阶段产出的 .png 图像
│   └── splits/               # train/val/test 的 manifest.csv
├── docs/
│   ├── 01_实现准则.md          # 具体实现准则（可复现、确定性、选型）
│   ├── 02_信号构建.md          # 如何构建一维信号（公式 + 含义）
│   ├── 03_转图方法.md          # GAF / MTF / 递归图 原理与选择
│   ├── 04_长度处理.md          # 不同长度文本怎么处理
│   ├── 05_实验计划.md          # 按天的实验路线图（决策门、风险应对）
│   └── 06_实验日志.md          # 结果与结论的持续记录（当天实验当天记）
├── src/
│   ├── data/load_data.py     # 数据加载与 manifest 构建
│   ├── signals/extract_signals.py
│   ├── imaging/signal_to_image.py
│   ├── models/cnn.py
│   ├── training/train.py
│   ├── evaluation/evaluate.py
│   └── utils/run_logger.py   # 实验日志落盘（时间戳_实验名.log）
├── experiments/configs/      # yaml 实验配置
├── scripts/                  # 快捷 shell 脚本
├── logs/                     # 每次运行的完整日志（自动生成，只增不删）
└── tests/                    # 单元测试
```

## 快速开始

实验怎么排、每天做什么、结果记在哪：先读 [docs/05_实验计划.md](docs/05_实验计划.md)（按天路线图）与 [docs/06_实验日志.md](docs/06_实验日志.md)（结果记录）。

```bash
# 1. 安装依赖
pip install -r requirements.txt --break-system-packages

# 2. 把 4 个数据集 json 复制到 data/raw/ 下（文件名保持 train/dev/test/test_with_label）

# 3. 走完四步（每一步都会把中间产物缓存到 data/processed/）
#    每次运行自动落盘日志 logs/YYYYMMDD_HHMMSS_<实验名>.log（头部含来源与目的）
python run.py extract --name smoke_extract --purpose "冒烟：验证提取链路"  # ① 提取概率信号 -> .npz
python run.py to_image   # ② 信号转图像   -> .png
python run.py train      # ③ 训练 CNN
python run.py eval       # ④ 评估 + 消融
```

首次运行会从 HuggingFace 下载默认基础模型 `Qwen/Qwen2.5-0.5B`（可换成 `Qwen2.5-0.5B-Instruct` 等，见 `config.py`）。若无法翻墙，先设置 `HF_ENDPOINT=https://hf-mirror.com`。

## 数据字段说明

| 文件 | 字段 | 说明 |
|------|------|------|
| `train.json` | `text / label / model / source` | `label` 0=人工 1=机器；`model` 记录机器来源（如 gpt4o）或 human |
| `dev.json` | `text / label` | 验证集 |
| `test.json` | `text / id` | 盲测集（无标签；本课题不提交预测，不进管线） |
| `test_with_label.json` | `text / label / id` | 带标签测试集，用于本地评测 |

## 关键实现要点（详见 docs/）

- **信号通道**：默认提取 5 条通道 `logp / rank / rank_norm / entropy / top_prob`，每条长度 = token 数 T。
- **转图方法**：GASF、GADF、MarkovTransitionField、RecurrencePlot 四种，可单通道也可三通道拼 RGB，供消融。
- **长度处理**：训练期统一截断/补齐到固定 T（推荐 512），补长用边缘值反射避免引入人为零纹理；长文本用滑窗 + 预测聚合。
- **可复现**：全链路固定随机种子、固定 tokenizer、信号落在磁盘缓存，训练时只读缓存。