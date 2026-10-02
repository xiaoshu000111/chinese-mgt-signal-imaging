# Colab + Google Drive + GitHub + 本地实验工作流

本项目不依赖实验室服务器。推荐把四个环境的职责固定下来：

| 环境 | 只负责什么 | 不负责什么 |
|---|---|---|
| 本地 Mac | 改代码、写文档、审查结果、提交 Git | 不保存大规模中间产物，不承担全量 GPU 计算 |
| GitHub | 保存代码、配置、实验协议、小型结果和文档 | 不保存原始数据、`.npz`、图片、checkpoint |
| Google Drive | 保存数据、模型缓存、信号、图片、权重、Colab 日志 | 不作为代码版本管理工具 |
| Colab | 从 GitHub 拉取代码，在 GPU 上运行固定实验 | 不直接修改并长期保存代码 |

## 一、Google Drive 目录约定

在 Drive 中建立一个目录。规范布局如下：

```text
MyDrive/chinese-mgt-signal-imaging/
├── raw/                         # train/dev/test/test_with_label.json
├── hf_cache/                    # Hugging Face 模型缓存
└── runs/
    ├── D0_smoke_qwen025b_T512/
    │   ├── processed/           # signals/、images/、scaling_*.json
    │   ├── splits/              # 本次运行的 manifest CSV
    │   ├── checkpoints/         # 本次运行权重
    │   ├── analysis/            # KS 表、曲线图等
    │   ├── outputs/             # ablation.csv 等小型输出
    │   └── logs/                # 本次运行原始日志
    └── D3_logp_gasf_qwen025b_T512/
```

如果上传的是完整项目文件夹，脚本也会自动识别
`chinese-mgt-signal-imaging/data/raw/`，因此不需要重新上传数据。规范布局中的
`raw/` 优先级更高；四个文件必须命名为 `train.json`、`dev.json`、`test.json` 和
`test_with_label.json`。

每个 `RUN_ID` 是一个不可混用的实验空间。基础模型、目标长度、信号通道或转图方法发生变化时，必须新建 `RUN_ID`，不能复用旧的 `processed/`。

## 二、第一次使用 Colab

先在本地完成代码修改并推送 GitHub。当前仓库还没有配置远程地址，因此把下面的地址替换为你的 GitHub 仓库地址：

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!git clone <YOUR_GITHUB_REPO_URL> /content/chinese-mgt-signal-imaging
%cd /content/chinese-mgt-signal-imaging
!bash scripts/colab_bootstrap.sh \
    --drive-root /content/drive/MyDrive/chinese-mgt-signal-imaging \
    --run-id D0_smoke_qwen025b_T512
```

`colab_bootstrap.sh` 会：

1. 创建当前 run 的 Drive 目录；
2. 生成未纳入 Git 的 `scripts/colab.env`；
3. 安装 `requirements-colab.txt`，不会覆盖 Colab 自带的 CUDA 版 PyTorch；
4. 把信号、图片、权重、日志等路径指向当前 run。

把四个 JSON 文件放到 Drive 的 `raw/` 或 `data/raw/` 目录后运行：

```bash
!bash scripts/colab_run.sh preflight
```

如果仓库是私有仓库，不要把 GitHub token 写进 notebook；使用 Colab Secrets 或先在本地下载代码后上传到 Colab。

## 三、标准实验执行顺序

一个新的 run 只允许对应一组固定的基础模型、目标长度、通道、转图方法和随机种子。

### D0：环境检查

```bash
!MGT_DEVICE=cuda MGT_MAX_SAMPLES=200 MGT_BATCH_SIZE=2 \
  bash scripts/colab_run.sh preflight
```

### D1：冒烟实验

```bash
!MGT_DEVICE=cuda MGT_MAX_SAMPLES=200 MGT_BATCH_SIZE=2 \
  bash scripts/colab_run.sh extract --name D1_extract --purpose "验证逐 token 概率信号提取、断点续跑和 Drive 写入"

!MGT_DEVICE=cuda MGT_MAX_SAMPLES=200 \
  bash scripts/colab_run.sh to_image --name D1_to_image --purpose "验证固定长度、分位数缩放和 logp/GASF 转图"

!MGT_DEVICE=cuda MGT_MAX_SAMPLES=200 MGT_EPOCHS=1 MGT_TRAIN_BATCH=16 \
  bash scripts/colab_run.sh train --name D1_train --purpose "验证 CNN 训练、checkpoint 写入和验证指标"

!MGT_DEVICE=cuda MGT_MAX_SAMPLES=200 \
  bash scripts/colab_run.sh eval --name D1_eval --purpose "验证评估链路；本次指标不作研究结论"
```

冒烟实验只检查链路，不比较方法优劣。OOM 时先降低 `MGT_BATCH_SIZE`，不要改模型结构。

### D2：信号质量自查

```bash
!MGT_DEVICE=cuda MGT_MAX_SAMPLES=2000 \
  bash scripts/colab_run.sh ks --name D2_ks --purpose "比较五条概率信号的人机可分性"
```

至少一条信号通道达到预先设定的 KS 决策门后，才进入全量提取。否则先更换基础 LM 或检查 tokenizer、special token 和缩放方式。

### D3：全量缓存

确认 D0/D1/D2 通过后，为同一 `RUN_ID` 执行：

```bash
!MGT_DEVICE=cuda MGT_MAX_SAMPLES=none MGT_BATCH_SIZE=4 \
  bash scripts/colab_run.sh extract --name D3_extract_full --purpose "一次性生成全量 token 概率信号缓存"

!MGT_DEVICE=cuda MGT_MAX_SAMPLES=none \
  bash scripts/colab_run.sh to_image --name D3_to_image_logp_gasf --purpose "生成固定配置的 logp/GASF 全量图像"
```

信号提取支持按 `.npz` 文件断点续跑。Colab 运行中断后，只要继续使用同一个 `RUN_ID`，不要删除已完成的文件。

### D4：首个 baseline

```bash
!MGT_DEVICE=cuda MGT_MAX_SAMPLES=none MGT_EPOCHS=20 \
  bash scripts/colab_run.sh train --name D4_baseline_train --purpose "logp/GASF baseline；只用 dev 早停"

!MGT_DEVICE=cuda MGT_MAX_SAMPLES=none \
  bash scripts/colab_run.sh eval --name D4_baseline_eval --purpose "在 test_with_label 上进行一次冻结口径评估"
```

### D5 之后：消融

一次只改变一个因素，例如只改 `MGT_CHANNEL_INDEX` 或 `MGT_METHOD`。每个组合使用新的 `RUN_ID`，并记录：

```text
基础模型、模型版本、tokenizer、TARGET_LEN、channel、method、seed、batch、epoch、学习率、代码 commit、Colab GPU、运行时长
```

推荐命名：

```text
D4_baseline_qwen025b_T512_logp_gasf_s42
D5_qwen025b_T512_ranknorm_gasf_s42
D5_qwen025b_T512_logp_mtf_s42
```

## 四、实验规范

### 1. 数据隔离

- `train`：允许拟合模型参数。
- `dev`：允许早停、选择超参数、选择阈值。
- `test_with_label`：只做预先声明的最终评估，不用于选模型或调阈值。
- `test.json`：盲测集，不进入当前本地有标签实验管线。

如果已经查看过某个 test 结果，就必须把它标记为 post-hoc 分析，不能再称为无偏的最终结果。

### 2. 缓存隔离

- 不同基础模型必须使用不同 `RUN_ID`。
- 不同 `TARGET_LEN`、通道、转图方法也建议使用不同 `RUN_ID`。
- 不要手动把一个 run 的 `scaling_*.json`、manifest 或 checkpoint 拷贝到另一个 run。
- Drive 是中间产物的唯一长期存储；GitHub 只保存小型结果。

### 3. 结果记录

每条命令必须带 `--name` 和 `--purpose`。日志头会自动写入代码版本、run ID、数据路径和关键配置。

实验结束后，把以下小文件从 Drive 取回本地并提交 Git：

```text
logs/ 或对应 run 的日志摘要
experiments/analysis/ 中的 CSV/PNG
experiments/ablation.csv
docs/06_实验日志.md 中的数字和结论
```

不要提交：原始 JSON、`.npz`、全量 PNG、Hugging Face cache、checkpoint。

### 4. 研究结论纪律

- 先看 AUC 和分组指标，再看固定阈值 F1；不要只看 accuracy。
- 任何“某方法更好”的结论至少需要 dev 选择、一次冻结 test 评估和对应配置记录。
- 主实验与消融必须使用同一数据切分、seed、长度策略和评估代码。
- 每次实验只回答一个问题；如果同时改了通道、长度和网络结构，结果不能归因。
- 训练失败、OOM、指标异常也要保留日志，并写明原因和处理方式。

## 五、建议的阶段决策门

| 阶段 | 通过条件 | 不通过时 |
|---|---|---|
| D1 冒烟 | 四步链路成功、产物写入 Drive、能断点续跑 | 只修环境/路径/显存，不改研究假设 |
| D2 信号自查 | 至少一条通道有稳定类间差异 | 检查 LM、tokenizer、精度和缩放 |
| D3 全量 | 样本数完整、无损坏信号、scaling 只来自 train | 修数据或缓存，不训练 CNN |
| D4 baseline | AUC 达到预设门槛且 train/dev gap 可解释 | 先诊断，不直接扩大消融网格 |
| D5-D7 消融 | 结果可重复，最优组合不是单次偶然 | 重跑 seed 或缩小实验范围 |

## 六、本地与 Colab 的日常循环

```text
本地修改代码/文档
    ↓
本地 pytest + git commit + git push
    ↓
Colab 新 checkout / git pull
    ↓
Drive 上建立新的 RUN_ID
    ↓
preflight → smoke → 正式实验
    ↓
从 Drive 取回小型结果与日志摘要
    ↓
本地更新 docs/06_实验日志.md，git commit
```

不要在 Colab 正在运行时执行 `git pull`，也不要在同一个 run 中修改代码后继续跑；需要改代码时结束当前 run，建立新 commit 和新 `RUN_ID`。
