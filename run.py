"""命令行主入口：
  python run.py extract   提取信号
  python run.py to_image  信号转图
  python run.py ks        信号质量自查（各通道类间 KS + logp 曲线图，D2 决策门依据）
  python run.py train     训练
  python run.py eval      评估（含可选 ablation）

每次运行自动落盘日志：logs/YYYYMMDD_HHMMSS_<实验名>.log，
头部写入来源（完整命令行 + git 版本 + 关键配置）与目的（--purpose）。

  python run.py train --name baseline_v1 --purpose "D4 logp/GASF 基线，验证决策门 AUC>=0.75"
"""
import argparse


# 各命令的默认实验名 / 默认目的（未显式传 --name / --purpose 时使用）
DEFAULTS = {
    "extract": ("extract", "用基础 LM 提取逐 token 概率信号，落盘 .npz 缓存"),
    "to_image": ("to_image", "读取 .npz 信号，统一长度 + 跨样本缩放后转图存 png 并生成 manifest"),
    "ks": ("ks", "信号质量自查：各通道类间 KS 统计量 + 人工/机器 logp 曲线图"),
    "train": ("train", "训练 CNN 二分类器，按 val AUC 保存最优权重"),
    "eval": ("eval", "加载权重在 test_with_label 上评测 acc/f1/auc（含 dev 调优阈值的 f1_tuned）"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="help",
                        choices=["extract", "to_image", "ks", "train", "eval", "help"])
    parser.add_argument("extra", nargs="*", help="额外位置参数，如 eval ablation")
    parser.add_argument("--name", help="实验名（日志文件名一部分；默认 按命令+当前配置自动生成）")
    parser.add_argument("--purpose", default="", help="本段实验的目的，写入日志头部")
    parser.add_argument("--split", default="train", help="ks 命令：用哪个 split 的信号（默认 train）")
    parser.add_argument("--max-n", type=int, default=3000, help="ks 命令：最多读多少条信号（默认 3000）")
    args = parser.parse_args()

    if args.command == "help":
        print(__doc__)
        return

    import config
    from src.utils.run_logger import start_experiment_log

    channel = config.SIGNAL_CHANNELS[config.ACTIVE_CHANNEL_IDX]
    method = config.METHOD
    default_name, default_purpose = DEFAULTS[args.command]

    # 实验名默认带通道/方法，日志文件名即可分辨是哪组实验
    if args.name:
        name = args.name
    elif args.command in ("to_image", "train", "eval"):
        name = f"{default_name}_{channel}_{method}"
    else:
        name = default_name
    purpose = args.purpose or default_purpose

    # 头部关键配置：按命令记录各自最影响复现的参数
    if args.command == "extract":
        meta = {
            "base_model": config.BASE_MODEL_NAME, "max_seq_len": config.MAX_SEQ_LEN,
            "batch_size": config.BATCH_SIZE, "max_samples": config.MAX_SAMPLES,
            "device": config.DEVICE,
        }
    elif args.command == "to_image":
        meta = {
            "channel": channel, "method": method, "target_len": config.TARGET_LEN,
            "mtf_n_bins": config.MTF_N_BINS, "rp_percentile": config.RP_PERCENTILE,
            "scaling": "train 分位数(1%/99%)，复用 data/processed/scaling_*.json",
        }
    elif args.command == "ks":
        meta = {"split": args.split, "max_n": args.max_n,
                "base_model": config.BASE_MODEL_NAME,
                "decision_gate": "至少一条通道 KS >= 0.3，否则换基础模型"}
    elif args.command == "train":
        meta = {
            "channel": channel, "method": method, "img_size": config.IMG_SIZE,
            "n_channels": config.N_CHANNELS, "epochs": config.EPOCHS,
            "lr": config.LR, "train_batch": config.TRAIN_BATCH, "seed": config.SEED,
        }
    else:  # eval
        meta = {
            "channel": channel, "method": method,
            "ckpt": str(config.CKPT_DIR / f"{channel}_{method}.pt"),
            "ablation": "ablation" in args.extra,
        }

    start_experiment_log(name, purpose, meta)

    if args.command == "extract":
        from src.signals.extract_signals import SignalExtractor
        ext = SignalExtractor()
        for split in ["train", "dev", "test_with_label"]:
            ext.run_split(split)
    elif args.command == "to_image":
        from src.imaging.to_image import run_split
        for split in ["train", "dev", "test_with_label"]:
            run_split(split, channel, method)
    elif args.command == "ks":
        from src.analysis.signal_stats import run as ks_run
        ks_run(split=args.split, max_n=args.max_n)
    elif args.command == "train":
        from src.training.train import main as train_main
        train_main()
    elif args.command == "eval":
        if "ablation" in args.extra:
            from src.evaluation.evaluate import ablation
            ablation()
        else:
            from src.evaluation.evaluate import evaluate_ckpt
            r = evaluate_ckpt(channel, method)
            print(f"{channel}/{method}: acc={r['acc']:.4f} f1={r['f1']:.4f} auc={r['auc']:.4f} "
                  f"f1_tuned={r['f1_tuned']:.4f}(阈值={r['threshold']:.2f}，dev 调优)")


if __name__ == "__main__":
    main()
