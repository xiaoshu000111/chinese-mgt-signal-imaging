"""实验日志落盘。

命名：logs/YYYYMMDD_HHMMSS_<实验名>.log
头部：开始时间、来源（完整命令行 + git 版本 + 关键配置）、目的（purpose）。
之后进程内所有 stdout / stderr 同步写入该文件，tqdm 的原地刷新（\\r）只上屏不落盘。
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class _Tee:
    """把 stdout/stderr 同时写到终端和日志文件，过滤进度条的 \\r 刷新。"""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, data):
        self._stream.write(data)
        # tqdm 进度条用 \r 原地刷新：无换行的片段只上屏，不落盘
        if "\r" in data and "\n" not in data:
            return
        if "\r" in data:                      # 带 \n 的刷新行，取最终形态
            data = data.rsplit("\r", 1)[-1]
        self._log_file.write(data)
        self._log_file.flush()

    def flush(self):
        self._stream.flush()
        self._log_file.flush()

    def __getattr__(self, item):              # 透传 isatty 等属性给终端流
        return getattr(self._stream, item)


def _git_info():
    """当前代码版本（git 短 hash + 是否有未提交改动），非 git 仓库时返回提示。"""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if not commit:
            return "非 git 仓库"
        return f"git {commit}" + ("（有未提交改动）" if dirty else "")
    except Exception:
        return "git 不可用"


def start_experiment_log(name, purpose, meta=None):
    """开始记录一段实验日志，返回日志文件路径。

    name:    实验名（用于日志文件名，非法字符替换为 _）
    purpose: 这段实验的目的，写在日志头部
    meta:    关键配置 dict，逐行写入头部，便于事后追溯当时参数
    """
    import config

    logs_dir = config.PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if (c.isalnum() or c in "-_") else "_" for c in name)
    log_path = logs_dir / f"{ts}_{safe_name}.log"

    lines = [
        "=" * 78,
        f"实验名: {name}",
        f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"来源: {' '.join(sys.argv)}",
        f"代码版本: {_git_info()}",
        f"目的: {purpose}",
    ]
    if meta:
        lines.append("关键配置:")
        lines.extend(f"  {k}: {v}" for k, v in meta.items())
    lines.append("=" * 78)

    f = open(log_path, "a", encoding="utf-8")
    f.write("\n".join(lines) + "\n")
    f.flush()

    sys.stdout = _Tee(sys.stdout, f)
    sys.stderr = _Tee(sys.stderr, f)
    print(f"[log] 本段日志落盘 -> {log_path}")
    return log_path
