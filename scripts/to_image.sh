#!/usr/bin/env bash
# ② 信号转图像
set -e
cd "$(dirname "$0")/.."
python run.py to_image