#!/usr/bin/env bash
# ① 提取概率信号（需先把 4 个 json 放到 data/raw/）
set -e
cd "$(dirname "$0")/.."
python run.py extract