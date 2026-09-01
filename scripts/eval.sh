#!/usr/bin/env bash
# ④ 评估（默认单配置）；加参数 ablation 跑消融
set -e
cd "$(dirname "$0")/.."
python run.py eval "$@"