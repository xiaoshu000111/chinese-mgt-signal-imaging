#!/usr/bin/env bash
# 服务器端实验入口：自动带 Python 路径与 HF 缓存，用法与 run.py 一致
#   ./scripts/server.sh extract --name smoke_extract --purpose "..."
# 用它等价于：
#   export HF_HOME=/data/DDing/hf_cache
#   /data/miniconda3/envs/dding/bin/python run.py ...
set -e
cd "$(dirname "$0")/.."
export HF_HOME=/data/DDing/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
PY=/data/miniconda3/envs/dding/bin/python
exec "$PY" run.py "$@"
