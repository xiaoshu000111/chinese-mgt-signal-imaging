#!/usr/bin/env bash
# 本地 <-> 云服务器 同步（git 只在本地，服务器不做版本管理）
#
# 用法：
#   ./scripts/sync.sh push         代码/文档 -> 服务器（改完本地代码后执行）
#   ./scripts/sync.sh push-data     data/raw -> 服务器（仅首次）
#   ./scripts/sync.sh pull          logs + 实验结果(experiments 除权重) -> 本地（每天收工）
#   ./scripts/sync.sh pull-signals  信号缓存 -> 本地（D3 备份要求，约 1GB）
#   ./scripts/sync.sh pull-ckpt    模型权重 -> 本地（按需，单个文件几百 MB）
#
# 连接配置：复制 scripts/sync.config.example 为 scripts/sync.config 并填写
#（该文件已 gitignore，含服务器地址不会被提交）
set -e
cd "$(dirname "$0")/.."

if [ -f scripts/sync.config ]; then
  source scripts/sync.config
fi
if [ -z "$SYNC_USER" ] || [ -z "$SYNC_HOST" ]; then
  echo "错误：未配置服务器信息。请 cp scripts/sync.config.example scripts/sync.config 并填写" >&2
  exit 1
fi
SSH_PORT="${SYNC_PORT:-22}"
USER_HOST="$SYNC_USER@$SYNC_HOST"
REMOTE_DIR="${SYNC_DIR:-~/chinese-mgt-signal-imaging}"
RSYNC_SSH="ssh -p $SSH_PORT"

case "${1:-}" in
  push)
    rsync -avz --delete -e "$RSYNC_SSH" \
      --exclude='__pycache__' --exclude='.DS_Store' \
      src docs scripts tests config.py run.py requirements.txt README.md .gitignore \
      "$USER_HOST:$REMOTE_DIR/"
    echo "[push] 代码已同步 -> $USER_HOST:$REMOTE_DIR"
    ;;
  push-data)
    # 注：Mac 自带 openrsync 对大文件带 --progress 会崩溃，故不用 --progress
    rsync -avz -e "$RSYNC_SSH" data/raw/ "$USER_HOST:$REMOTE_DIR/data/raw/"
    echo "[push-data] 原始数据已同步（约 90MB，仅首次需要）"
    ;;
  pull)
    rsync -avz -e "$RSYNC_SSH" "$USER_HOST:$REMOTE_DIR/logs/" logs/
    rsync -avz --exclude='checkpoints' -e "$RSYNC_SSH" \
      "$USER_HOST:$REMOTE_DIR/experiments/" experiments/
    echo "[pull] 日志与实验结果已拉回；请 git add logs experiments 并提交"
    ;;
  pull-signals)
    rsync -avz -e "$RSYNC_SSH" \
      "$USER_HOST:$REMOTE_DIR/data/processed/signals/" data/processed/signals/
    echo "[pull-signals] 信号缓存已备份到本地（D3 备份要求）"
    ;;
  pull-ckpt)
    rsync -avz -e "$RSYNC_SSH" \
      "$USER_HOST:$REMOTE_DIR/experiments/checkpoints/" experiments/checkpoints/
    echo "[pull-ckpt] 模型权重已拉回"
    ;;
  *)
    echo "用法: $0 {push|push-data|pull|pull-signals|pull-ckpt}"
    exit 1
    ;;
esac
