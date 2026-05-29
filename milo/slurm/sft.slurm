#!/bin/bash
#SBATCH --job-name=milo-sft
#SBATCH --output=logs/milo-sft-%j.out
#SBATCH --error=logs/milo-sft-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=${MILO_NNODES:-1}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:${MILO_GPUS_PER_NODE:-8}
#SBATCH --requeue                # plan §28.2 — recover from spot preemption
#SBATCH --open-mode=append
set -euo pipefail

# Phase 12 + 19.1 — SFT warmstart of Qwen2.5-Coder-32B on golden traces.
# Parameterised via env vars; defaults match the plan §19.1 reference run.
#
# Required:
#   MILO_SFT_CONFIG     — path to the run config (e.g. milo/sft/sft_v1.py)
#   MILO_DATA_DIR       — directory containing train.parquet / validation.parquet
#   MILO_CKPT_DIR       — where to write the SFT checkpoint
#
# Optional:
#   MILO_NNODES=1
#   MILO_GPUS_PER_NODE=8
#   MILO_BASE_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
#   MILO_LOGGER=wandb

: "${MILO_SFT_CONFIG:?MILO_SFT_CONFIG is required}"
: "${MILO_DATA_DIR:?MILO_DATA_DIR is required}"
: "${MILO_CKPT_DIR:?MILO_CKPT_DIR is required}"
: "${MILO_NNODES:=1}"
: "${MILO_GPUS_PER_NODE:=8}"
: "${MILO_BASE_MODEL:=Qwen/Qwen2.5-Coder-32B-Instruct}"
: "${MILO_LOGGER:=wandb}"

cd "${MILO_REPO_ROOT:-$(pwd)}"

echo "[$(date)] Starting Milo SFT (nodes=$MILO_NNODES, gpus/node=$MILO_GPUS_PER_NODE)"
echo "  config=$MILO_SFT_CONFIG"
echo "  data=$MILO_DATA_DIR"
echo "  ckpt=$MILO_CKPT_DIR"
echo "  model=$MILO_BASE_MODEL"

srun uv run --isolated --extra fsdp --extra miniswe -m milo.sft.train \
    --config "$MILO_SFT_CONFIG" \
    --train-data "$MILO_DATA_DIR/sft_train.jsonl" \
    --val-data "$MILO_DATA_DIR/sft_val.jsonl" \
    --base-model "$MILO_BASE_MODEL" \
    --output-dir "$MILO_CKPT_DIR" \
    --logger "$MILO_LOGGER"

echo "[$(date)] SFT done — verifying checkpoint"
uv run --isolated --extra dev python -m milo.tools.checkpoint_verify \
    "$MILO_CKPT_DIR" --require-transformers

echo "[$(date)] Registering as qwen2.5-coder-32b-milo-sft-v1"
uv run --isolated --extra dev python -m milo.tools.registry register \
    --name qwen2.5-coder-32b-milo-sft-v1 \
    --path "$MILO_CKPT_DIR"

echo "[$(date)] SFT slurm job complete"
