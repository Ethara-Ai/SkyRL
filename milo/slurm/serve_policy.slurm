#!/bin/bash
#SBATCH --job-name=milo-policy-serve
#SBATCH --output=logs/milo-policy-%j.out
#SBATCH --error=logs/milo-policy-%j.err
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:${MILO_TP_SIZE:-4}
#SBATCH --requeue
set -euo pipefail

# Phase 13 — long-running vLLM policy server (full-weight reload via READY sentinel).
# v0.7: NO companion reference server — reference logprobs come from the
# offline S3 cache (`milo/algos/reference_cache.py`).

: "${MILO_MODEL_PATH:?MILO_MODEL_PATH is required}"
: "${MILO_TP_SIZE:=4}"
: "${MILO_PORT:=8000}"
: "${MILO_MAX_MODEL_LEN:=65536}"
: "${MILO_SCRATCH_DIR:?MILO_SCRATCH_DIR (for hot-reload sentinels) is required}"

cd "${MILO_REPO_ROOT:-$(pwd)}"

# Start the vLLM server.
uv run --isolated --extra fsdp --extra miniswe python -m vllm.entrypoints.openai.api_server \
    --model "$MILO_MODEL_PATH" \
    --tensor-parallel-size "$MILO_TP_SIZE" \
    --max-model-len "$MILO_MAX_MODEL_LEN" \
    --port "$MILO_PORT" \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.92 \
    --dtype bfloat16 &
VLLM_PID=$!

# Hot-reload watcher (Phase 13).
uv run --isolated --extra dev python -m milo.serving.hot_reload \
    --scratch-dir "$MILO_SCRATCH_DIR" \
    --vllm-port "$MILO_PORT" \
    --vllm-pid "$VLLM_PID" &
WATCHER_PID=$!

# Background syncer for trace logs (Phase 5.3 S3 sync).
if [[ -n "${MILO_LOG_BUCKET:-}" ]]; then
  uv run --isolated --extra dev python -m milo.logging.s3_syncer \
      --log-root "${MILO_LOG_ROOT:-$MILO_SCRATCH_DIR/logs}" \
      --run-id "${MILO_RUN_ID:-$(date +%Y%m%dT%H%M%S)}" \
      --bucket "$MILO_LOG_BUCKET" &
fi

trap 'kill $VLLM_PID $WATCHER_PID 2>/dev/null || true' EXIT
wait $VLLM_PID
