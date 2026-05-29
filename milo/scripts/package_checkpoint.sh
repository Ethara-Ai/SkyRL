#!/usr/bin/env bash
# Phase 21.2 — upload the reference checkpoint + manifest to S3.
#
# Required env:
#   MILO_CKPT_DIR              Local checkpoint dir
#   MILO_CKPT_NAME             Name in the registry, e.g. qwen2.5-coder-32b-milo-rl-v1
#   MILO_CHECKPOINT_BUCKET     S3 bucket (default milo-checkpoints)

set -euo pipefail

: "${MILO_CKPT_DIR:?MILO_CKPT_DIR is required}"
: "${MILO_CKPT_NAME:?MILO_CKPT_NAME is required}"
: "${MILO_CHECKPOINT_BUCKET:=milo-checkpoints}"

cd "$(dirname "$0")/../.."

# Verify before uploading.
uv run --isolated --extra dev python -m milo.tools.checkpoint_verify "$MILO_CKPT_DIR"

# Upload directory contents (model.safetensors, tokenizer.json, manifest.json, etc.)
aws s3 sync "$MILO_CKPT_DIR" "s3://${MILO_CHECKPOINT_BUCKET}/checkpoints/${MILO_CKPT_NAME}/" \
    --sse aws:kms \
    --exclude "*.tmp" \
    --exclude "events.out.tfevents.*"

# Register locally so the trainer / eval harness know about it.
uv run --isolated --extra dev python -m milo.tools.registry register \
    --name "$MILO_CKPT_NAME" \
    --path "s3://${MILO_CHECKPOINT_BUCKET}/checkpoints/${MILO_CKPT_NAME}/"

echo ""
echo "Done. Checkpoint uploaded to s3://${MILO_CHECKPOINT_BUCKET}/checkpoints/${MILO_CKPT_NAME}/"
