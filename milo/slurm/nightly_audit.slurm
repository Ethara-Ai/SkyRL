#!/bin/bash
#SBATCH --job-name=milo-nightly-audit
#SBATCH --output=logs/milo-nightly-%j.out
#SBATCH --error=logs/milo-nightly-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --requeue
set -euo pipefail

# Phase 17.5 — nightly invariant audit.
# Replays 20 random training rollouts via `milo replay` and asserts
# reward decomposition matches stored. Catches gym/trainer schema drift.

: "${MILO_TRACES_ROOT:?MILO_TRACES_ROOT is required}"
: "${MILO_N_SAMPLE:=20}"
: "${MILO_AUDIT_OUTPUT:=logs/audit-$(date +%Y%m%d).json}"

cd "${MILO_REPO_ROOT:-$(pwd)}"

echo "[$(date)] Running nightly audit on $MILO_TRACES_ROOT (n=$MILO_N_SAMPLE)"
uv run --isolated --extra dev python -m milo.observability.nightly_audit_cli \
    --traces-root "$MILO_TRACES_ROOT" \
    --n-sample "$MILO_N_SAMPLE" \
    --output "$MILO_AUDIT_OUTPUT"

# Exit non-zero if audit failed (alarms wired downstream).
PASSED=$(python3 -c "import json; print(json.load(open('$MILO_AUDIT_OUTPUT'))['passed'])")
if [[ "$PASSED" != "True" ]]; then
  echo "[$(date)] Audit FAILED — paging on-call"
  exit 1
fi
echo "[$(date)] Audit passed"
