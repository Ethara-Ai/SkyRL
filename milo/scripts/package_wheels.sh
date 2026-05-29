#!/usr/bin/env bash
# Phase 21.1 — build distributable wheels for the milo package.
set -euo pipefail

cd "$(dirname "$0")/../.."   # cd to SkyRL repo root

mkdir -p dist
uv build milo/ --out-dir dist/
echo ""
echo "Wheels:"
ls -la dist/
