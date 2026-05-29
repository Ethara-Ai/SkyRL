#!/usr/bin/env bash
# Phase 21.3 — cross-account ECR image copy.
#
# Source:  Ethara ECR in ap-south-1
# Target:  AGIF ECR (account ID + region provided at handoff)
#
# Required env:
#   ETHARA_ACCOUNT      Ethara AWS account id
#   ETHARA_REGION       default ap-south-1
#   AGIF_ACCOUNT        AGIF AWS account id
#   AGIF_REGION         AGIF target region
#   ETHARA_REPO_PREFIX  default rfp-coding-q1

set -euo pipefail

: "${ETHARA_ACCOUNT:?ETHARA_ACCOUNT is required}"
: "${AGIF_ACCOUNT:?AGIF_ACCOUNT is required}"
: "${AGIF_REGION:?AGIF_REGION is required}"
: "${ETHARA_REGION:=ap-south-1}"
: "${ETHARA_REPO_PREFIX:=rfp-coding-q1}"

ETHARA_HOST="${ETHARA_ACCOUNT}.dkr.ecr.${ETHARA_REGION}.amazonaws.com"
AGIF_HOST="${AGIF_ACCOUNT}.dkr.ecr.${AGIF_REGION}.amazonaws.com"

echo "Logging in to both registries"
aws ecr get-login-password --region "$ETHARA_REGION" | \
    docker login --username AWS --password-stdin "$ETHARA_HOST"
aws ecr get-login-password --region "$AGIF_REGION" | \
    docker login --username AWS --password-stdin "$AGIF_HOST"

echo "Enumerating repositories under $ETHARA_REPO_PREFIX/"
REPOS=$(aws ecr describe-repositories --region "$ETHARA_REGION" \
    --query "repositories[?starts_with(repositoryName, '${ETHARA_REPO_PREFIX}/')].repositoryName" \
    --output text)

for REPO in $REPOS; do
  echo ""
  echo "==> Transferring $REPO"
  # Ensure target repo exists.
  aws ecr create-repository --region "$AGIF_REGION" --repository-name "$REPO" 2>/dev/null || true

  TAGS=$(aws ecr list-images --region "$ETHARA_REGION" --repository-name "$REPO" \
      --query 'imageIds[].imageTag' --output text)

  for TAG in $TAGS; do
    [[ "$TAG" == "None" ]] && continue
    SRC="${ETHARA_HOST}/${REPO}:${TAG}"
    DST="${AGIF_HOST}/${REPO}:${TAG}"
    echo "  ${TAG}"
    docker pull "$SRC"
    docker tag "$SRC" "$DST"
    docker push "$DST"
  done
done

echo ""
echo "Done. Image SHA stability preserved (no rebuilds)."
