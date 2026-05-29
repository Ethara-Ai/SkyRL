"""Canonical ECR image-name helper for milo-bench instances.

Mirrors the convention used by `benchmarks/multiswebench/build_images.py` in the
freya repo (the source-of-truth lineage for our Docker images). Keep this in
sync — if upstream changes the prefix or tag scheme, change it here too.

Reference:
    https://github.com/<freya>/multiswebench/blob/main/build_images.py
    (function `get_official_docker_image`)

The convention:
    {prefix}/{org}_m_{repo}:pr-{number}   (all lowercased)

The default prefix is the Ethara-internal ECR for the RFP-coding-q1 batch:
    426628337772.dkr.ecr.ap-south-1.amazonaws.com/rfp-coding-q1

Override via the `EVAL_DOCKER_IMAGE_PREFIX` env var (matches the upstream
multiswebench convention so users get familiar behavior).

The integrator-facing prefix is overridable, so AGIF or any other client can
re-tag and serve from their own registry without code changes.
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_ECR_PREFIX = "426628337772.dkr.ecr.ap-south-1.amazonaws.com/rfp-coding-q1"


def get_image_prefix() -> str:
    """Returns the configured image prefix. Override with EVAL_DOCKER_IMAGE_PREFIX."""
    return os.environ.get("EVAL_DOCKER_IMAGE_PREFIX", DEFAULT_ECR_PREFIX)


def get_image_name(instance: dict[str, Any], prefix: str | None = None) -> str:
    """Return the canonical docker image name for a milo-bench instance.

    Accepts an instance dict with at minimum:
        - org   (or extracted from repo)
        - repo  (may be "org/name" or just "name")
        - number (PR number; falls back to parsing from instance_id)

    Returns e.g.:
        '426628337772.dkr.ecr.ap-south-1.amazonaws.com/rfp-coding-q1/locustio_m_locust:pr-1541'
    """
    if prefix is None:
        prefix = get_image_prefix()

    repo = instance.get("repo", "")
    if "/" in repo:
        org, repo_name = repo.split("/", 1)
    else:
        org = instance.get("org", repo)
        repo_name = repo

    number = instance.get("number")
    if number is None:
        iid = instance.get("instance_id", "")
        if "-" in iid:
            number = iid.rsplit("-", 1)[-1]

    tag = f"pr-{number}" if number else "base"
    return f"{prefix}/{org}_m_{repo_name}:{tag}".lower()


def parse_instance_id(image_name: str) -> str:
    """Inverse of get_image_name: extract the milo instance_id from an image name.

    Example:
        '<prefix>/locustio_m_locust:pr-1541' -> 'locustio__locust-1541'
    """
    name_tag = image_name.split("/")[-1]
    if ":" not in name_tag:
        return name_tag.replace("_m_", "__")
    name, tag = name_tag.split(":", 1)
    name = name.replace("_m_", "__")
    if tag.startswith("pr-"):
        return f"{name}-{tag[3:]}"
    return f"{name}-{tag}"
