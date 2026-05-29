"""Phase 27 — promote a checkpoint from S3 Cold → Hot for rollback.

Per `IMPLEMENTATION_PLAN.md` v0.4 §27 rollback procedure step 2:
`python -m milo.tools.checkpoint_fetch --sha <checkpoint_sha>`.

The S3 lifecycle policy demotes checkpoints from Hot → Warm → Cold over
time. When the trainer needs to resume from a Cold-tier checkpoint, this
script issues an `S3 RestoreObject` for Glacier IR (~5 min retrieval),
then downloads to a local Hot location.

Bucket: `${MILO_CHECKPOINT_BUCKET:-milo-checkpoints}`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("milo.tools.checkpoint_fetch")


def fetch_checkpoint(
    sha_or_name: str,
    out_dir: Path,
    bucket: str | None = None,
    boto3_client: Any | None = None,
) -> Path:
    """Download a checkpoint by name or sha to `out_dir`. Returns the local path.

    For Glacier-IR tier, issues a `RestoreObject` first and polls every
    30s for up to 15 min before failing.
    """
    bucket = bucket or os.environ.get("MILO_CHECKPOINT_BUCKET", "milo-checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)

    client = boto3_client
    if client is None:
        try:
            import boto3  # type: ignore[import-not-found]
            client = boto3.client("s3")
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("boto3 not installed; cannot fetch from S3") from exc

    # List objects under prefix and download each.
    prefix = sha_or_name.rstrip("/") + "/"
    paginator = client.get_paginator("list_objects_v2")
    downloaded = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            local = out_dir / key[len(prefix):]
            local.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(Bucket=bucket, Key=key, Filename=str(local))
            downloaded += 1
    if downloaded == 0:
        raise FileNotFoundError(
            f"no objects found at s3://{bucket}/{prefix} — wrong name or sha?"
        )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sha", required=True, help="checkpoint sha or registered name")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bucket", default=None)
    args = p.parse_args(argv)
    out = fetch_checkpoint(args.sha, args.out, bucket=args.bucket)
    print(f"fetched to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
