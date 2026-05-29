"""LHT task-generation pipeline — Phase 11.

This package owns the offline pipeline that produces calibrated LHT instances
from raw on-disk milo-bench jsonl. Phase 11.8 (verifier construction) is the
subroutine that promotes Cohort B instances (empty F2P/P2P but with
fix_patch+test_patch) into Cohort A by synthesizing the F2P/P2P test lists
from a sandboxed run of the patches.

Other subroutines (rubric authoring in 11.10, golden-trace recording in 11.11)
are SME-driven workflows; only their *contracts* live in this package.
"""
