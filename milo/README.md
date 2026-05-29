# milo/ — Ethara Milo-bench RL extensions on SkyRL

All Ethara-authored code lives here. Upstream SkyRL packages
(`skyrl/`, `skyrl-agent/`, `skyrl-gym/`, `skyrl-train/`, `skyrl-tx/`,
`docker/`, `examples/`, `docs/`) are untouched so weekly upstream merges
stay conflict-free.

**Source-of-truth docs** (in `freya/milo-bench/spec/`):
- `RL_GYM_SPEC.md` v0.7 — what we're building
- `IMPLEMENTATION_PLAN.md` v0.4 — how we're building it
- `OSS_FOUNDATION_ANALYSIS.md` v0.1 — why we forked SkyRL
- `BRUTAL_REVIEW_MILO_ONLY.md` — the review that drove plan v0.3 → v0.4

## Implementation status (Phase 0 → 21 scaffold complete)

| Phase | Status | Notes |
|---|---|---|
| 0   Repo bootstrap | ✅ done | branch `milo/phase-0-bootstrap` |
| 0.5 End-to-end spike | ✅ scaffolded | needs Docker image + GPU to execute; see `milo/spike/README.md` |
| 0.6 Dataset audit | ✅ done, verified | 135 Cohort A + 215 Cohort B = 350 against freya/milo-bench/dataset |
| 1   LHT adapter | ✅ scaffolded | `BaseTextEnv` + `SkyRLGymGenerator` subclass per v0.4 §1 |
| 2   Verifier (3-Docker-run + invariant guards) | ✅ scaffolded | `milo/verifier/three_run.py` + 8 language runners |
| 3   Rubric judge service | ✅ scaffolded | Bedrock + Anthropic backends; LMDB cache; SHA-pinned prompt |
| 4   Reward aggregator | ✅ done | composite + pure_rlvr presets; 8-invariant integration |
| 5   Logging — OpenHands overlay | ✅ done | trace_format + recorder + s3_syncer + converter |
| 6   Anti-hacking invariants I-1..I-8 | ✅ done | v0.7-hardened I-2; new I-8; full test pack |
| 7   Policy adapters (5) | ✅ done | Bedrock direct + 4 via litellm + auto-routing parsers |
| 8   Calibration runner | ✅ scaffolded | pass@8 across two frontier models; env-driven IDs |
| 9   Replay tool + smoke pack | ✅ scaffolded | 5-task manifest; record-golden mode |
| 10  Gym CI | ✅ done | `.github/workflows/milo-ci.yml` |
| 11  LHT pipeline | ⚙️ partial | 11.8 verifier construction done; 11.10/11.11 SME-driven |
| 12  SFT pipeline | ✅ scaffolded | builder + sft_v1 config + validator |
| 13  vLLM policy serving | ✅ scaffolded | full-weight reload via READY sentinel (v0.7) |
| 14  RL trainer + KL k3 + offline ref cache | ✅ done | β_KL=0.01 anneal 0.005, no live ref server |
| 15  Customization Protocols | ✅ done | 5 Protocols + 4 cookbook entries |
| 16  Eval harness + bootstrap CI | ✅ done | 10K-resample paired bootstrap; release gate |
| 17  Observability — alarms + dashboards | ✅ done | 3 alarms + 5 W&B dashboards + nightly audit |
| 18  Ops — Slurm + K8s + Dockerfiles | ✅ done | 4 slurm scripts + 2 K8s manifests + 2 Dockerfiles |
| 19  Reference SFT + RL run | ⏸ pending | scripts ready; waits on H100 reservation |
| 20  Ablations (15 runs) | ⏸ pending | scripts ready; waits on Phase 19 winner |
| 21  Packaging & handoff | ✅ scaffolded | INTEGRATOR_GUIDE + RFP mapping + 3 transfer scripts |
| 22  Upstream merge cadence | ✅ documented | weekly per CTO direction (plan §22) |

**Numbers (this commit):**
- 138 Python modules (compile clean)
- 32 test files, **188/188 unit tests passing** (CPU-only)
- 21,844 lines across `milo/` + Dockerfiles + CI workflow

## Directory layout

| Path | Phase | Contents |
|---|---|---|
| `milo/spike/` | 0.5 | End-to-end spike scaffold |
| `milo/audit/` | 0.6 | Dataset audit + contamination check |
| `milo/lht_adapter/` | 1 | LHT env + `SkyRLGymGenerator` subclass + docker runtime |
| `milo/verifier/` | 2 | 3-Docker-run verifier + 8 per-language test runners |
| `milo/judge/` | 3 | Rubric judge (Bedrock + Anthropic backends + LMDB cache) |
| `milo/reward/` | 4 | Composite reward aggregator + TIR + decomposition |
| `milo/logging/` | 5 | OpenHands trace overlay + S3 syncer + legacy converter |
| `milo/invariants/` | 6 | I-1..I-8 checks + runner orchestrator |
| `milo/adapters/` | 7 | 5 policy adapters + 3 tool-call parsers + registry |
| `milo/calibration/` | 8 | Pass@8 calibration runner + tier assignment |
| `milo/replay/` | 9 | Replay CLI + smoke pack manifest |
| `milo/lht_pipeline/` | 11 | Verifier construction subroutine (11.8) |
| `milo/sft/` | 12 | SFT dataset builder + sft_v1 config + validator |
| `milo/serving/` | 13 | vLLM config defaults + hot-reload watcher |
| `milo/algos/` | 14 | GRPO loss wrapper + Schulman k3 + offline ref cache |
| `milo/customization/` | 15 | 5 swap-point Protocols + cookbook |
| `milo/eval/` | 16 | pass@k evaluator + paired bootstrap CI |
| `milo/observability/` | 17 | 3 alarms + 5 W&B dashboards + nightly audit |
| `milo/tools/` | 18 | Checkpoint verify + registry + reproducibility manifest |
| `milo/slurm/` | 18 | 4 Slurm scripts (sft / train / serve / nightly-audit) |
| `milo/k8s/` | 18 | K8s alternatives (train.yaml, serve.yaml) |
| `milo/docs/` | 21 | Integrator guide + cookbook + RFP §5 mapping + smoke test |
| `milo/scripts/` | 21 | ECR transfer + wheel packaging + checkpoint upload |
| `milo/data/` | various | Local cache for instance jsonl, splits, rubrics, golden traces |

## Quick start (today, on a developer laptop)

```bash
# 0. install
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra dev

# 1. run the dataset audit (no GPU, ~1 min)
uv run python -m milo.audit.audit_dataset \
    --milo-dataset-dir /path/to/freya/milo-bench/dataset \
    --output-dir milo/data/audit_v1

# 2. run the unit tests
uv run pytest milo/ -v --ignore=milo/lht_adapter/tests/test_env.py \
                       --ignore=milo/verifier/tests/test_three_run.py \
                       --ignore=milo/serving/tests/test_hot_reload.py
```

## With GPU (Phase 0.5 spike onwards)

```bash
uv sync --extra fsdp --extra miniswe --extra dev

export EVAL_DOCKER_IMAGE_PREFIX=<your-ECR-prefix>
export MILO_JUDGE_MODEL=claude-opus-4-6    # today's GA; env-overridable
export WANDB_API_KEY=<your-key>

# Phase 0.5 spike on the locustio__locust-1541 task (Cohort A, F2P=1, P2P=309)
bash milo/spike/run_spike.sh
```

## Configuration model

All operational knobs are env vars — adaptable for any client infra:

| Env var | Default | Used by |
|---|---|---|
| `EVAL_DOCKER_IMAGE_PREFIX` | `426628337772.dkr.ecr.ap-south-1.amazonaws.com/rfp-coding-q1` | `milo/lht_adapter/image_naming.py` |
| `MILO_JUDGE_MODEL` | `claude-opus-4-6` | `milo/judge/` |
| `MILO_CALIBRATION_MODEL_2` | `gemini-2.5-pro` | `milo/calibration/` |
| `MILO_POLICY_MODEL` | `anthropic.claude-opus-4-6` | `milo/adapters/bedrock.py` |
| `MILO_BEDROCK_REGION` | `ap-south-1` | `milo/adapters/bedrock.py` |
| `MILO_LOG_BUCKET` | (unset → local-only) | `milo/logging/s3_syncer.py` |
| `MILO_CHECKPOINT_BUCKET` | `milo-checkpoints` | `milo/tools/checkpoint_fetch.py` |
| `MILO_REGISTRY_PATH` | `milo/data/registry.json` | `milo/tools/registry.py` |
| `MILO_BETA_KL_INITIAL` | `0.01` (v0.4 default) | `milo/algos/grpo_wrapper.py` |
| `MILO_BETA_KL_FINAL` | `0.005` | `milo/algos/grpo_wrapper.py` |
| `MILO_BETA_KL_ANNEAL_STEPS` | `4800` | `milo/algos/grpo_wrapper.py` |

## What's intentionally NOT done

Spec + plan explicitly defer the following:
- **Phase 19 reference run** — waits on H100 reservation, $90K compute.
- **Phase 20 ablations** — waits on Phase 19 LoRA-vs-FT bake-off winner.
- **Phase 11.10/11.11 SME work** — rubric authoring + golden traces are SME-driven; 12 SMEs × 5 weeks per plan §11.0.
- **Phase 11.9 contamination culling** — proceeds after audit + SME signoff.

See `milo/BLOCKERS.md` for the gating decisions still open.

## Pointers for the AGIF reviewer

| Question | Read |
|---|---|
| "How do I run the system end-to-end?" | `milo/docs/INTEGRATOR_GUIDE.md` |
| "How do I reproduce the reference checkpoint?" | `milo/docs/REPRODUCING_REFERENCE_CHECKPOINT.md` |
| "How does this map to the RFP §5 deliverables?" | `milo/docs/RFP_DELIVERABLE_MAPPING.md` |
| "How do I customize each swap point?" | `milo/customization/cookbook/` |
| "What does the AGIF day-1 smoke test look like?" | `milo/docs/HANDOFF_SMOKE_TEST.md` |
| "How was QC done?" | `milo/docs/QC_PROCESS.md` |
| "What's the action-level tool API?" | `milo/docs/TOOL_SPEC.md` |
