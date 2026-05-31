#!/bin/bash
# ============================================================
# GTPO Training on Harbor Tasks
#
# Paper: https://arxiv.org/abs/2511.14846
# Algorithm: Group Turn Policy Optimization (GTPO)
#
# Key differences from GRPO (run_codecontest.sh):
#   - adv_estimator=gtpo (turn-level discounted returns)
#   - gamma=0.9 (temporal discounting, paper optimal)
#   - gtpo_turn_rewards=true (per-turn reward assignment)
#   - gtpo_format_penalty=-0.1 (penalize bad tool calls)
# ============================================================

set -euo pipefail

uv run --isolated --extra fsdp \
    -m examples.train_integrations.harbor.entrypoints.main_harbor_gtpo \
    \
    trainer.strategy=fsdp \
    trainer.policy.model.path=Qwen/Qwen3-8B \
    trainer.algorithm.adv_estimator=gtpo \
    trainer.algorithm.gamma=0.9 \
    trainer.algorithm.max_seq_len=32768 \
    trainer.algorithm.loss_reduction=token_mean \
    trainer.algorithm.kl_loss_coef=0.01 \
    trainer.algorithm.eps_clip_low=0.2 \
    trainer.algorithm.eps_clip_high=0.28 \
    trainer.algorithm.grpo_norm_by_std=true \
    trainer.train_batch_size=32 \
    trainer.n_samples_per_prompt=8 \
    trainer.ppo_epochs=1 \
    trainer.eval_interval=50 \
    trainer.save_interval=100 \
    \
    trainer.policy.optimizer.type=adamw \
    trainer.policy.optimizer.lr=2e-5 \
    trainer.policy.optimizer.weight_decay=0.01 \
    trainer.policy.optimizer.betas="[0.9, 0.999]" \
    trainer.policy.max_grad_norm=1.0 \
    \
    trainer.policy.use_lora=true \
    trainer.policy.lora.rank=16 \
    trainer.policy.lora.alpha=32 \
    trainer.policy.lora.dropout=0.05 \
    trainer.policy.lora.target_modules="[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]" \
    \
    generator.step_wise_trajectories=true \
    generator.merge_stepwise_output=true \
    generator.apply_overlong_filtering=true \
    generator.gtpo_turn_rewards=true \
    generator.gtpo_format_penalty=-0.1 \
    generator.inference_engine.n_engines=2 \
    generator.inference_engine.engine_kwargs.tensor_parallel_size=1 \
    generator.rate_limit.max_tps=5 \
    generator.rate_limit.max_concurrent=512 \
    \
    data.train_data=dataset.json \
    data.chat_template=qwen3_acc_thinking.jinja2 \
    \
    "$@"
