#!/bin/bash
# One-shot: the ΔHV + n=4 config for PR #2's zero-variance check.
#
# What it does, end to end:
#   1. regenerate per-run episodes (reward=hypervolume via config, own GP/output per run);
#   2. seed each run's GP from the warm base;
#   3. launch the SFT + ΔHV run (R3) with n_samples_per_prompt=4.
# Then watch wandb `rollout/zero_std/count_0.0` — it should drop vs the n=2 / acquisition runs.
#
# ── set these for your cluster (Isambard paths differ from ours) ──────────────
export REPO_ROOT=${REPO_ROOT:?set REPO_ROOT to the checkout, e.g. /path/to/LDM}
export SLIME_ROOT=${SLIME_ROOT:-$REPO_ROOT/rl/slime}
export MEGATRON_ROOT=${MEGATRON_ROOT:?set MEGATRON_ROOT (Megatron-LM checkout)}
export CONDA_PREFIX=${CONDA_PREFIX:?set CONDA_PREFIX (env with torch/TE/slime/sglang)}
MODEL_HF=${MODEL_HF:?set MODEL_HF (SFT model dir, e.g. .../LDM-CoT-SFT)}
MODEL_REF=${MODEL_REF:?set MODEL_REF (its Megatron torch_dist dir)}
# WANDB_KEY optional: export WANDB_KEY=...
# ─────────────────────────────────────────────────────────────────────────────
set -eux
CONFIG=$REPO_ROOT/rl/slime_launch/config_real.json
BASE_GP=$(python3 -c "import json;print(json.load(open('$CONFIG'))['gp_history_file'])")

cd "$REPO_ROOT/rl/slime_launch"

# 1. per-run episodes (config already has reward=hypervolume, kernel=fp, n_samples=4)
bash gen_episodes_runs.sh

# 2. seed each run's GP from the warm base (run run_warmup_real_slime.sh first if $BASE_GP is missing)
test -s "$BASE_GP" || { echo "WARN: $BASE_GP missing — run run_warmup_real_slime.sh first"; exit 1; }
mkdir -p "$REPO_ROOT/rl/gp_history"
for r in R1 R2 R3 R4; do cp "$BASE_GP" "$REPO_ROOT/rl/gp_history/$r.jsonl"; done

# 3. launch R3 = SFT + ΔHV, n_samples=4  (override N_SAMPLES here without touching config)
N_SAMPLES=4 \
MODEL_HF="$MODEL_HF" MODEL_REF="$MODEL_REF" \
EPISODES="$REPO_ROOT/rl_episodes_sm_R3.jsonl" \
SAVE="$REPO_ROOT/rl/qwen3.5-9B_rl_R3_sft_hv_n4" \
WANDB_RUN=R3_sft_hv_n4 \
  bash run_train_real_9b.sh

echo "launched R3 (ΔHV, n=4). Watch wandb rollout/zero_std/count_0.0 and rollout/raw_reward."
