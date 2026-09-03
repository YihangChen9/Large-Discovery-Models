#!/bin/bash
# One-shot for the measured-best config: acquisition + n=4 (PR #2).
# Properly measured (using the std<1e-6 gate, not exact-equality), acquisition+n=4
# had 0 degenerate steps vs ΔHV's ~21%, so acquisition is the primary reward and
# n=4 the group size. Runs R2 = SFT + acquisition, single-node (no cross-node
# placement split), with the precision-aware optimizer so it fits on one node.
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

# 1. per-run episodes (config: reward=acquisition, kernel=fp, n_samples=4)
bash gen_episodes_runs.sh

# 2. seed each run's GP from the warm base (run run_warmup_real_slime.sh first if missing)
test -s "$BASE_GP" || { echo "WARN: $BASE_GP missing — run run_warmup_real_slime.sh first"; exit 1; }
mkdir -p "$REPO_ROOT/rl/gp_history"
for r in R1 R2 R3 R4; do cp "$BASE_GP" "$REPO_ROOT/rl/gp_history/$r.jsonl"; done

# 3. launch R2 = SFT + acquisition, n=4  (N_SAMPLES overrides config without editing it)
N_SAMPLES=4 \
MODEL_HF="$MODEL_HF" MODEL_REF="$MODEL_REF" \
EPISODES="$REPO_ROOT/rl_episodes_sm_R2.jsonl" \
SAVE="$REPO_ROOT/rl/qwen3.5-9B_rl_R2_sft_acq_n4" \
WANDB_RUN=R2_sft_acq_n4 \
  bash run_train_real_9b.sh

echo "launched R2 (acquisition, n=4). Watch the std<1e-6 gate, raw_reward, and train/grad_norm (nan => update skipped)."
