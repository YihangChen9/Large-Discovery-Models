#!/bin/bash
# Qwen2.5-7B-Instruct (dense) — the model we take END-TO-END first.
#
# History: this launcher was first the dense CONTROL for the 9B nan blocker. The
# control ran (finite grad_norm on a dense 7B where 9B produced none), so its
# question is answered and we now use the same launcher to get the FIRST complete
# result on this pipeline: a run that actually trains, checkpoints, and feeds the
# offline G12C/G12D hypervolume eval — a working baseline while 9B's nan is chased.
#
# ── the one thing that makes "complete" mean something ────────────────────────
# In the control every arm moved weights by exactly one bf16 ULP (1.526e-05 =
# 2^-16): at lr=1e-6 with bf16 optimizer momenta the Adam update (~lr in size)
# lands on the bf16 grid, so the optimizer APPLIES a step but the model does not
# LEARN. A "complete" run under that config would just reproduce the SFT/base
# hypervolume. So this launcher defaults to:
#   - fp32 optimizer momenta (MOMENTA_DTYPE=fp32) so the update is representable;
#   - an overridable LR (default bumped off 1e-6).
# BEFORE the full run: do a short pre-check (a few steps) and confirm the per-
# tensor weight delta is >> 1 ULP. If it still quantises, raise LR further.
# fp32 momenta costs optimizer memory (~10->16 bytes/param); if it OOMs on the
# 96GB card, set MOMENTA_DTYPE=bf16 and lean on LR instead, or raise TP.
#
# ── set for your cluster ──────────────────────────────────────────────────────
export REPO_ROOT=${REPO_ROOT:?set REPO_ROOT, e.g. /path/to/LDM}
export SLIME_ROOT=${SLIME_ROOT:-$REPO_ROOT/rl/slime}
export MEGATRON_ROOT=${MEGATRON_ROOT:?set MEGATRON_ROOT}
export CONDA_PREFIX=${CONDA_PREFIX:?set CONDA_PREFIX (torch/TE/slime/sglang env)}
MODEL_HF=${MODEL_HF:?set MODEL_HF (Qwen2.5-7B-Instruct dir)}
MODEL_REF=${MODEL_REF:?set MODEL_REF (its Megatron torch_dist dir; convert with convert_dense7b.sh)}
EPISODES=${EPISODES:-$REPO_ROOT/rl_episodes_sm_R2.jsonl}
SAVE=${SAVE:-$REPO_ROOT/rl/qwen2.5-7B_rl_densecontrol}
WANDB_PROJECT=${WANDB_PROJECT:-ldm-sm-rl}
WANDB_RUN=${WANDB_RUN:-$(basename "$SAVE")}
# ──────────────────────────────────────────────────────────────────────────────
set -eux
CONFIG=${CONFIG:-$REPO_ROOT/rl/slime_launch/config_real.json}

mkdir -p /root/cudart_block 2>/dev/null || true
touch /root/cudart_block/libcudart.so.13 2>/dev/null || true
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=/root/cudart_block:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$MEGATRON_ROOT:$REPO_ROOT/rl:$REPO_ROOT:$PYTHONPATH
export CUDA_HOME=$CONDA_PREFIX
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export CUDA_DEVICE_MAX_CONNECTIONS=1

jq_get() { python3 -c "import json;print(json.load(open('$CONFIG'))['training']['$1'])"; }
NUM_ROLLOUT=$(jq_get num_rollout)
ROLLOUT_BATCH=$(jq_get rollout_batch_size)
N_SAMPLES=${N_SAMPLES:-$(jq_get n_samples_per_prompt)}
UPDATES_PER_ROLLOUT=${UPDATES_PER_ROLLOUT:-1}
GLOBAL_BATCH=$(( ROLLOUT_BATCH * N_SAMPLES / UPDATES_PER_ROLLOUT ))
RESP_LEN=$(jq_get rollout_max_response_len)
MAX_TOKENS=$(jq_get max_tokens_per_gpu)
TEMPERATURE=$(jq_get rollout_temperature)
# config lr is 1e-6 -> 1-ULP updates (see header). Overridable; default bumped.
LR=${LR:-1e-5}
# fp32 momenta so the Adam update is not quantised to the bf16 grid. Override to
# bf16 only if the fp32 optimizer state OOMs the card.
MOMENTA_DTYPE=${MOMENTA_DTYPE:-fp32}
SAVE_INTERVAL=$(jq_get save_interval)

cd "$SLIME_ROOT"

# Dense Qwen2.5-7B architecture (standard transformer; no hybrid spec, no MTP).
source "$SLIME_ROOT/scripts/models/qwen2.5-7B.sh"   # sets MODEL_ARGS=(...)

CKPT_ARGS=(--hf-checkpoint "$MODEL_HF" --ref-load "$MODEL_REF" --save "$SAVE" --save-interval "$SAVE_INTERVAL")
ROLLOUT_ARGS=(
   --prompt-data "$EPISODES" --input-key prompt --label-key label
   --num-rollout "$NUM_ROLLOUT" --rollout-batch-size "$ROLLOUT_BATCH" --n-samples-per-prompt "$N_SAMPLES"
   --rollout-max-response-len "$RESP_LEN" --rollout-temperature "$TEMPERATURE"
   --global-batch-size "$GLOBAL_BATCH" --balance-data
)
# Single node, 4 GPUs: TP=2 actor + 2 sglang. Master params fp32; momenta dtype
# is MOMENTA_DTYPE (default fp32) so the update is not rounded to the bf16 grid.
# Set MOMENTA_DTYPE=bf16 to fall back to the memory-lean control config.
PERF_ARGS=(
   --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 --context-parallel-size 1
   --use-distributed-optimizer
   --use-precision-aware-optimizer --exp-avg-dtype "$MOMENTA_DTYPE" --exp-avg-sq-dtype "$MOMENTA_DTYPE" --main-params-dtype fp32
   --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
   --use-dynamic-batch-size --max-tokens-per-gpu "$MAX_TOKENS"
)
APEX_ARGS=(--no-gradient-accumulation-fusion)
GRPO_ARGS=(--advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl --eps-clip 0.2 --eps-clip-high 0.28)
OPTIMIZER_ARGS=(--optimizer adam --lr "$LR" --lr-decay-style constant --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98 --clip-grad 1.0)
SGLANG_ARGS=(--rollout-num-gpus 2 --sglang-mem-fraction-static 0.7)
CUSTOM_ARGS=(--custom-generate-function-path ldm_rl.bridge.generate --custom-rm-path ldm_rl.bridge.reward_func)
WANDB_ARGS=()
[[ -n "${WANDB_KEY:-}" ]] && WANDB_ARGS=(--use-wandb --wandb-project "$WANDB_PROJECT" --wandb-key "$WANDB_KEY" --wandb-run-name "$WANDB_RUN")

echo "resolved: n_samples=$N_SAMPLES global_batch=$GLOBAL_BATCH lr=$LR momenta=$MOMENTA_DTYPE (dense Qwen2.5-7B end-to-end, TP=2)"
echo "PRE-CHECK before the full run: run a few steps, confirm per-tensor weight delta >> 1 bf16 ULP (1.526e-05); if not, raise LR."

ray stop --force 2>/dev/null || true
sleep 3
ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 --disable-usage-stats
RUNTIME_ENV_JSON="{\"env_vars\": {\"PYTHONPATH\": \"$MEGATRON_ROOT:$REPO_ROOT/rl:$REPO_ROOT\", \"LD_LIBRARY_PATH\": \"/root/cudart_block:$CONDA_PREFIX/lib\", \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"}}"
ray job submit --address="http://127.0.0.1:8265" --runtime-env-json="$RUNTIME_ENV_JSON" \
   -- python3 train.py \
   --actor-num-nodes 1 --actor-num-gpus-per-node 2 --rollout-num-gpus 2 \
   ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} ${PERF_ARGS[@]} ${GRPO_ARGS[@]} ${OPTIMIZER_ARGS[@]} ${APEX_ARGS[@]} ${SGLANG_ARGS[@]} ${CUSTOM_ARGS[@]} ${WANDB_ARGS[@]}
