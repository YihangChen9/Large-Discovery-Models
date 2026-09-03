# Small-molecule RL — run matrix

Real-mode GRPO on the small-molecule LDM loop. **Train on KRAS G12D** (ready
evaluator: `best_g12d_model.joblib` + 8UN5), **evaluate on G12C and G12D**.
**One episode file per run** — each run gets its **own** `gp_history_file` +
`output_dir` (sharing a GP across runs couples their rewards; see PR #2 / nb 06).
`config_real.json` default reward is now **hypervolume (ΔHV)**; the acquisition
variants are kept as an ablation. Two axes, sharing R2 as the pivot:

| Run | Model (`MODEL_HF`) | Reward | Episodes file |
|-----|--------------------|--------|---------------|
| **R1** | Qwen3.5-9B base | acquisition-**max** | `rl_episodes_sm_R1.jsonl` |
| **R2** | SFT (no-GP) | acquisition-**max** | `rl_episodes_sm_R2.jsonl` |
| **R3** | SFT (no-GP) | **hypervolume** (ΔHV, real outcome) | `rl_episodes_sm_R3.jsonl` |
| **R4** | SFT (no-GP) | acquisition-**mean** | `rl_episodes_sm_R4.jsonl` |

- R1 vs R2 isolates the **model** (base vs SFT-init).
- R2 vs R3 vs R4 isolates the **reward** (decision-time max / real outcome / decision-time mean).
- Seeds: change the **environment** seed by regenerating with
  `python -m ldm_rl.episodes --seed-offset N` (it is an `episodes.py` flag, not a
  slime one); ≥3–5 seeds per run for error bars.
- With acquisition, watch `zero_std/count_0.0`: if GRPO groups collapse to equal
  rewards (EHVI decay / collisions), the gradient is zero — prefer ΔHV (R3).

## 0. One-time prep
```bash
cd /mnt/data0/ys/LDM/rl/slime_launch
# convert both models HF -> Megatron torch_dist
MODEL_HF=/mnt/data0/hf_models/models/Qwen3.5-9B                 SAVE=/mnt/data0/ys/LDM/rl/qwen3.5-9B_torch_dist      bash convert_9b.sh
MODEL_HF=/mnt/data0/hf_models/models/LDM-CoT-SFT               SAVE=/mnt/data0/ys/LDM/rl/qwen3.5-9B-sft_torch_dist  bash convert_9b.sh
# generate one episode file per run (R1-R4), each with its own GP + output dir
bash gen_episodes_runs.sh
# warm the BASE GP once (rollout-only, docks warmup.num_samples molecules)
bash run_warmup_real_slime.sh    # GP history is model-agnostic
# seed each run's GP from the warm base (so runs are isolated but warm-started)
BASE_GP=/mnt/data0/ys/LDM/rl/sm_rl_gp_history.jsonl
for r in R1 R2 R3 R4; do cp "$BASE_GP" /mnt/data0/ys/LDM/rl/gp_history/$r.jsonl; done
```

## 1. Launch runs (4 GPUs each, TP=2)
```bash
BASE=/mnt/data0/hf_models/models/Qwen3.5-9B ;      BASE_REF=/mnt/data0/ys/LDM/rl/qwen3.5-9B_torch_dist
SFT=/mnt/data0/hf_models/models/LDM-CoT-SFT ;      SFT_REF=/mnt/data0/ys/LDM/rl/qwen3.5-9B-sft_torch_dist
R=/mnt/data0/ys/LDM/rl ; export WANDB_KEY=<your-wandb-key>   # optional

# R1 base, acq-max
MODEL_HF=$BASE MODEL_REF=$BASE_REF EPISODES=$R/../rl_episodes_sm_R1.jsonl  SAVE=$R/qwen3.5-9B_rl_R1_base_acqmax  WANDB_RUN=R1_base_acqmax  bash run_train_real_9b.sh
# R2 sft, acq-max
MODEL_HF=$SFT  MODEL_REF=$SFT_REF  EPISODES=$R/../rl_episodes_sm_R2.jsonl  SAVE=$R/qwen3.5-9B_rl_R2_sft_acqmax   WANDB_RUN=R2_sft_acqmax   bash run_train_real_9b.sh
# R3 sft, ΔHV (real outcome)
MODEL_HF=$SFT  MODEL_REF=$SFT_REF  EPISODES=$R/../rl_episodes_sm_R3.jsonl  SAVE=$R/qwen3.5-9B_rl_R3_sft_hv       WANDB_RUN=R3_sft_hv       bash run_train_real_9b.sh
# R4 sft, acq-mean
MODEL_HF=$SFT  MODEL_REF=$SFT_REF  EPISODES=$R/../rl_episodes_sm_R4.jsonl  SAVE=$R/qwen3.5-9B_rl_R4_sft_acqmean  WANDB_RUN=R4_sft_acqmean  bash run_train_real_9b.sh
```

## 2. Reward semantics
- `reward: acquisition`, `acquisition_agg: max|mean` — per-round reward is the
  max (or mean) EHVI acquisition score of the evaluated candidate(s); episode
  reward is the sum over rounds. (See `rl/ldm_rl/env.py::_acquisition_reward`.)
- `reward: hypervolume` — per-round **Pareto-front hypervolume improvement (ΔHV)**
  from the real Vina + activity outcomes, summed over rounds (telescopes to the
  campaign's total HV gain). `reward_ref_point` (oriented space) fixes the ref,
  else a per-round nadir is used. `reward: improvement` (sum of per-objective
  gains) also available but gameable across objectives.

## 3. Notes / risks
- **Env**: use a matched torch + TE stack (see HANDOFF §2); a mismatched stack
  SIGSEGVs on GRPO backward.
- **First 9B run**: smoke with a **tiny real** episode set (count=1, iterations=2)
  to confirm the hybrid backward + memory before scaling up. Training is always
  real — do not use mock.
- **Memory**: 9B + TP=2/4 + sglang on 4 GPUs (96GB each on GH200). If OOM, raise
  TP or drop `max_tokens_per_gpu`; recompute-full is already on.
- **Docking throughput** is the real bottleneck for real reward; enable the
  docking cache and raise `vina_max_workers` in `config_real.json` real_kwargs.
- All 4 runs **train on G12D**; the G12C transfer number comes from the
  offline eval harness, not from training.
