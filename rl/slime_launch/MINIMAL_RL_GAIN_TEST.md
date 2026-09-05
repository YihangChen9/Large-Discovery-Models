# The smallest comparison that could actually show an RL gain

Written 2026-09-05. **Nothing here is launched.** The prerequisite below is not satisfied,
so the design exists to be executed once it is, not before.

## Blocking prerequisite, stated first

**No 9B run has ever completed an optimizer step with a finite gradient** — 238 logged
gradient norms across the seven 9B runs that produced checkpoints, all nan or inf. Until
one 9B run logs finite gradients over a meaningful stretch, there is no model to compare
and no evaluation can be informative. That is the next decisive step, and it is a training
problem, not an evaluation problem.

The 1.5B work is a **pipeline pilot**. It should not be reported as progress on the main
objective and cannot answer it: the plan's question is about the 9B SFT checkpoint.

## The design, for when a 9B checkpoint exists

**Comparison:** same SFT starting checkpoint, RL-trained against frozen-policy, under one
environment, one budget, one seed convention.

| element | value | why fixed this way |
|---|---|---|
| arms | RL from `qwen3.5-9B-sft_torch_dist`, and the same checkpoint with `--lr 0 --weight-decay 0` | isolates the weight update; the frozen arm shares the sampler, the GP, the docking cache and the episode file |
| **checkpoint selection** | **fixed step, decided before any evaluation** — the last step at which the RL arm logged finite gradients, applied identically to both arms | selecting "best" by evaluation score would fit the test set; this rule never reads an evaluation number |
| eval sets | **G12D** in-distribution, **G12C** transfer | as the plan specifies; G12C offline only |
| budget | **80** proposals | plan's value |
| seeds | **5 evaluation seeds per arm per set**, training seed recorded separately and held fixed | training seed and sampling seed are different things and were never distinguished in past logs |
| metric | **Pareto hypervolume**, reference point and objective normalisation declared in the manifest before the first run | a reference point chosen after seeing fronts is a free parameter |
| unit of independence | **the run**, not the proposal | proposals within a run share a GP and a cache |

## Success / failure / insufficient

| verdict | rule |
|---|---|
| **RL gain supported** | HV(RL) − HV(frozen) > 0 on **G12D**, 95% CI excludes 0, across 5 evaluation seeds; and the sign is preserved on **G12C** |
| **RL gain refuted** | CI excludes 0 in the negative direction on G12D |
| **Insufficient** | CI contains 0. Then report the **achieved resolvable |ΔHV|** and stop; do not enlarge the design to chase significance |

Transfer is read separately: a gain on G12D that does not survive on G12C is overfitting to
the training target, which the plan explicitly wants to detect.

## Failure handling

Runs that crash or fall short of budget 80 **stay in the denominator** with their reason
recorded. Removing a failure because its cause looks environmental is selecting on the
outcome — the error this campaign has already made once and documented.

## Cost ceiling and stopping point

| | |
|---|---|
| cost ceiling | **to be set when the design is authorised** — no figure is carried over |
| explicitly not reused | the 27.5 GPU-hour authorisation for the lr=0 control arm was spent (21.93 used) on a different question. **Its remainder is not available to this work** and no part of this design draws on it |
| stopping point | the moment either verdict above is reached on G12D, or the pre-declared seed count is exhausted |

## CPU work that can proceed before any of this

Already done: dependency chain fixed and verified (`code/cpu_eval_prereqs.sh`).
Still open and CPU-only: locate the 8UN5 receptor file (not found within depth 4 of the
project tree), and dry-run the budget-80 loop against the 1.5B pilot checkpoints to
validate the harness — **labelled a harness check, never as a main-objective result**.
