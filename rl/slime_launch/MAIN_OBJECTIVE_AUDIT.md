# Main-objective audit — SFT+RL versus SFT-only

Rebuilt 2026-09-05 from the files on disk, not from prose. Scope is SID
`3edc5462-19e3-492e-9aeb-534e7b0e31c5` / Large Discovery Model only.

## What the plan actually asks for

`_wt_handoff/rl/slime_launch/TRAINING_PLAN.md`:

| | |
|---|---|
| question | **Does RL on top of SFT beat SFT-only?** |
| primary metric | **Pareto front hypervolume**, search **budget = 80** |
| evaluation | **G12D** in-distribution and **G12C** transfer (activity model differs, docking receptor 8UN5 shared; G12C is offline-only and never enters the RL loop) |
| runs | R1 base+acq-max, R2 SFT+acq-max (pivot), R3 SFT+real ΔHV, R4 SFT+acq-mean |
| seeds | ≥3–5 per run |

**Spearman(vina, proposal index) and the zero-variance rate are not this metric.** They came
from a side investigation into whether the RL was doing anything at all, and no amount of
either substitutes for a hypervolume number.

## The main results table, as the files support it

| cell | status | evidence |
|---|---|---|
| **HV at budget 80, G12D, any arm** | **NOT-DONE** | zero artifacts. `results/` has 8 files mentioning hypervolume; **none is a budget-80 evaluation output** |
| **HV at budget 80, G12C, any arm** | **NOT-DONE** | same |
| **R1 (base) 9B trained model** | **NOT-DONE** | `R1-base-v2` exists with 3 checkpoints, but its log records **0** grad_norm entries — no optimizer step was ever logged |
| **R2 (SFT) 9B trained model** | **NOT-DONE** | `R2-clip001`, `R2-lr1e8`, `R2-nonanguard`: 168 grad_norm records, **168 nan/inf, 0 finite** |
| **R3 (SFT, ΔHV) 9B trained model** | **NOT-DONE** | `R3a`, `R3f`, `R3j`: 70 grad_norm records, **70 nan/inf, 0 finite** |
| **R4 (SFT, acq-mean)** | **NOT-DONE** | no run directory |
| seeds per run | **NOT-DONE** | no run records `--seed`; training seed is not distinguishable from sampling seed in any log |
| **1.5B pipeline** | **DEMONSTRATED as a pilot only** | 100 of 248 `hidden=1536` run directories carry checkpoints; this is the plumbing working, not the main objective |

### The blocking gap, located precisely

Across **all seven 9B runs that produced a checkpoint**, there are **238 logged gradient
norms and every single one is nan or inf — zero finite values.** Those checkpoints are
therefore "SFT weights plus some number of nan updates", not trained models. Initialisation
was verified from expanded argv rather than run names: `R1-base-v2` loads
`qwen3.5-9B_torch_dist` (base) and all R2/R3 runs load `qwen3.5-9B-sft_torch_dist` (SFT),
which does match the design — the arms are right, the optimisation never worked.

**No 9B model exists that can be evaluated.** Every downstream question about RL gain is
blocked on that, not on evaluation capacity.

## What the earlier lr=0 work does and does not settle

Kept as-is, and it is a diagnostic, not the main result: with the policy provably frozen
(226/226 parameter tensors byte-identical across 2,944.8 MiB) the docking objective still
fails to improve, so a weight change is not necessary for the decline. Difference not
detected (diff −0.0660, 95% CI [−0.1475, +0.0155], p=0.0961), equivalence not established
(TOST upper p=0.0011 rejects *the trained arm exceeding the control by 0.10*; lower p=0.176
does not reject the reverse), underpowered at the frozen margin. All of it is a
**descriptive** contrast between two conditionally-selected groups, on **1.5B**, and it
says nothing about hypervolume.

## CPU work completed this round

The evaluation chain could not run at all before today. No single environment had rdkit and
sklearn/joblib and gpytorch simultaneously: `miniforge3` lacked rdkit/gpytorch/botorch,
`bigbang129` lacked sklearn/joblib. Fixed in user site, shared environments untouched.

The QSAR pickles then failed with `ModuleNotFoundError: No module named 'train_g12c_qsar'`,
which reads like a missing dependency but is a path problem: the models were saved by that
script running as `__main__`, so loading requires its directory on `sys.path`.

Now verified end to end on CPU, reproducible via `code/cpu_eval_prereqs.sh`:

- 11/11 packages import
- **G12C** `AverageRegressor`, **G12D** `Pipeline`, both load and predict from raw SMILES
  (`["CCO", "c1ccccc1C(=O)N"]` → G12C `[5.244, 4.949]`, G12D `[5.270, 5.130]`)
- botorch `Hypervolume` computes (worked example HV = 4.5 at ref `[-12, -2]`)
- 6 of 7 `tasks.small_molecule.core` modules import; the 7th needed `gauche`, now installed

Still missing and **not** resolvable on CPU alone: the 8UN5 receptor file was not found
within depth 4 of the project tree, so real docking cannot be confirmed runnable yet.

## Verdicts

| | |
|---|---|
| **Supported** | The 1.5B pipeline runs end to end. The evaluation chain's CPU half now works and is reproducible. The lr=0 diagnostic's frozen-weight claim. |
| **Refuted** | Nothing in the main objective — there is no main-objective result to refute. Retracted separately: "raw_reward declines therefore the policy got worse" (a frozen policy reproduces the decline) and the gradient attributions listed in `plan/PLAN.md`. |
| **Not done** | **The entire main objective.** No hypervolume at budget 80 on either G12D or G12C, for any arm, because no 9B run ever completed an optimizer step with a finite gradient. |
