---
name: register-ldm-task
description: Scaffold, implement, register, scientifically qualify, and production-check an LDM domain task in this repository. Use when adding or repairing a task adapter, task manifest, experiment.json benchmark contract, proposal-provider capabilities, metric roles, qualification evidence, official evaluation budget, campaign profile, dependency checker, mock/real config, GP-guided search, durable budget/status reporting, or staged real-run qualification.
---

# Register And Qualify An LDM Task

Build a domain adapter through the manifest-driven task seam, then qualify its
scientific and operational contract before calling it production-ready.

Read [references/task-contract.md](references/task-contract.md) before editing.
Read [references/qualification.md](references/qualification.md) before adding a
real config or launching an external evaluator. Treat `tasks/README.md` as the
authoritative human-facing repository contract when present.

## Establish The Contract

Before scaffolding, discover or ask for:

- the candidate domain and its parser/validator boundary;
- each reservoir-expansion action and whether it emits candidates, configures a
  generator, edits a candidate, or updates the expansion schema;
- the surrogate representation, dimension policy, encoder, and version;
- the benchmark source URL, immutable commit, and task path;
- the proposal provider kind, whether it requires endpoint preflight, and whether
  accepted actions support fine-tuning collection;
- reported, optimized, and diagnostic metrics with directions;
- one expensive evaluation and its official per-candidate limits;
- search, LLM-attempt, expensive-evaluation, and baseline budgets;
- required datasets, artifacts, binaries, accelerators, and seed observations;
- resume expectations, comparison axis, and required run artifacts.

Do not infer an official budget from a smoke run. Record unknowns explicitly and
keep `experiment.json` at `qualification: draft` until primary-source evidence
and real evaluator checks support `qualified`.

## Implement Registration

1. Inspect `tasks/README.md`, `ldm_tts/registration/registry.py`, the closest task, and
   the domain benchmark.
2. Select a lowercase Python `task_id`. Confirm `tasks/<task_id>/` and
   `config/<task_id>/` do not already exist.
3. Run the non-overwriting scaffolder:

   ```bash
   python scripts/scaffold_task.py <task_id> --description "<one-line description>"
   ```

4. Replace every semantic placeholder. Keep `ldm_task/procedure.py` shallow;
   put candidate, prompt, reservoir-expansion, surrogate-adapter, and evaluator
   code in `core/`, versioned inputs in `resources/`, and outputs in ignored
   `runs/`.
5. Complete `experiment.json`. Keep registration identity in `task.json`; keep
   scientific provenance, proposal-provider capabilities, metric roles,
   evaluator settings, limits, and named runner-enforced campaign profiles in
   `experiment.json`.
6. Implement the campaign through `ldm_tts.campaign.run_campaign`. Supply a
   `CampaignRecipe` with task-owned `ReservoirExpander`,
   `CandidateDomainAdapter`, and `CandidateEvaluator` adapters. Add
   `SurrogateEncoder` and `AcquisitionSelector` only for surrogate-guided
   methods. Declare absolute budgets with `CampaignBudget`; do not open
   `CampaignRuntime` or assemble budget ledgers in task code. Use
   `ProposalClient` for model transport.
   Reuse `ldm_tts.optimization.search`, `ldm_tts.optimization.gp`, and
   `ldm_tts.optimization.acquisition`
   behind those adapters before adding task-local infrastructure.
7. Define the fine-tuning collection boundary after response parsing and
   validation. Append canonical `ldm-2.0` IR through
   `DataCollectionSink.from_env`; keep provenance/outcomes out of model-visible
   state and never collect rejected attempts or incompatible fallback actions.
8. Add a lightweight `dependencies.py` hook only for meaningful external
   prerequisites. Never import optional heavy dependencies at module import.

## Qualify In Stages

Finish each stage before starting the next:

1. `registered`: manifest, layout, draft experiment contract, and imports.
2. `mock_verified`: deterministic mock run and collection test.
3. `contract_verified`: candidate parser plus CPU/GPU tensor, parameter, and
   evaluator assembly checks.
4. `seed_evaluated`: one official-budget seed evaluation with source commit,
   metrics, and artifacts recorded.
5. `tiny_campaign_verified`: provider-specific preflight when required, one
   generated reservoir, one acquisition-selected candidate, and one real
   evaluation.
6. `campaign_qualified`: named contract profile, durable resume, budget/status
   files, comparison budget, and monitored durable launch.

Use the exact gates and expected artifacts in
[references/qualification.md](references/qualification.md). Registration is not
the same as campaign qualification; report both states explicitly. Update
`resources/qualification_evidence.json` at each completed gate and cite only
repository-relative evidence paths that exist in a clean checkout. Never cite
ignored `runs/`, `ldm_runs/`, or `data/generated/` output: promote the relevant
contract digest, counters, result, provenance, and raw-artifact digests into a
compact checked-in record under `tasks/<task_id>/resources/`.

Local existence is not sufficient: every cited evidence path must also be
tracked by Git. Before advancing a gate, run `git status --short --untracked-files=all
-- tasks/<task_id> config/<task_id>` and verify each cited path with
`git ls-files --error-unmatch -- <repository-relative-evidence-path>`. This is
especially important for `config/<task_id>/mock.yaml`: an untracked config can
make local validation pass while a clean GitHub Actions checkout reports that
the task has no config directory and that the evidence path is missing.

## Required Verification

Run from the repository root:

```bash
python scripts/validate_tasks.py --task <task_id>
git status --short --untracked-files=all -- tasks/<task_id> config/<task_id>
# Repeat for every path cited by resources/qualification_evidence.json.
git ls-files --error-unmatch -- <repository-relative-evidence-path>
uv run --locked --project tasks/<task_id> python -m pytest tasks/<task_id>/tests
uv run --locked --project tasks/<task_id> python scripts/check_task_dependencies.py \
  config/<task_id>/mock.yaml --no-optional
uv run --locked --project tasks/<task_id> python scripts/run_ldm_tts.py \
  config/<task_id>/mock.yaml --dry-run
uv run --locked --project tasks/<task_id> python scripts/run_ldm_tts.py \
  config/<task_id>/mock.yaml
python -m pytest -q tests
git diff --check
```

After the seed and tiny LDM gates justify changing the contract to `qualified`,
also run `python scripts/validate_tasks.py --task <task_id> --require-qualified`.
For a campaign-readiness claim, also run `python scripts/validate_tasks.py
--task <task_id> --require-stage campaign_qualified`. The normal validator
accepts honest draft scaffolds and warns when legacy evidence is absent.

Before a real launch, also verify the selected `contract_profile` appears in the
runner dry run, any provider-required preflight succeeds, `status.json` and
`budget.json` are created, and the first selected candidate enters the intended
evaluator rather than a standalone benchmark agent.

## Interface Rules

- Register only through `task.json`; do not add a central task-name branch.
- Use the canonical terms in `docs/concepts.md`: candidate domain, reservoir,
  reservoir expansion, expansion schema, and surrogate representation.
- Keep `experiment.json` versioned, strict, secret-free, and source-pinned.
- Put task CLI options under config `args`, environment values under `env`, and
  select enforced production settings with top-level `contract_profile`.
- Define `main(argv)`, `parse_args`, and a runtime-faithful `describe_ldm_task`.
- Do not call a task engine-native merely because it emits `LDMTaskSpec`; the
  executed campaign must run through `run_campaign` and delegate lifecycle
  ownership to the shared campaign algorithm.
- Make the deterministic mock execute at least one complete shared-campaign
  round and assert that `events.jsonl`, `checkpoint.json`, and `summary.json`
  exist.
- Count LLM calls, valid search states, selected candidates, expensive attempts,
  successful evaluations, benchmark jobs, and outer iterations separately.
- Serialize every declared budget counter, including zeros, and preserve true
  fractional values while writing integral values as JSON integers.
- Use expensive evaluations, not wall time or generated states, as the default
  fair-comparison x-axis unless the benchmark specifies otherwise.
- Snapshot the active experiment contract into every qualified run.
- Preflight before iteration 1 only when the declared proposal provider requires
  it. Pause resumably when a required service circuit opens; deterministic,
  dataset-backed, or simulator providers must not be blocked by endpoint-only
  gates.
- Store artifact references relative to the run directory. Prefer a portable
  `result.json` and `trajectory.csv` for completed campaigns.
- Represent detached work with a backend-neutral durable execution handle, not
  a local-PID assumption.
- Keep credentials in environment variables or ignored protected files. Never
  write them to configs, logs, manifests, prompts, or command arguments.

## External Execution Backends

When registration is tested through Delta or another remote backend, require a
task-aware upload bundle, archive download for complete run directories,
idempotent terminal lifecycle operations, and a blocking `kill --wait`
equivalent. Treat those as backend/CLI requirements; do not imitate them with
repository-local PID files or partial artifact copies.

## Existing Task Repair

Run validation first. Preserve the stable task ID and config paths. Add
`experiment.json` without changing schema-version-1 `task.json`, migrate generic
GP/budget/endpoint behavior to shared modules where practical, and retain
compatibility exports when callers already import task-local names.
