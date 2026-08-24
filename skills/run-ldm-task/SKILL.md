---
name: run-ldm-task
description: Validate, configure, dry-run, smoke-test, execute, monitor, and summarize an existing manifest-registered LDM task through this repository's config runner. Use when asked to run LDM, run a task config or suite, test an existing task, perform a minimal first real run, verify an OpenAI-compatible served model before LDM execution, resume a task run, or diagnose preflight failures for nanogpt, small_molecule, antibody, or another registered task.
---

# Run An Existing LDM Task

Execute registered tasks through their checked-in configs and the shared runner.
Preserve task configuration unless the user asks for an edit. Use temporary
`--set` overrides only when they do not violate a selected experiment-contract
profile.

Read [references/built-in-tasks.md](references/built-in-tasks.md) when running
`nanogpt`, `small_molecule`, or `antibody`. For another task, read its
`tasks/<task_id>/README.md` and `task.json` instead of inventing flags.

## Resolve The Run

1. Work from the repository root.
2. Identify the task, config, requested mode, budget, and acquisition function
   from the request. If the config is unspecified, list configs with:

   ```bash
   python scripts/run_ldm_tts.py --list
   ```

3. Read the selected config, `tasks/<task_id>/task.json`, and the relevant task
   README. Also read `experiment.json` when present. Confirm that the config's
   `task` matches the intended task.
4. Classify the runtime implementation:
   - **Engine-native**: the executed task path calls
     `ldm_tts.campaign.run_campaign` with a `CampaignRecipe`; expect
     the shared lifecycle, budget, event, checkpoint, status, and summary
     artifacts. All built-in tasks (`nanogpt`, `small_molecule`, `antibody`,
     `llm_kv_adaptive_quantization`, `causal_discovery_discrete`,
     `ai4bio_mutation_effect_prediction`) are engine-native.
   - **Compatibility**: only legacy or experimental tasks use a task-specific
     loop or `run_budgeted_search`; follow their README for artifacts, counters,
     and resume behavior.
   - Emitting `LDMTaskSpec` does not by itself make a task engine-native. Verify
     the executed code path rather than inferring runtime ownership from names.
5. Classify the requested execution level:
   - **Inspect**: list or explain configs; make no run.
   - **Mock**: local deterministic execution with no model or evaluator.
   - **Contract**: validate resolution and task specification without objective
     evaluation; use runner `--dry-run` first, then task-level dry/zero work if
     documented.
   - **Tiny real**: one or a few real proposals and evaluations.
   - **Full real**: the checked-in or explicitly overridden production budget.
6. Do not silently promote a mock/contract request to a real run. Do not launch
   a full real budget merely because a tiny real run succeeds.

## Use The Task Environment

Run task-aware commands through the task project:

```bash
uv run --locked --project tasks/<task_id> python <script> ...
```

Use `--no-sync` only when the environment is already prepared and the user does
not want dependency synchronization. If `uv` needs network or user-cache access
and the environment blocks it, request the required approval rather than
bypassing the task environment with an unrelated interpreter.

## Preflight Every Execution

Run these gates in order:

```bash
python scripts/validate_tasks.py --task <task_id>
uv run --locked --project tasks/<task_id> python scripts/check_task_dependencies.py \
  <config_path> [--set path=value ...] [--no-optional]
uv run --locked --project tasks/<task_id> python scripts/run_ldm_tts.py \
  <config_path> --dry-run [--set path=value ...]
```

Use `--no-optional` only when the selected task path genuinely omits those
dependencies. Never use it to suppress a dependency required by the requested
evaluation.

When a config selects `contract_profile`, treat its locked arguments as
immutable. Do not shrink a profiled campaign with `--set`. Use a checked-in
smoke profile when available. Clear `contract_profile` only when the task README
explicitly documents that action for a non-qualified diagnostic or tiny run,
and report that the resulting run is outside the named campaign contract.

Stop before execution when registration validation or a required dependency
check fails. Report the exact failed dependency and configured path. Warnings
may proceed only when they do not invalidate the requested mode.

## Verify Required Providers

Read `experiment.json.proposal_provider` before a tiny or full real run. When
`requires_endpoint_preflight` is true, identify the configured base URL, API key
source, and model ID. Probe both `/models` and `/chat/completions` using the
environment variable names documented by the task. Keep the base URL at the API
root, normally ending in `/v1`. Skip endpoint-only probes when the declared
provider does not require them, while still running its documented dependency
and contract checks.

Do not print, log, commit, or place a real key on a command line as a literal.
Use an existing environment variable. `EMPTY` is acceptable only for a local
server that does not validate credentials. If the endpoint is unreachable, stop
and report it; do not start or replace a model server unless the user asks.

## Execute Progressively

For mock mode, run the checked-in mock config after the three preflight gates.

For real mode, reread and follow the task README's **Minimal First Real Run**
exactly. Do not copy an older recipe from this skill over a newer task README:

1. Run the provider-specific preflight when required.
2. Run the light dependency check.
3. Run the task-level zero-iteration or dry contract smoke.
4. Run the documented tiny real budget.
5. Inspect its summary, trajectory, and failure status.
6. Run the full config only when the user requested full execution or confirms
   escalation after the tiny run.

Use `--set` for temporary output directories and budgets so existing artifacts
are not overwritten. Choose a new run name or trajectory directory unless the
user explicitly requests resume behavior.

## Monitor And Report

Keep long-running commands attached or poll their execution session until they
finish, fail, or the user stops them. Do not leave a model/evaluator run active
without reporting its session state. Track detached work by its durable backend
handle (local PID or remote execution ID), and treat repeated terminal status
or cancellation responses as idempotent.

For an engine-native run, inspect `ldm_task_spec.json`, `events.jsonl`,
`checkpoint.json`, `budget.json`, `status.json`, and `summary.json`. When a
qualified contract is active, also inspect `experiment_contract.json`. Built-in
tasks also re-export their historical trajectory files (see
[references/built-in-tasks.md](references/built-in-tasks.md)). For a
compatibility run, inspect the task-specific artifacts named by its README and
do not claim shared-engine resume or budget semantics unless the executed path
actually provides them.

For a remote backend, pull the complete run directory as an archive when
available rather than reconstructing selected files. A remote backend used for
campaigns should provide task-aware upload, archive pull, and blocking
cancellation (`kill --wait` or equivalent); report missing capabilities instead
of substituting local PID semantics.

After execution, report:

- task, config, mode, and effective overrides;
- interpreter/project used;
- dependency and endpoint results;
- runtime classification and whether a named contract profile remained active;
- output, event/trajectory, checkpoint, status, summary, and best-candidate
  paths that apply to that runtime;
- iterations/evaluations completed and early-stop reason;
- acquisition function and objective directions;
- best observed candidate and objective values when available;
- any skipped integration, warning, failure, or residual external requirement.

Do not claim scientific success from a contract smoke, mock score, surrogate
prediction, or unevaluated candidate.
