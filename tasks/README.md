# Registering LDM Tasks

Each directory under `tasks/` is a domain adapter behind the shared runner
interface. A task registers itself by adding a versioned `task.json` manifest;
no shared Python registry or dependency-check dispatch table needs editing.

## Standard Layout

```text
tasks/<task_id>/
├── task.json                 # registration manifest
├── experiment.json           # benchmark, metrics, evaluator, and budget contract
├── README.md                 # domain setup and run tutorial
├── QUICKSTART.md             # validated clean-room workflow
├── pyproject.toml            # isolated task dependencies
├── __init__.py
├── ldm_task/
│   ├── __init__.py
│   ├── procedure.py          # stable shared-runner adapter
│   └── dependencies.py       # optional dependency-check adapter
├── core/                     # private task implementation
│   └── __init__.py
├── resources/                # versioned schemas, seeds, inputs, models
│   ├── README.md
│   └── qualification_evidence.json  # ordered machine-readable gate evidence
├── scripts/                  # optional maintenance/training CLIs
├── environments/             # optional Conda or external-tool specs
├── tests/                    # task-local tests
│   └── test_procedure.py
└── runs/                     # generated runtime artifacts; Git-ignored

config/<task_id>/
├── mock.yaml                 # local, service-free smoke run
└── real.yaml                 # real model and evaluator settings
```

`ldm_task` is the external seam. The shared runner and dependency checker know
only `procedure.main(argv)` and the optional manifest hook. Keep implementation
out of this package so the interface remains small and stable.

The remaining directories have one ownership rule each:

- `core/` contains importable search, surrogate, evaluator, and model-client code.
- `resources/` contains versioned non-generated inputs required at runtime.
- `scripts/` contains auxiliary CLIs that are not runner entrypoints.
- `environments/` contains optional environment specifications beyond `pyproject.toml`.
- `tests/` verifies the task interface and implementation.
- `runs/` contains all generated artifacts and must never be committed.

Do not create alternate task entrypoints or implementation packages beside
these directories. Add domain-specific subpackages inside `core/` and organize
resource types inside `resources/` instead.

## Scaffold A Task

From the repository root:

```bash
python scripts/scaffold_task.py protein_design \
  --description "Optimize protein candidates against structure objectives."
```

The command creates `tasks/protein_design/` and
`config/protein_design/mock.yaml`. It never overwrites an existing task. The
generated mock adapter runs immediately, but deliberately contains semantic
placeholders that must be replaced before a real run.

## Registration Manifest

`task.json` uses schema version 1:

```json
{
  "schema_version": 1,
  "task_id": "protein_design",
  "description": "Optimize protein candidates against structure objectives.",
  "dependency_checker": "tasks.protein_design.ldm_task.dependencies:check_dependencies"
}
```

Rules:

- `task_id` must match the directory name and be a lowercase Python identifier.
- `description` must be a non-empty one-line domain description.
- `dependency_checker` is optional. When present, it must use
  `python.module:function` notation.
- Module and working-directory paths are convention-derived and cannot be
  overridden by the manifest:
  `tasks.<task_id>.ldm_task.procedure` and `tasks/<task_id>`.
- Unknown manifest fields and schema versions fail registration.

The runner discovers manifests when a process starts. Adding the directory and
manifest is sufficient to make a task ID available to configs.

## Experiment Contract

New tasks also carry `experiment.json`. `task.json` answers "what adapter is
registered?" while `experiment.json` answers "what scientific and operational
claim does this run enforce?" The contract records immutable benchmark
provenance, reported/optimized/diagnostic metric roles, official evaluator
settings, per-candidate limits, proposal-provider capabilities, and named
campaign profiles. Endpoint preflight is required only when
`proposal_provider.requires_endpoint_preflight` is true.

Scaffolds begin at `qualification: draft`. Change this to `qualified` only after
one official-budget seed evaluation and a tiny LDM-selected real evaluation pass.
Real configs select a profile at top level:

```yaml
contract_profile: official_campaign
```

The shared runner validates the profile's locked arguments before importing the
task procedure. Qualified procedures snapshot the active contract into the run
directory and use `ldm_tts.engine.run_store` for durable `budget.json` and `status.json`.

New scaffolds also start
`resources/qualification_evidence.json` at `scaffolded`. Advance it through
`registered`, `mock_verified`, `contract_verified`, `seed_evaluated`,
`tiny_campaign_verified`, and `campaign_qualified`, citing existing
repository-relative artifacts for every passed gate. Validate an explicit claim
with `scripts/validate_tasks.py --task <task_id> --require-stage <stage>`.

Those evidence paths must exist in a clean checkout. Do not cite ignored
`runs/`, `ldm_runs/`, or `data/generated/` files; promote the relevant campaign
result, contract and artifact digests, counters, metrics, and provenance into a
compact versioned record under the task's `resources/` directory.

A path that exists only as an untracked working-tree file is not clean-checkout
evidence. Before advancing a qualification gate, inspect `git status --short
--untracked-files=all -- tasks/<task_id> config/<task_id>` and require
`git ls-files --error-unmatch -- <evidence-path>` to succeed for every cited
path. This check prevents a local pass followed by a CI failure because a cited
mock or real config was never added to Git.

## Procedure Interface

`ldm_task/procedure.py` must define:

```python
def main(argv: list[str] | None = None) -> int | None:
    ...
```

The shared runner changes into the task directory, applies config environment
variables, imports the procedure module, and calls `main(plan_argv)`. The
adapter owns argument parsing, task-contract description, execution dispatch,
and result exit status. Candidate generation, model calls, surrogate fitting,
acquisition, and evaluators belong in `core/`. Versioned schemas and seed
inputs belong in `resources/`.

New tasks should implement mock and real campaigns through
`ldm_tts.campaign.run_campaign`. Supply a `CampaignRecipe` with the minimum
behavioral adapters: `ReservoirExpander`, `CandidateDomainAdapter`, and
`CandidateEvaluator`. Add `SurrogateEncoder` plus `AcquisitionSelector` only
when the task uses surrogate-guided selection. Keep the procedure adapter
limited to CLI parsing, `LDMTaskSpec` construction, dependency preparation, and
campaign dispatch; do not open `CampaignRuntime` or assemble budget ledgers in
task workflows.

The campaign algorithm creates authoritative `Candidate`, `EvaluationResult`, and
`Observation` records. Do not introduce task-local equivalents unless the task
payload needs a private intermediate record behind an engine adapter. Use
`CampaignRuntime` for run identity, contract snapshots, budgets, events,
checkpoints, status, and final summaries.

For consistency and inspectability, also define:

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ...

def describe_ldm_task(...) -> LDMTaskSpec:
    ...
```

`describe_ldm_task` is task-internal and may accept domain-specific prepared
objects. It must describe the candidate domain, reservoir-expansion actions,
surrogate representation, measured objectives, model response contracts, and
acquisition rule using the shared `ldm_tts.contracts` types. Every reservoir
expansion must reference a declared response space, and at least one expansion
must produce candidates. Keep domain dependencies and encodings inside the task
implementation.

Every task must provide a deterministic `mode: mock` config that avoids remote
models, external evaluators, GPUs, and large datasets. This is the contract test
used before real dependencies are introduced.

## Optional Dependency Hook

A task-specific hook receives the already-resolved runner plan and returns
shared `DependencyCheck` records:

```python
from typing import Any

from ldm_tts.registration.dependencies import (
    DependencyCheck,
    ok,
    plan_check_context,
)


def check_dependencies(
    plan: dict[str, Any], *, include_optional: bool = True
) -> list[DependencyCheck]:
    task, args, env, cwd, mode = plan_check_context(plan)
    return [ok(task, "adapter", "Task dependency hook loaded.", str(cwd))]
```

Declare the hook in `task.json`. Keep domain checks in the task module; reuse
shared helpers for LLM settings, paths, and CUDA when useful. If the manifest
omits the hook, the dependency checker returns a warning rather than blocking
the task. Keep `dependencies.py` import-light: import optional scientific or
model packages inside the check function so the hook can report that they are
missing instead of failing during module import.

## Fine-Tuning Data Collection Contract

Every task that accepts model-generated actions must make an explicit data
collection decision. Use `DataCollectionSink.from_env` from `ldm_tts.data` and a
run-local `<run_dir>/ldm_data` default when the accepted action can be represented
as canonical `ldm-2.0` IR. Collection remains off unless
`LDM_DATA_COLLECTION_ENABLED` or `LDM_DATA_COLLECTION_DIR` enables it.

The hook belongs at the accepted-action boundary: after response parsing and
task validation, but before downstream evaluation changes what was visible to
the model. Build the target from the accepted structured payload. Never train on
the first raw response, rejected retries, or random fallbacks. Put run identity,
selection results, and evaluator outcomes under `collection.provenance` or
`collection.outcome`; those fields are intentionally excluded from rendered
instructions.

Different semantic actions need different contracts. A direct candidate
proposal must not share a target schema with a search-policy or DSL update unless
the inference API intentionally supports both. When a task cannot yet emit a
trainable action, document the reason and intended future boundary in its README.

Mock coverage must enable collection, execute one accepted action, validate the
emitted IR, and confirm collection-only metadata is absent from rendered SFT.
Stable run/task provenance is required because fine-tuning preparation groups
related records to prevent split leakage.

## Config Contract

Create configs under `config/<task_id>/` using the registered `task_id`:

```yaml
name: protein_design_mock
task: protein_design
algorithm: mean
mode: mock
args:
  mock: true
  iterations: 1
```

Task CLI flags belong under `args`, environment variables under `env`, and
literal positional arguments under `extra_args`. A config normally does not
need `runner.cwd` or `runner.module`; those escape hatches are intended for
temporary experiments, not registration.

Use `contract_profile` for real configs whose evaluator/search budgets must not
drift. Do not select a qualified profile from a mock config.

## Required Verification

Run these checks in order:

```bash
python scripts/validate_tasks.py --task protein_design
uv run --locked --project tasks/protein_design python -m pytest tasks/protein_design/tests
python scripts/check_task_dependencies.py config/protein_design/mock.yaml --no-optional
python scripts/run_ldm_tts.py config/protein_design/mock.yaml --dry-run
python scripts/run_ldm_tts.py config/protein_design/mock.yaml
```

Before adding a real config, replace every generated placeholder, document the
model endpoint and evaluator requirements in the task README, and add a staged
first-real-run sequence: endpoint probe, dependency check, zero-iteration or
dry contract run, then a tiny evaluated run.

## Registration Checklist

- The manifest validates without warnings or errors.
- `experiment.json` identifies metric roles and honestly reports `draft` or
  `qualified` status.
- `main(argv)` runs through the shared runner rather than a separate launcher.
- `ldm_task/` contains only the adapter files accepted by task validation.
- Importable implementation lives under `core/`; versioned inputs live under `resources/`.
- Every generated file is written beneath `runs/` or another explicitly ignored temporary path.
- `describe_ldm_task` matches actual candidates, objectives, response parsing,
  and acquisition behavior.
- Acquisition scoring uses `ldm_tts.optimization.acquisition` unless a documented domain
  algorithm requires additional task-local behavior.
- Mock tests cross the same procedure interface as real runs.
- Secrets are supplied through environment variables or ignored local files.
- Generated runs, caches, model downloads, and virtual environments remain
  untracked.
- Qualified runs snapshot the contract, enforce a budget ledger, emit status
  heartbeats, serialize zero-valued counters, and run provider-required
  preflight before iteration 1.
- Run artifact references are portable and relative to the run directory;
  scalar campaigns should export `result.json` and `trajectory.csv` when
  practical.
