# Task Contract Reference

## Registration

Required layout:

```text
tasks/<task_id>/
├── task.json
├── experiment.json
├── README.md
├── pyproject.toml
├── __init__.py
├── ldm_task/
│   ├── __init__.py
│   ├── procedure.py
│   └── dependencies.py
├── core/
│   └── __init__.py
├── resources/
│   ├── README.md
│   └── qualification_evidence.json
└── tests/

config/<task_id>/
└── mock.yaml
```

Manifest schema version 1 accepts only:

```json
{
  "schema_version": 1,
  "task_id": "example_task",
  "description": "One-line domain description.",
  "dependency_checker": "tasks.example_task.ldm_task.dependencies:check_dependencies"
}
```

`dependency_checker` is optional. The directory name and `task_id` must match.
The module and working directory are inferred as
`tasks.<task_id>.ldm_task.procedure` and `tasks/<task_id>`.

## Experiment Contract

`experiment.json` remains optional only for legacy task discovery. It is
required for newly scaffolded tasks and for any campaign qualification claim.
New scaffolds include a valid draft contract. It records benchmark provenance,
proposal-provider capabilities, metric roles, official evaluator settings and
per-candidate limits, generic budget facts, and named campaign profiles. A
profile can lock config arguments; select it with top-level `contract_profile`
in a real config. The shared runner rejects missing or changed locked arguments
before task import.

Declare `proposal_provider.kind` as `unspecified`, `deterministic`,
`model_endpoint`, `external_service`, `dataset`, `simulator`, or `hybrid`.
Also declare `requires_endpoint_preflight` and `supports_collection`. A
`model_endpoint` provider must require preflight; a deterministic provider must
not. Older contracts without this object load as `unspecified` for compatibility.

Use `qualification: draft` until the official source and seed evaluator are
verified. Qualified runs should call
`load_active_experiment_contract()` and
`snapshot_experiment_contract(contract, run_dir, profile=profile)`.

Metric roles are explicit:

- `reported`: official benchmark and comparison outputs;
- `optimized`: continuous objectives used by the surrogate/acquisition;
- `diagnostic`: component metrics and operational measurements.

The same metric may be both reported and optimized. Never silently substitute a
surrogate-only transformation for the official reported metric.
Use optional metric `modes` when a metric exists only in selected modes, such as
`["mock"]` for a synthetic selection score.

## Qualification Evidence

New scaffolds create `resources/qualification_evidence.json` at `scaffolded`.
Advance it through `registered`, `mock_verified`, `contract_verified`,
`seed_evaluated`, `tiny_campaign_verified`, and `campaign_qualified`. Every
stage through the declared current stage must be `passed` and cite at least one
existing repository-relative evidence path that is available in a clean
checkout. Do not cite ignored runtime output; promote a compact campaign record
with result, budget, provenance, contract digest, and raw-artifact digests into
the task's `resources/` directory. Campaign-qualified evidence pins the
benchmark commit and names a profile defined by `experiment.json`.

Legacy tasks may omit this file during normal validation. A new qualification
claim must pass `scripts/validate_tasks.py --task <task_id> --require-stage
<stage>`.

## Procedure

Required external interface:

```python
def main(argv: list[str] | None = None) -> int | None:
    ...
```

Recommended task-local interfaces:

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ...

def describe_ldm_task(...) -> LDMTaskSpec:
    ...
```

`describe_ldm_task` must use the vocabulary in `docs/concepts.md` and separately
declare:

- the candidate domain and validation rules;
- the finite reservoir and every action that can expand it;
- the expansion schema policy when structured proposal parameters may change;
- the surrogate representation, encoder, version, and dimension policy;
- response contracts, objectives, proposal-search topology, and acquisition.

Every `ReservoirExpansionSpec.response_space` must name a declared
`ResponseSpaceSpec`. At least one expansion action must produce candidates;
schema-only actions cannot form a runnable task by themselves.

New procedures should run the campaign through the shared
`ldm_tts.campaign.run_campaign` algorithm with task-owned behavioral adapters
under `core/`:

```python
result = run_campaign(
    CampaignRequest(
        run_dir=run_dir,
        budget=CampaignBudget(
            rounds=iterations,
            reservoir_size=reservoir_size,
            batch_size=evaluations_per_round,
            target_observations=iterations * evaluations_per_round,
        ),
        config=jsonable_args,
        resume=resume_requested,
        artifact_projector=materialize_task_artifacts,  # optional legacy exports
    ),
    CampaignRecipe(
        task_spec=describe_ldm_task(args),
        expander=task_expander,
        candidate_domain=task_candidate_domain,
        evaluator=task_evaluator,
        surrogate_encoder=task_encoder,  # optional, paired with selector
        selector=task_selector,          # optional, paired with encoder
    ),
)
```

Candidate admission returns `Candidate` or `CandidateRejection`; external
evaluation returns `EvaluationResult`. The shared campaign algorithm is
responsible for runtime creation, reservoir deduplication, observation
construction, objective validation, budget enforcement, events, checkpoints,
failure classification, and summaries. Task code must not open
`CampaignRuntime`, assemble budget ledgers, or duplicate those policies around
the campaign.

Do not treat emitting `LDMTaskSpec` or importing shared optimization helpers as
a campaign migration. A task is engine-native only when its executed campaign
runs through `run_campaign` (or constructs `LDMEngine` directly when a
specialized lifecycle is required) and delegates lifecycle ownership to the
shared algorithm. When repairing a legacy task, identify compatibility paths
explicitly and migrate them without silently changing budgets, artifacts, or
resume behavior.

The runner applies config environment variables, changes to the task directory,
imports the conventional module, and calls `main(argv)`. The task owns all
domain execution behind that interface. Keep `ldm_task/procedure.py` as a thin
adapter; put importable reservoir-expansion, model, surrogate, and evaluator
implementation under `core/`, and versioned runtime inputs under `resources/`.

## Dependency Hook

Use this exact callable shape:

```python
from typing import Any

from ldm_tts.registration.dependencies import DependencyCheck, plan_check_context


def check_dependencies(
    plan: dict[str, Any], *, include_optional: bool = True
) -> list[DependencyCheck]:
    task, args, env, cwd, mode = plan_check_context(plan)
    ...
```

Return only `DependencyCheck` objects. Use `ok`, `warn`, `fail`, and `skip`
from `ldm_tts.registration.dependencies`. Never print or return unmasked credentials.

## Fine-Tuning Data Collection

Tasks that accept model-generated actions must expose an opt-in runtime
collection path through the public `ldm_tts.data` module:

```python
from ldm_tts.data import DataCollectionSink

sink = DataCollectionSink.from_env(default_root=run_dir / "ldm_data")
```

Append only after the task parser and validator have accepted the action. Build
the canonical `ldm-2.0` IR from the accepted payload, not from an unvalidated
response transcript. Store run IDs, task-specific source IDs, selection results,
and evaluator outcomes under `collection.provenance` or `collection.outcome` so
the renderer cannot leak them into the prompt.

Do not collect rejected attempts, deterministic/random fallbacks, or a different
semantic response type under an existing dataset contract. For example, direct
sequence proposals and search-policy DSL updates require separate action
contracts. Use a run-local ignored `ldm_data/` directory unless
`LDM_DATA_COLLECTION_DIR` explicitly selects an aggregate campaign directory.

The mock task test must enable `LDM_DATA_COLLECTION_ENABLED=1`, execute at least
one accepted action, validate the emitted IR, and verify that collection-only
metadata is absent from rendered SFT instructions. If collection is inapplicable,
the task README must state why and identify the future accepted-action boundary.

## Completion Gates

- `scripts/validate_tasks.py --task <task_id>` has no errors.
- `ldm_task/` contains only the runner and dependency-check adapters.
- Importable implementation and static inputs live in `core/` and `resources/`.
- Task tests pass in the task environment.
- Mock dependency check passes without external systems.
- Mock runner dry-run resolves the registered module and task directory.
- Mock runner execution succeeds.
- Mock execution runs through the shared campaign algorithm and writes
  campaign, event, checkpoint, status, budget, task-spec, and summary
  artifacts.
- Shared tests and `git diff --check` pass.
- The mock collection test emits valid `ldm-2.0` IR, or the task documents why
  its response contract is not collectable.
- No scaffold placeholders, task-name dispatch branches, secrets, or generated
  artifacts remain.
- `experiment.json` is valid and its qualification state is reported honestly.
- Qualified tasks pass `scripts/validate_tasks.py --task <task_id>
  --require-qualified`.
- Real configs select a runner-enforced profile when official budgets are known.
- Qualified campaigns snapshot the contract and emit durable budget/status files.
- Qualified run artifacts use run-relative references; budget snapshots include
  zero-valued counters; completed scalar campaigns prefer `result.json` and
  `trajectory.csv` exports.
