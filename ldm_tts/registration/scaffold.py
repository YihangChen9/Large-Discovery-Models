"""Deterministic scaffolding for manifest-registered LDM tasks."""

from __future__ import annotations

import json
from pathlib import Path

from ldm_tts.registration.registry import REPO_ROOT, TASK_ID_PATTERN


class TaskScaffoldError(ValueError):
    """Raised when a task skeleton cannot be created without overwriting files."""


def scaffold_task(
    task_id: str,
    *,
    description: str,
    repository_root: Path = REPO_ROOT,
) -> tuple[Path, ...]:
    """Create a conventional task skeleton and mock experiment config."""

    task_id = str(task_id).strip()
    description = str(description).strip()
    if not TASK_ID_PATTERN.fullmatch(task_id):
        raise TaskScaffoldError(
            "task_id must start with a lowercase letter and contain only "
            "lowercase letters, digits, and underscores"
        )
    if not description:
        raise TaskScaffoldError("description must not be empty")

    repository_root = Path(repository_root).resolve()
    task_root = repository_root / "tasks" / task_id
    config_root = repository_root / "config" / task_id
    files = _task_files(task_id, description, task_root, config_root)
    conflicts = sorted(path for path in files if path.exists())
    if task_root.exists() or conflicts:
        conflict_text = ", ".join(str(path) for path in conflicts) or str(task_root)
        raise TaskScaffoldError(
            f"Refusing to overwrite existing task files: {conflict_text}"
        )

    created: list[Path] = []
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        created.append(path)
    return tuple(created)


def _task_files(
    task_id: str,
    description: str,
    task_root: Path,
    config_root: Path,
) -> dict[Path, str]:
    package_name = task_id.replace("_", "-")
    manifest = {
        "schema_version": 1,
        "task_id": task_id,
        "description": description,
    }
    return {
        task_root / "task.json": json.dumps(manifest, indent=2) + "\n",
        task_root / "experiment.json": _experiment_contract_template(task_id),
        task_root / "__init__.py": f'"""{description}"""\n',
        task_root / "ldm_task" / "__init__.py": '"""Shared-runner adapter for this task."""\n',
        task_root / "ldm_task" / "procedure.py": _procedure_template(task_id),
        task_root / "core" / "__init__.py": '"""Private task implementation."""\n',
        task_root / "core" / "mock_engine.py": _mock_engine_template(task_id),
        task_root / "resources" / "README.md": _resources_readme_template(task_id),
        task_root / "resources" / "qualification_evidence.json": (
            _qualification_evidence_template(task_id)
        ),
        task_root / "tests" / "__init__.py": "",
        task_root / "tests" / "test_procedure.py": _test_template(task_id),
        task_root / "README.md": _readme_template(task_id, description),
        task_root / "pyproject.toml": _pyproject_template(package_name, description),
        config_root / "mock.yaml": _config_template(task_id),
    }


def _procedure_template(task_id: str) -> str:
    return f'''#!/usr/bin/env python3
"""Procedure adapter for the ``{task_id}`` task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ldm_tts.contracts import (
    AcquisitionSpec,
    CandidateDomainSpec,
    LDMTaskSpec,
    ObjectiveSpec,
    ReservoirExpansionSpec,
    ReservoirSpec,
    ResponseSpaceSpec,
    ProposalSearchSpec,
    SurrogateSpaceSpec,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the {task_id} LDM task.")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/mock"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def describe_ldm_task(args: argparse.Namespace) -> LDMTaskSpec:
    return LDMTaskSpec(
        task="{task_id}",
        candidate_domain=CandidateDomainSpec(
            name="replace_me",
            kind="replace_me",
            dimension=None,
            representation="Replace with the task candidate representation.",
        ),
        objectives=(
            ObjectiveSpec(
                name="objective",
                direction="maximize",
                description="Replace with the measured task objective.",
            ),
        ),
        response_spaces=(
            ResponseSpaceSpec(
                name="proposal",
                output_kind="json",
                description="Replace with the model response contract.",
            ),
        ),
        acquisition=AcquisitionSpec(
            name="mean",
            objective_names=("objective",),
            score_direction="maximize",
            selection_rule="Replace with the task selection rule.",
        ),
        reservoir=ReservoirSpec(
            name="candidate_reservoir",
            expansions=(
                ReservoirExpansionSpec(
                    name="direct_proposal",
                    action_kind="emit_candidate",
                    response_space="proposal",
                    produces_candidates=True,
                    description="Replace with the task's reservoir expansion action.",
                ),
            ),
            candidate_validator="Replace with the task candidate validator.",
            deduplication_key="Replace with the canonical candidate identity.",
        ),
        surrogate=SurrogateSpaceSpec(
            kind="none",
            representation="not configured in the draft scaffold",
            dimension_policy="none",
        ),
        proposal_search=ProposalSearchSpec(name="single_turn"),
        metadata={{"mock": bool(args.mock)}},
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.mock and not args.dry_run:
        raise SystemExit("Implement real task execution before running without --mock.")
    task_spec = describe_ldm_task(args)
    run_dir = args.out_dir
    payload = {{
        "task": "{task_id}",
        "iterations": max(0, args.iterations),
        "mock": bool(args.mock),
        "ldm_task_spec": task_spec.to_dict(),
    }}
    if args.mock and not args.dry_run:
        from ldm_tts.engine.run_store import unique_run_dir
        from tasks.{task_id}.core.mock_engine import run_mock_campaign

        run_dir = unique_run_dir(args.out_dir)
        result = run_mock_campaign(
            task_spec,
            iterations=max(0, args.iterations),
            run_dir=run_dir,
        )
        payload["engine_summary"] = result.engine.summary
        payload["run_dir"] = str(run_dir.resolve())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _mock_engine_template(task_id: str) -> str:
    return f'''"""Deterministic shared-campaign smoke adapter for ``{task_id}``."""

from __future__ import annotations

from pathlib import Path

from ldm_tts.campaign import (
    CampaignBudget,
    CampaignRecipe,
    CampaignRequest,
    CampaignResult,
    run_campaign,
)
from ldm_tts.contracts import (
    CallableCandidateEvaluator,
    Candidate,
    CandidateRejection,
    LDMTaskSpec,
    RawProposal,
)
from ldm_tts.engine.expansion import (
    CallableReservoirExpander,
    ExpansionRequest,
    ExpansionResult,
)


class DraftCandidateDomain:
    """Replace with task-owned canonicalization and scientific validation."""

    def admit(self, proposal: RawProposal) -> Candidate | CandidateRejection:
        try:
            value = int(proposal.payload)
        except (TypeError, ValueError):
            return CandidateRejection("invalid", "mock integer required", proposal.source)
        return Candidate(
            candidate_id=f"draft-{{value}}",
            payload=value,
            canonical_key=str(value),
            source=proposal.source,
        )


def _expand(request: ExpansionRequest) -> ExpansionResult:
    return ExpansionResult(
        proposals=(RawProposal(request.round_idx, "deterministic_mock"),),
    )


def run_mock_campaign(
    task_spec: LDMTaskSpec,
    *,
    iterations: int,
    run_dir: Path,
) -> CampaignResult:
    return run_campaign(
        CampaignRequest(
            run_dir=run_dir,
            budget=CampaignBudget(
                rounds=iterations,
                reservoir_size=1,
                batch_size=1,
                target_observations=iterations,
            ),
            config={{"mode": "mock", "iterations": iterations}},
        ),
        CampaignRecipe(
            task_spec=task_spec,
            expander=CallableReservoirExpander(_expand),
            candidate_domain=DraftCandidateDomain(),
            evaluator=CallableCandidateEvaluator(
                lambda candidate: {{"objective": float(candidate.payload)}}
            ),
        ),
    )
'''


def _experiment_contract_template(task_id: str) -> str:
    payload = {
        "schema_version": 1,
        "task_id": task_id,
        "qualification": "draft",
        "benchmark": {
            "source_url": "local://unqualified",
            "source_commit": "unqualified",
        },
        "proposal_provider": {
            "kind": "unspecified",
            "requires_endpoint_preflight": False,
            "supports_collection": False,
        },
        "metrics": {
            "reported": [{"name": "objective", "direction": "maximize"}],
            "optimized": [{"name": "objective", "direction": "maximize"}],
            "diagnostic": [],
        },
        "evaluation": {
            "datasets": ["mock"],
            "settings": {},
            "per_candidate_limits": {},
        },
        "budget": {},
        "profiles": {},
    }
    return json.dumps(payload, indent=2) + "\n"


def _qualification_evidence_template(task_id: str) -> str:
    pending = {"status": "pending", "evidence": []}
    payload = {
        "schema_version": 1,
        "task_id": task_id,
        "stage": "scaffolded",
        "benchmark_commit": "unqualified",
        "contract_profile": "",
        "gates": {
            "scaffolded": {
                "status": "passed",
                "evidence": [
                    f"tasks/{task_id}/task.json",
                    f"tasks/{task_id}/experiment.json",
                ],
            },
            "registered": dict(pending),
            "mock_verified": dict(pending),
            "contract_verified": dict(pending),
            "seed_evaluated": dict(pending),
            "tiny_campaign_verified": dict(pending),
            "campaign_qualified": dict(pending),
        },
    }
    return json.dumps(payload, indent=2) + "\n"


def _test_template(task_id: str) -> str:
    return f'''from tasks.{task_id}.ldm_task.procedure import main, parse_args


def test_mock_procedure(tmp_path, capsys) -> None:
    assert parse_args(["--mock", "--iterations", "0"]).iterations == 0
    assert main([
        "--mock",
        "--iterations",
        "1",
        "--out-dir",
        str(tmp_path / "mock_run"),
    ]) == 0
    assert '"task": "{task_id}"' in capsys.readouterr().out
    assert (tmp_path / "mock_run" / "events.jsonl").is_file()
    assert (tmp_path / "mock_run" / "summary.json").is_file()
'''


def _readme_template(task_id: str, description: str) -> str:
    return f'''# {task_id}

{description}

## Mock Run

From the repository root:

```bash
uv sync --project tasks/{task_id} --group dev
uv run --project tasks/{task_id} \\
  python scripts/run_ldm_tts.py config/{task_id}/mock.yaml
```

The mock path already exercises the shared `run_campaign` algorithm. Replace the
generated candidate-domain admission adapter, reservoir expander, evaluator,
surrogate representation, objective, and response contract before adding a
real-run config. Complete `experiment.json`
with source-pinned benchmark provenance, metric roles, evaluator limits, and a
named campaign profile. Keep `qualification` set to `draft` until a real seed
and tiny LDM-selected evaluation pass.
'''


def _resources_readme_template(task_id: str) -> str:
    return f'''# {task_id} Resources

Store versioned, non-generated task inputs here: schemas, prompts, seed
programs, small reference datasets, or redistributable model artifacts.
Runtime outputs belong under `../runs/` and must not be committed.
'''


def _pyproject_template(package_name: str, description: str) -> str:
    escaped_description = description.replace('"', '\\"')
    return f'''[project]
name = "ldm-tts-{package_name}"
version = "0.1.0"
description = "{escaped_description}"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "pyyaml>=6.0",
]

[dependency-groups]
dev = [
    "pytest>=7.0",
]

[tool.uv]
package = false
'''


def _config_template(task_id: str) -> str:
    return f'''name: {task_id}_mock
task: {task_id}
algorithm: mean
mode: mock
description: Local contract smoke test for {task_id}.
args:
  mock: true
  iterations: 1
'''


__all__ = ["TaskScaffoldError", "scaffold_task"]
