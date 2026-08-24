from __future__ import annotations

import json
from pathlib import Path

import pytest

import ldm_tts.registration.registry as task_registry
import scripts.validate_tasks as validate_tasks_script
from ldm_tts.registration.registry import (
    TASK_DEFINITIONS,
    TaskRegistrationError,
    discover_task_definitions,
    load_task_manifest,
    validate_task_layout,
)
from ldm_tts.registration.scaffold import TaskScaffoldError, scaffold_task
from ldm_tts.registration.qualification import load_qualification_evidence


def test_builtin_tasks_are_discovered_from_manifests() -> None:
    assert set(TASK_DEFINITIONS) == {
        "ai4bio_mutation_effect_prediction",
        "antibody",
        "causal_discovery_discrete",
        "llm_kv_adaptive_quantization",
        "nanogpt",
        "small_molecule",
    }
    for task_id, definition in TASK_DEFINITIONS.items():
        assert definition.relative_root == Path("tasks") / task_id
        assert definition.module == f"tasks.{task_id}.ldm_task.procedure"
        assert definition.manifest_path == Path("tasks") / task_id / "task.json"
        assert definition.dependency_checker


def test_builtin_qualification_evidence_avoids_ignored_runtime_directories() -> None:
    forbidden_parts = {"runs", "ldm_runs", "generated"}
    for task_id, definition in TASK_DEFINITIONS.items():
        evidence_path = (
            Path(__file__).resolve().parents[1]
            / definition.relative_root
            / "resources"
            / "qualification_evidence.json"
        )
        if not evidence_path.is_file():
            continue
        evidence = load_qualification_evidence(
            evidence_path,
            repository_root=Path(__file__).resolve().parents[1],
            expected_task_id=task_id,
        )
        ignored_references = [
            reference
            for gate in evidence.gates.values()
            for reference in gate.evidence
            if forbidden_parts.intersection(Path(reference).parts)
        ]
        assert ignored_references == []


def test_builtin_task_layouts_have_no_validation_errors() -> None:
    for definition in TASK_DEFINITIONS.values():
        issues = validate_task_layout(definition)
        assert [issue for issue in issues if issue.level == "error"] == []


def test_scaffolded_task_is_discoverable_and_valid(tmp_path: Path) -> None:
    (tmp_path / "tasks").mkdir()
    created = scaffold_task(
        "protein_design",
        description="Optimize protein candidates.",
        repository_root=tmp_path,
    )

    assert len(created) == 14
    assert (tmp_path / "tasks" / "protein_design" / "experiment.json") in created
    evidence_path = (
        tmp_path
        / "tasks"
        / "protein_design"
        / "resources"
        / "qualification_evidence.json"
    )
    assert evidence_path in created
    evidence = load_qualification_evidence(
        evidence_path,
        repository_root=tmp_path,
        expected_task_id="protein_design",
    )
    assert evidence.stage == "scaffolded"
    contract_payload = json.loads(
        (tmp_path / "tasks" / "protein_design" / "experiment.json").read_text()
    )
    assert contract_payload["proposal_provider"] == {
        "kind": "unspecified",
        "requires_endpoint_preflight": False,
        "supports_collection": False,
    }
    definitions = discover_task_definitions(tmp_path)
    definition = definitions["protein_design"]
    assert definition.module == "tasks.protein_design.ldm_task.procedure"
    assert definition.dependency_checker is None
    assert validate_task_layout(definition, repository_root=tmp_path) == []
    pyproject = (tmp_path / "tasks" / "protein_design" / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert '"numpy>=1.24"' in pyproject
    assert '"pyyaml>=6.0"' in pyproject
    readme = (tmp_path / "tasks" / "protein_design" / "README.md").read_text(
        encoding="utf-8"
    )
    assert "uv sync --project" in readme
    assert "uv sync --locked" not in readme
    procedure = (
        tmp_path / "tasks" / "protein_design" / "ldm_task" / "procedure.py"
    ).read_text(encoding="utf-8")
    assert "CandidateDomainSpec" in procedure
    assert "ReservoirExpansionSpec" in procedure
    assert "SurrogateSpaceSpec" in procedure
    mock_engine = (
        tmp_path / "tasks" / "protein_design" / "core" / "mock_engine.py"
    ).read_text(encoding="utf-8")
    assert "run_campaign" in mock_engine
    assert "CampaignRecipe" in mock_engine
    assert "DraftCandidateDomain" in mock_engine


def test_qualification_stage_gate_is_explicit_and_ordered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "tasks").mkdir()
    scaffold_task("custom", description="Custom task.", repository_root=tmp_path)
    definitions = discover_task_definitions(tmp_path)
    monkeypatch.setattr(validate_tasks_script, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(validate_tasks_script, "TASK_DEFINITIONS", definitions)
    monkeypatch.setattr(validate_tasks_script, "TASK_DISCOVERY_ERROR", None)

    scaffold_rows = validate_tasks_script.validate_registered_tasks(
        "custom", require_stage="scaffolded"
    )
    assert not any(row["level"] == "error" for row in scaffold_rows)

    registered_rows = validate_tasks_script.validate_registered_tasks(
        "custom", require_stage="registered"
    )
    assert any(
        row["level"] == "error" and "required stage is 'registered'" in row["message"]
        for row in registered_rows
    )


def test_qualification_gate_allows_drafts_only_in_normal_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "tasks").mkdir()
    scaffold_task("custom", description="Custom task.", repository_root=tmp_path)
    definitions = discover_task_definitions(tmp_path)
    monkeypatch.setattr(validate_tasks_script, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(validate_tasks_script, "TASK_DEFINITIONS", definitions)
    monkeypatch.setattr(validate_tasks_script, "TASK_DISCOVERY_ERROR", None)

    normal_rows = validate_tasks_script.validate_registered_tasks("custom")
    assert not any(row["level"] == "error" for row in normal_rows)
    assert any("contract is draft" in row["message"] for row in normal_rows)

    assert validate_tasks_script.main(
        ["--task", "custom", "--require-qualified"]
    ) == 1
    output = capsys.readouterr().out
    assert "[ERROR] custom: Experiment contract is draft" in output


def test_layout_rejects_implementation_inside_adapter(tmp_path: Path) -> None:
    (tmp_path / "tasks").mkdir()
    scaffold_task("custom", description="Custom task.", repository_root=tmp_path)
    extra = tmp_path / "tasks" / "custom" / "ldm_task" / "search.py"
    extra.write_text("", encoding="utf-8")

    definition = discover_task_definitions(tmp_path)["custom"]
    issues = validate_task_layout(definition, repository_root=tmp_path)

    assert [(issue.level, issue.path) for issue in issues] == [("error", extra)]


def test_scaffolder_never_overwrites_existing_task(tmp_path: Path) -> None:
    (tmp_path / "tasks").mkdir()
    scaffold_task("custom", description="Custom task.", repository_root=tmp_path)

    with pytest.raises(TaskScaffoldError, match="Refusing to overwrite"):
        scaffold_task("custom", description="Replacement.", repository_root=tmp_path)


@pytest.mark.parametrize("task_id", ["BadName", "has-hyphen", "9starts_with_digit"])
def test_scaffolder_rejects_non_package_task_ids(tmp_path: Path, task_id: str) -> None:
    (tmp_path / "tasks").mkdir()
    with pytest.raises(TaskScaffoldError, match="task_id"):
        scaffold_task(task_id, description="Invalid.", repository_root=tmp_path)


def test_manifest_rejects_unknown_fields_and_directory_mismatch(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "tasks" / "wrong_directory"
    manifest_dir.mkdir(parents=True)
    manifest = manifest_dir / "task.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "task_id": "different",
        "description": "Mismatch.",
        "extra": True,
    }), encoding="utf-8")

    with pytest.raises(TaskRegistrationError, match="Unknown task manifest"):
        load_task_manifest(manifest, repository_root=tmp_path)

    manifest.write_text(json.dumps({
        "schema_version": 1,
        "task_id": "different",
        "description": "Mismatch.",
    }), encoding="utf-8")
    with pytest.raises(TaskRegistrationError, match="directory name"):
        load_task_manifest(manifest, repository_root=tmp_path)


def test_manifest_rejects_invalid_dependency_hook(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "tasks" / "custom"
    manifest_dir.mkdir(parents=True)
    manifest = manifest_dir / "task.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "task_id": "custom",
        "description": "Custom.",
        "dependency_checker": "not a hook",
    }), encoding="utf-8")

    with pytest.raises(TaskRegistrationError, match="dependency_checker"):
        load_task_manifest(manifest, repository_root=tmp_path)


def test_registry_surfaces_discovery_errors_before_lookup(monkeypatch) -> None:
    error = TaskRegistrationError("broken task manifest")
    monkeypatch.setattr(task_registry, "TASK_DISCOVERY_ERROR", error)

    with pytest.raises(TaskRegistrationError, match="broken task manifest"):
        task_registry.get_task_definition("nanogpt")
