"""Engine integration tests for the nanoGPT model-based campaign.

The model-based loop now runs through ``ldm_tts.engine.LDMEngine``: one engine
round is one model-based iteration, the depth traversal lives inside the
reservoir expander, and the engine owns evaluations, budgets, events, and
checkpoints. These tests drive ``core.workflow.main`` end to end with the
deterministic mock generator and mock training script.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tasks.nanogpt.core import workflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _mock_argv(tmp_path: Path, run_name: str, **overrides) -> list[str]:
    argv = [
        "--project-root",
        str(PROJECT_ROOT),
        "--train-file",
        "resources/train/mock_train.py",
        "--operation-schema",
        "resources/schemas/mock_operations.json",
        "--generator",
        "operation_mock",
        "--method",
        "best_of_n",
        "--breadth",
        "2",
        "--depth",
        "1",
        "--iterations",
        "2",
        "--warmup",
        "1",
        "--seed-policy",
        "best",
        "--initial-expansion-parameters",
        "5",
        "--max-expansion-parameters",
        "0",
        "--disable-expansion-schema-updates",
        "--allow-new-expansion-parameters",
        "--mock-expand-every",
        "0",
        "--surrogate-mode",
        "lcb",
        "--gp-beta",
        "1.0",
        "--gp-xi",
        "0.001",
        "--out-dir",
        str(tmp_path),
        "--run-name",
        run_name,
        "--eval-command",
        "python {train_path}",
    ]
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    return argv


def test_model_based_campaign_runs_through_engine(tmp_path):
    rc = workflow.main(_mock_argv(tmp_path, "engine_ok"))
    assert rc == 0
    run_dir = tmp_path / "engine_ok"
    for name in (
        "campaign.json",
        "events.jsonl",
        "checkpoint.json",
        "summary.json",
        "budget.json",
        "status.json",
        "ldm_task_spec.json",
        "model_based_summary.json",
    ):
        assert (run_dir / name).exists(), name

    events = [
        json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]
    event_types = [event["event_type"] for event in events]
    # Warm-up and optimization both run through the shared campaign.
    assert event_types.count("reservoir_expanded") == 3
    assert event_types.count("candidate_evaluated") == 3
    assert "campaign_finished" in event_types
    selections = [event for event in events if event["event_type"] == "candidates_selected"]
    assert selections
    assert selections[0]["payload"]["metadata"]["mode"] == "warmup_order"
    assert all(
        event["payload"]["metadata"]["mode"] == "precomputed_surrogate_score"
        for event in selections[1:]
    )

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["engine_summary"]["successful_evaluation_count"] == 3

    model_based = json.loads(
        (run_dir / "model_based_summary.json").read_text(encoding="utf-8")
    )
    iterations = model_based["iterations"]
    assert len(iterations) == 2
    assert all(record["selected_state_id"] for record in iterations)
    assert all(record["selected_real_score"] is not None for record in iterations)
    assert model_based["best_state_id"] == iterations[-1]["selected_state_id"]


def test_skip_eval_runs_rounds_without_evaluations(tmp_path):
    rc = workflow.main(_mock_argv(tmp_path, "engine_skip", skip_eval=True))
    assert rc == 0
    run_dir = tmp_path / "engine_skip"
    events = [
        json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]
    event_types = [event["event_type"] for event in events]
    assert event_types.count("reservoir_expanded") == 2
    assert event_types.count("candidate_evaluated") == 0
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["engine_summary"]["successful_evaluation_count"] == 0


def test_resume_extends_campaign_through_engine_checkpoint(tmp_path):
    assert workflow.main(_mock_argv(tmp_path, "engine_resume")) == 0
    run_dir = tmp_path / "engine_resume"
    first_summary = json.loads(
        (run_dir / "model_based_summary.json").read_text(encoding="utf-8")
    )
    assert len(first_summary["iterations"]) == 2

    assert (
        workflow.main(
            _mock_argv(tmp_path, "engine_resume", iterations=1, resume_from=str(run_dir))
        )
        == 0
    )
    resumed = json.loads(
        (run_dir / "model_based_summary.json").read_text(encoding="utf-8")
    )
    assert len(resumed["iterations"]) == 3
    assert [record["iteration"] for record in resumed["iterations"]] == [1, 2, 3]
    events = [
        json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]
    event_types = [event["event_type"] for event in events]
    assert event_types.count("reservoir_expanded") == 4
    assert event_types.count("candidate_evaluated") == 4
    assert event_types.count("campaign_resumed") == 1


def test_warmup_and_engine_share_max_real_evaluation_cap(tmp_path):
    assert (
        workflow.main(
            _mock_argv(
                tmp_path,
                "engine_cap",
                max_real_evaluations=1,
            )
        )
        == 0
    )
    run_dir = tmp_path / "engine_cap"
    model_based = json.loads(
        (run_dir / "model_based_summary.json").read_text(encoding="utf-8")
    )
    events = [
        json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]

    assert model_based["warmup"]["real_evaluations"] == 1
    assert sum(record["real_evaluations"] for record in model_based["iterations"]) == 0
    assert sum(event["event_type"] == "candidate_evaluated" for event in events) == 1
