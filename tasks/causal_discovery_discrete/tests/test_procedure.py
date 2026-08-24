from __future__ import annotations

import json
from pathlib import Path

import pytest

from ldm_tts.contracts import RawProposal
from ldm_tts.data import DataCollectionSink
from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.optimization.gp import RBFGPUCBSelector
from ldm_tts.registration.experiment import ExperimentContractError, load_experiment_contract, validate_profile_args
from tasks.causal_discovery_discrete.core.candidate import CausalAlgorithmCandidateDomain, normalize_algorithm_spec
from tasks.causal_discovery_discrete.core.evaluator import OFFICIAL_COMMIT, TASK_PATH
from tasks.causal_discovery_discrete.core.proposals import DeterministicCausalExpander
from tasks.causal_discovery_discrete.core.surrogate import CausalSpecEncoder, FEATURE_DIMENSION, FEATURE_VERSION
from tasks.causal_discovery_discrete.ldm_task.procedure import describe_ldm_task, main, parse_args


BASE_SPEC = {"min_association": 0.05, "max_degree": 6}


def test_candidate_contract_and_collection(tmp_path: Path) -> None:
    domain = CausalAlgorithmCandidateDomain(DataCollectionSink(tmp_path / "collection"))
    admitted = domain.admit(RawProposal(BASE_SPEC, "test_model", {"collectable": True}))
    assert admitted.candidate_id.startswith("causal-")
    assert admitted.payload["spec"] == BASE_SPEC
    rejected = domain.admit(RawProposal({**BASE_SPEC, "max_degree": 0}, "test_model"))
    assert rejected.reason == "invalid_algorithm"
    ir = json.loads((tmp_path / "collection" / "ldm_ir.jsonl").read_text())
    sft = json.loads((tmp_path / "collection" / "ldm_sft.jsonl").read_text())
    assert ir["schema_version"] == "ldm-2.0"
    assert "collection" not in sft["instruction"]


@pytest.mark.parametrize("replacement", [{"min_association": float("nan")}, {"min_association": 2.0}, {"max_degree": True}, {"max_degree": 21}, {"extra": 1}])
def test_candidate_rejects_invalid_specs(replacement) -> None:
    with pytest.raises(ValueError):
        normalize_algorithm_spec({**BASE_SPEC, **replacement})


def test_reservoir_and_surrogate_are_distinct() -> None:
    proposals = DeterministicCausalExpander().expand(ExpansionRequest(round_idx=0, reservoir_size=4)).proposals
    domain = CausalAlgorithmCandidateDomain()
    candidates = [domain.admit(proposal) for proposal in proposals]
    assert len({candidate.canonical_key for candidate in candidates}) == 4
    encoder = CausalSpecEncoder()
    vectors = {candidate.candidate_id: encoder.encode(candidate) for candidate in candidates}
    assert all(len(vector.values) == FEATURE_DIMENSION for vector in vectors.values())
    selection = RBFGPUCBSelector(objective_name="selection_score", feature_version=FEATURE_VERSION).select(candidates, vectors, count=1)
    assert len(selection.selected_candidate_ids) == 1


def test_task_spec_has_no_placeholders() -> None:
    payload = json.dumps(describe_ldm_task(parse_args(["--mock"])).to_dict(), sort_keys=True)
    assert "replace" + "_me" not in payload
    assert FEATURE_VERSION in payload


def test_mock_writes_complete_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("LDM_DATA_COLLECTION_ENABLED", "1")
    assert main(["--mock", "--iterations", "1", "--reservoir-size", "4", "--out-dir", str(tmp_path), "--run-name", "mock"]) == 0
    output = json.loads(capsys.readouterr().out)
    run_dir = Path(output["run_dir"])
    for name in ("budget.json", "campaign.json", "checkpoint.json", "events.jsonl", "experiment_contract.json", "status.json", "summary.json", "ldm_task_spec.json", "search_manifest.json", "selection_record.json", "result.json", "trajectory.csv"):
        assert (run_dir / name).is_file(), name
    assert (run_dir / "ldm_data" / "ldm_ir.jsonl").is_file()
    assert output["engine_summary"]["successful_evaluation_count"] == 1
    counters = json.loads((run_dir / "budget.json").read_text())["counters"]
    assert counters == {"benchmark_jobs": 1, "expensive_evaluation_attempts": 1, "external_evaluations": 1, "llm_requests": 0, "outer_iterations": 1, "proposal_attempts": 0, "selected_candidates": 1, "successful_evaluations": 1, "valid_search_candidates": 4}


def test_upstream_pin_and_twenty_iteration_profile() -> None:
    assert OFFICIAL_COMMIT == "cfd57a7e0139c72753e32e31bca593719b098717"
    assert TASK_PATH == "tasks/causal-discovery-discrete"
    contract = load_experiment_contract(Path(__file__).resolve().parents[1] / "experiment.json")
    assert contract.qualification == "qualified"
    profile = validate_profile_args(contract, "official_campaign_20_iterations", {"iterations": 20, "reservoir-size": 4, "evaluations-per-round": 1, "proposal-mode": "deterministic", "acquisition-beta": 1.0, "evaluation-timeout": 3540})
    assert profile.budget["benchmark_jobs"] == 100
    with pytest.raises(ExperimentContractError):
        validate_profile_args(contract, "official_campaign_20_iterations", {**profile.locked_args, "reservoir-size": 5})
