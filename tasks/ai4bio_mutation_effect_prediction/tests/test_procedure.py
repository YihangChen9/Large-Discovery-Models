from __future__ import annotations

import json
from pathlib import Path

import pytest

from ldm_tts.contracts import RawProposal
from ldm_tts.data import DataCollectionSink
from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.optimization.gp import RBFGPUCBSelector
from ldm_tts.registration.experiment import (
    ExperimentContractError,
    load_experiment_contract,
    validate_profile_args,
)
from tasks.ai4bio_mutation_effect_prediction.core.candidate import (
    PARAMETER_LIMIT,
    MutationPredictorCandidateDomain,
    normalize_predictor_spec,
    predictor_parameter_count,
    render_predictor_source,
)
from tasks.ai4bio_mutation_effect_prediction.core.evaluator import (
    MLSBenchMutationEvaluator,
    OFFICIAL_COMMIT,
    TASK_PATH,
    UPSTREAM_ROOT_SHA256,
    UPSTREAM_SHA256,
    fixed_template_sha256,
    geometric_mean_nonnegative,
    materialize_harness,
    parse_test_metrics,
)
from tasks.ai4bio_mutation_effect_prediction.core.proposals import (
    DeterministicPredictorExpander,
)
from tasks.ai4bio_mutation_effect_prediction.core.surrogate import (
    FEATURE_DIMENSION,
    FEATURE_VERSION,
    PredictorSpecEncoder,
)
from tasks.ai4bio_mutation_effect_prediction.ldm_task.procedure import (
    describe_ldm_task,
    main,
    parse_args,
)


BASE_SPEC = {
    "feature_mode": "concat",
    "hidden_dims": [256, 128],
    "activation": "gelu",
    "dropout": 0.1,
    "layer_norm": True,
    "learning_rate": 0.001,
    "weight_decay": 0.05,
}


def test_candidate_contract_materialization_and_collection(tmp_path: Path) -> None:
    domain = MutationPredictorCandidateDomain(DataCollectionSink(tmp_path / "collection"))
    admitted = domain.admit(
        RawProposal(BASE_SPEC, "test_model", {"collectable": True})
    )
    assert admitted.candidate_id.startswith("predictor-")
    assert admitted.metadata["parameter_count"] == predictor_parameter_count(BASE_SPEC)
    assert admitted.metadata["parameter_count"] <= PARAMETER_LIMIT
    assert "class MutationPredictor(nn.Module):" in admitted.payload["code"]
    assert "torch.cat([embedding, delta_embedding]" in admitted.payload["code"]

    rejected = domain.admit(
        RawProposal({**BASE_SPEC, "hidden_dims": [2048]}, "test_model")
    )
    assert rejected.reason == "invalid_predictor"
    ir = json.loads((tmp_path / "collection" / "ldm_ir.jsonl").read_text())
    sft = json.loads((tmp_path / "collection" / "ldm_sft.jsonl").read_text())
    assert ir["schema_version"] == "ldm-2.0"
    assert ir["collection"]["provenance"]["candidate_id"] == admitted.candidate_id
    assert "collection" not in sft["instruction"]


@pytest.mark.parametrize(
    "replacement, message",
    [
        ({"feature_mode": "tokens"}, "feature_mode"),
        ({"activation": "tanh"}, "activation"),
        ({"dropout": float("nan")}, "finite"),
        ({"learning_rate": 1.0}, "learning_rate"),
        ({"layer_norm": 1}, "boolean"),
        ({"extra": True}, "unknown"),
    ],
)
def test_candidate_rejects_invalid_specs(replacement, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_predictor_spec({**BASE_SPEC, **replacement})


def test_rendered_linear_candidate_has_exact_interface() -> None:
    source = render_predictor_source(
        {**BASE_SPEC, "feature_mode": "delta", "hidden_dims": [], "layer_norm": False}
    )
    assert "nn.Linear(1280, 1)" in source
    assert "x = delta_embedding" in source
    assert "return self.network(x).squeeze(-1)" in source


def test_reservoir_is_distinct_and_gp_scores_every_candidate() -> None:
    proposals = DeterministicPredictorExpander().expand(
        ExpansionRequest(round_idx=0, reservoir_size=4)
    ).proposals
    assert {proposal.source for proposal in proposals} == {"deterministic_catalog"}
    domain = MutationPredictorCandidateDomain()
    candidates = [domain.admit(proposal) for proposal in proposals]
    assert len({candidate.canonical_key for candidate in candidates}) == 4
    encoder = PredictorSpecEncoder()
    vectors = {candidate.candidate_id: encoder.encode(candidate) for candidate in candidates}
    assert all(len(vector.values) == FEATURE_DIMENSION for vector in vectors.values())
    selection = RBFGPUCBSelector(
        objective_name="selection_score", feature_version=FEATURE_VERSION
    ).select(candidates, vectors, count=1)
    assert len(selection.selected_candidate_ids) == 1
    assert {item.candidate_id for item in selection.predictions} == {
        candidate.candidate_id for candidate in candidates
    }


def test_metric_parsing_and_aggregation() -> None:
    assert parse_test_metrics("TEST_METRICS spearman=0.625", "BLAT_ECOLX") == {
        "spearman_BLAT_ECOLX": 0.625
    }
    score = geometric_mean_nonnegative(
        {
            "spearman_BLAT_ECOLX": 0.5,
            "spearman_ESTA_BACSU": 0.5,
            "spearman_RASH_HUMAN": 0.5,
        }
    )
    assert score == pytest.approx(0.5)


def test_materialization_changes_only_the_two_editable_regions() -> None:
    lines = [f"# fixed line {index}" for index in range(1, 370)]
    lines[104] = "# EDITABLE SECTION START - MutationPredictor"
    lines[139] = "# EDITABLE SECTION END"
    lines[342] = "# EDITABLE SECTION START - CONFIG_OVERRIDES"
    lines[348] = "# EDITABLE SECTION END"
    original = "\n".join(lines) + "\n"
    source = render_predictor_source(BASE_SPEC)
    materialized = materialize_harness(
        original,
        source,
        {
            "learning_rate": BASE_SPEC["learning_rate"],
            "weight_decay": BASE_SPEC["weight_decay"],
        },
    )
    assert fixed_template_sha256(materialized) == fixed_template_sha256(original)
    assert source.rstrip() in materialized
    assert "CONFIG_OVERRIDES = {'learning_rate': 0.001, 'weight_decay': 0.05}" in materialized


def test_task_spec_has_no_scaffold_placeholders() -> None:
    args = parse_args(["--mock", "--reservoir-size", "4"])
    payload = json.dumps(describe_ldm_task(args).to_dict(), sort_keys=True)
    assert "replace" + "_me" not in payload
    assert "selection_score" in payload
    assert "mutation_predictor_spec_v1" in payload


def test_mock_procedure_writes_engine_artifacts_and_exact_counters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv("LDM_DATA_COLLECTION_ENABLED", "1")
    assert main(
        [
            "--mock",
            "--iterations",
            "1",
            "--reservoir-size",
            "4",
            "--out-dir",
            str(tmp_path),
            "--run-name",
            "mock_run",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    run_dir = Path(output["run_dir"])
    for name in (
        "budget.json",
        "campaign.json",
        "checkpoint.json",
        "events.jsonl",
        "experiment_contract.json",
        "status.json",
        "summary.json",
        "ldm_task_spec.json",
        "search_manifest.json",
        "selection_record.json",
    ):
        assert (run_dir / name).is_file(), name
    assert (run_dir / "ldm_data" / "ldm_ir.jsonl").is_file()
    assert output["engine_summary"]["successful_evaluation_count"] == 1
    counters = json.loads((run_dir / "budget.json").read_text())["counters"]
    assert counters == {
        "benchmark_jobs": 1,
        "expensive_evaluation_attempts": 1,
        "external_evaluations": 1,
        "llm_requests": 0,
        "outer_iterations": 1,
        "proposal_attempts": 0,
        "selected_candidates": 1,
        "successful_evaluations": 1,
        "valid_search_candidates": 4,
    }


def test_upstream_provenance_is_loaded_once_and_artifacts_are_run_relative(
    tmp_path: Path,
) -> None:
    assert OFFICIAL_COMMIT == "cfd57a7e0139c72753e32e31bca593719b098717"
    assert TASK_PATH == "tasks/ai4bio-mutation-effect-prediction"
    assert "edits/custom_template.py" in UPSTREAM_SHA256
    assert "src/mlsbench/scoring/evaluate.py" in UPSTREAM_ROOT_SHA256

    run_dir = tmp_path / "campaign"
    artifact = run_dir / "evaluations" / "candidate" / "evaluation_manifest.json"
    evaluator = MLSBenchMutationEvaluator(
        upstream_root=tmp_path / "upstream",
        data_dir=tmp_path / "data",
        cv_dir=tmp_path / "cv",
        run_dir=run_dir,
    )
    assert evaluator._artifact_path(artifact) == (
        "evaluations/candidate/evaluation_manifest.json"
    )


def test_real_budget_counts_all_three_official_jobs() -> None:
    from ldm_tts.campaign import CampaignBudget

    from tasks.ai4bio_mutation_effect_prediction.core.evaluator import OFFICIAL_ASSAYS

    budget = CampaignBudget(
        rounds=1,
        reservoir_size=4,
        batch_size=1,
        extra_limits={"benchmark_jobs": 1 * 1 * len(OFFICIAL_ASSAYS)},
    )
    assert budget.runtime_limits()["benchmark_jobs"] == 3


def test_official_campaign_profile_locks_budget_and_search_topology() -> None:
    contract = load_experiment_contract(
        Path(__file__).resolve().parents[1] / "experiment.json"
    )
    assert contract.qualification == "qualified"
    assert contract.proposal_provider == {
        "kind": "deterministic",
        "requires_endpoint_preflight": False,
        "supports_collection": True,
    }
    selection_metric = next(
        metric
        for metric in contract.metrics["optimized"]
        if metric["name"] == "selection_score"
    )
    assert selection_metric["modes"] == ["mock"]
    profile = validate_profile_args(
        contract,
        "official_campaign",
        {
            "iterations": 1,
            "reservoir-size": 4,
            "evaluations-per-round": 1,
            "proposal-mode": "deterministic",
            "acquisition-beta": 1.0,
            "evaluation-timeout": 3540,
        },
    )
    assert profile.budget["benchmark_jobs"] == 3
    with pytest.raises(ExperimentContractError, match="reservoir-size"):
        validate_profile_args(
            contract,
            "official_campaign",
            {
                **profile.locked_args,
                "reservoir-size": 5,
            },
        )


def test_three_iteration_profile_scales_extended_campaign_budget() -> None:
    contract = load_experiment_contract(
        Path(__file__).resolve().parents[1] / "experiment.json"
    )
    profile = validate_profile_args(
        contract,
        "official_campaign_3_iterations",
        {
            "iterations": 3,
            "reservoir-size": 4,
            "evaluations-per-round": 1,
            "proposal-mode": "deterministic",
            "acquisition-beta": 1.0,
            "evaluation-timeout": 3540,
        },
    )
    assert profile.budget == {
        "outer_iterations": 3,
        "llm_requests": 0,
        "proposal_attempts": 0,
        "valid_search_candidates": 12,
        "selected_candidates": 3,
        "external_evaluations": 3,
        "expensive_evaluation_attempts": 3,
        "successful_evaluations": 3,
        "benchmark_jobs": 9,
    }


def test_twenty_iteration_profile_scales_extended_campaign_budget() -> None:
    contract = load_experiment_contract(
        Path(__file__).resolve().parents[1] / "experiment.json"
    )
    profile = validate_profile_args(
        contract,
        "official_campaign_20_iterations",
        {
            "iterations": 20,
            "reservoir-size": 4,
            "evaluations-per-round": 1,
            "proposal-mode": "deterministic",
            "acquisition-beta": 1.0,
            "evaluation-timeout": 3540,
        },
    )
    assert profile.budget == {
        "outer_iterations": 20,
        "llm_requests": 0,
        "proposal_attempts": 0,
        "valid_search_candidates": 80,
        "selected_candidates": 20,
        "external_evaluations": 20,
        "expensive_evaluation_attempts": 20,
        "successful_evaluations": 20,
        "benchmark_jobs": 60,
    }


def test_resume_does_not_repeat_completed_evaluation(tmp_path: Path, capsys) -> None:
    argv = [
        "--mock",
        "--iterations",
        "1",
        "--reservoir-size",
        "4",
        "--out-dir",
        str(tmp_path),
        "--run-name",
        "resume",
    ]
    assert main(argv) == 0
    run_dir = Path(json.loads(capsys.readouterr().out)["run_dir"])
    before = json.loads((run_dir / "budget.json").read_text())["counters"]
    assert main(argv + ["--resume-from", str(run_dir)]) == 0
    capsys.readouterr()
    after = json.loads((run_dir / "budget.json").read_text())["counters"]
    assert after == before
