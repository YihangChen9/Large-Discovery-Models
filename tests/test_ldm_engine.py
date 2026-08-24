from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import pytest

from ldm_tts.contracts import (
    AcquisitionSpec,
    Candidate,
    CandidateDomainSpec,
    CandidateRejection,
    CallableCandidateEvaluator,
    EvaluationResult,
    LDMTaskSpec,
    ObjectiveSet,
    ObjectiveSpec,
    Observation,
    RawProposal,
    ReservoirBuilder,
    ReservoirExpansionSpec,
    ReservoirSpec,
    ResponseSpaceSpec,
    SurrogateSpaceSpec,
)
from ldm_tts.engine.run_store import BudgetExceededError, CampaignRuntime, unique_run_dir
from ldm_tts.optimization.records import BOObservation, SurrogateVector
from ldm_tts.engine import LDMEngine, LDMEngineConfig, LDMEngineState
from ldm_tts.engine.expansion import CallableReservoirExpander, ExpansionResult
from ldm_tts.engine.expansion import DirectEmissionExpander, ExpansionRequest
from ldm_tts.campaign import (
    CampaignBudget,
    CampaignRecipe,
    CampaignRequest,
    run_campaign,
)
from ldm_tts.optimization.gp import RBFGPSurrogate, RBFGPUCBSelector, select_max_ucb_record
from ldm_tts.transport import CallableProposalClient, ProposalRequest


@dataclass
class IntegerDomain:
    maximum: int = 9

    def admit(self, proposal: RawProposal) -> Candidate | CandidateRejection:
        try:
            value = int(proposal.payload)
        except (TypeError, ValueError):
            return CandidateRejection("invalid", "integer required", proposal.source)
        if not 0 <= value <= self.maximum:
            return CandidateRejection("out_of_range", source=proposal.source)
        return Candidate(
            candidate_id=f"integer-{value}",
            payload=value,
            canonical_key=str(value),
            source=proposal.source,
        )


class IntegerEncoder:
    def describe(self) -> SurrogateSpaceSpec:
        return SurrogateSpaceSpec(
            "vector",
            "integer scalar",
            "fixed",
            dimension=1,
            version="integer-v1",
        )

    def encode(self, candidate: Candidate) -> SurrogateVector:
        return SurrogateVector(
            (float(candidate.payload),),
            "integer-v1",
            source_id=candidate.candidate_id,
        )


def test_reservoir_builder_owns_history_deduplication_and_capacity() -> None:
    result = ReservoirBuilder(IntegerDomain()).build(
        [
            RawProposal("1", "llm-a"),
            RawProposal(1, "llm-b"),
            RawProposal(2, "llm-a"),
            RawProposal(3, "llm-a"),
            RawProposal("bad", "llm-a"),
            RawProposal(12, "llm-a"),
        ],
        evaluated_keys={"2"},
        max_size=1,
        metadata={"round": 4},
    )

    assert [item.payload for item in result.candidates] == [1]
    assert result.drop_counts == {
        "duplicate": 1,
        "already_evaluated": 1,
        "reservoir_capacity": 1,
        "invalid": 1,
        "out_of_range": 1,
    }
    assert result.metadata == {"round": 4}
    assert result.to_dict()["rejections"][0]["metadata"]["duplicate_of"] == "integer-1"


def test_reservoir_builder_rejects_invalid_adapter_results() -> None:
    class BrokenDomain:
        def admit(self, proposal: RawProposal):
            return None

    with pytest.raises(TypeError, match="must return Candidate"):
        ReservoirBuilder(BrokenDomain()).build([RawProposal(1, "broken")])


def test_candidate_identity_and_reservoir_size_are_validated() -> None:
    with pytest.raises(ValueError, match="candidate_id"):
        Candidate("", 1, "1")
    with pytest.raises(ValueError, match="canonical_key"):
        Candidate("one", 1, "")
    with pytest.raises(ValueError, match="max_size"):
        ReservoirBuilder(IntegerDomain()).build([], max_size=0)


def make_observation(candidate_id: str, x: int, **metrics: float) -> Observation:
    candidate = Candidate(candidate_id, x, str(x), source="test")
    return Observation(
        candidate,
        EvaluationResult(candidate_id, "succeeded", metrics=metrics),
    )


def test_objective_set_validates_directions_and_selects_incumbent() -> None:
    objectives = ObjectiveSet((ObjectiveSpec("loss", "minimize"),))
    first = make_observation("one", 1, loss=2.0)
    second = make_observation("two", 2, loss=1.0)

    assert objectives.names == ("loss",)
    assert objectives.minimize == (True,)
    assert objectives.to_vector(second.metrics) == (1.0,)
    assert objectives.orient_for_maximization(second.metrics) == (-1.0,)
    assert objectives.incumbent([first, second]) == second
    assert objectives.is_better(second.metrics, first.metrics)


def test_objective_set_computes_multiobjective_pareto_front() -> None:
    objectives = ObjectiveSet(
        (
            ObjectiveSpec("cost", "minimize"),
            ObjectiveSpec("quality", "maximize"),
        )
    )
    cheap = make_observation("cheap", 1, cost=1.0, quality=0.5)
    strong = make_observation("strong", 2, cost=2.0, quality=0.9)
    dominated = make_observation("dominated", 3, cost=3.0, quality=0.4)

    assert objectives.pareto_front([cheap, strong, dominated]) == (cheap, strong)
    with pytest.raises(ValueError, match="exactly one"):
        objectives.incumbent([cheap])


def test_observation_and_objective_validation_reject_malformed_results() -> None:
    candidate = Candidate("one", 1, "1")
    with pytest.raises(ValueError, match="must match"):
        Observation(candidate, EvaluationResult("two", "failed"))
    with pytest.raises(ValueError, match="missing objective"):
        ObjectiveSet((ObjectiveSpec("loss", "minimize"),)).validate_metrics({})
    with pytest.raises(ValueError, match="finite"):
        ObjectiveSet((ObjectiveSpec("loss", "minimize"),)).validate_metrics(
            {"loss": float("nan")}
        )
    with pytest.raises(ValueError, match="resource usage"):
        EvaluationResult("one", "succeeded", {"loss": 1.0}, resource_usage={"gpu_s": -1})


def test_campaign_runtime_persists_events_budgets_checkpoint_and_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    runtime = CampaignRuntime.open(
        run_dir,
        task="integer_search",
        config={"seed": 7},
        task_spec={"task": "integer_search"},
        budget_limits={"evaluations": 2},
        contract_snapshot={"qualification": "test"},
    )

    assert runtime.consume("evaluations") == 1
    event = runtime.record("candidate_evaluated", {"score": 0.5}, candidate_id="integer-1")
    runtime.checkpoint({"round": 1, "observed": ["1"]})
    runtime.finish({"best_candidate_id": "integer-1"})

    assert event.sequence == 1
    assert json.loads((run_dir / "budget.json").read_text())["counters"] == {"evaluations": 1}
    assert runtime.load_checkpoint() == {"round": 1, "observed": ["1"]}
    assert json.loads((run_dir / "status.json").read_text())["status"] == "completed"
    assert json.loads((run_dir / "summary.json").read_text())["best_candidate_id"] == "integer-1"
    assert [item["event_type"] for item in runtime.events()] == [
        "campaign_started",
        "candidate_evaluated",
        "checkpoint_written",
        "campaign_finished",
    ]

    with pytest.raises(FileExistsError, match="resume=True"):
        CampaignRuntime.open(run_dir, task="integer_search")


def test_campaign_runtime_resumes_identity_budget_and_event_sequence(tmp_path: Path) -> None:
    run_dir = tmp_path / "resume"
    first = CampaignRuntime.open(
        run_dir,
        task="integer_search",
        run_id="stable-run",
        budget_limits={"evaluations": 1},
    )
    first.consume("evaluations")

    resumed = CampaignRuntime.open(
        run_dir,
        task="integer_search",
        run_id="stable-run",
        budget_limits={"evaluations": 1},
        resume=True,
    )

    assert resumed.budget.counters == {"evaluations": 1}
    assert resumed.events()[-1]["event_type"] == "campaign_resumed"
    assert resumed.events()[-1]["sequence"] == 1
    with pytest.raises(BudgetExceededError):
        resumed.consume("evaluations")
    with pytest.raises(ValueError, match="cannot resume task"):
        CampaignRuntime.open(run_dir, task="other", resume=True)


def test_unique_run_dir_preserves_existing_campaigns(tmp_path: Path) -> None:
    requested = tmp_path / "campaign"
    assert unique_run_dir(requested) == requested
    requested.mkdir()
    assert unique_run_dir(requested) == tmp_path / "campaign_2"
    (tmp_path / "campaign_2").mkdir()
    assert unique_run_dir(requested) == tmp_path / "campaign_3"


def test_bo_observation_projects_authoritative_observation_and_shared_gp_records() -> None:
    observation = make_observation("two", 2, loss=-0.2)
    feature = SurrogateVector((0.0, 1.0), "integer-v1", source_id="two")
    projected = BOObservation.from_observation(
        observation,
        objective_names=("loss",),
        feature=feature,
    )
    other = BOObservation.scalar("one", -0.4, (1.0, 0.0), feature_version="integer-v1")
    surrogate = RBFGPSurrogate([projected, other], feature_version="integer-v1")

    selected_id, prediction = select_max_ucb_record(
        [("two", (0.0, 1.0)), ("one", (1.0, 0.0))],
        surrogate,
        beta=1.0,
    )

    assert selected_id in {"one", "two"}
    assert prediction.candidate_id == selected_id
    assert len(prediction.mean) == len(prediction.std) == 1
    assert prediction.metadata["surrogate"] == "exact_rbf_gp"


def test_shared_gp_ucb_selector_consumes_engine_owned_representations() -> None:
    selector = RBFGPUCBSelector(objective_name="score", feature_version="integer-v1")
    selector.fit(
        [
            BOObservation.scalar("zero", 0.0, (0.0,), feature_version="integer-v1"),
            BOObservation.scalar("one", 1.0, (1.0,), feature_version="integer-v1"),
        ]
    )
    candidates = (
        Candidate("two", 2, "2"),
        Candidate("three", 3, "3"),
    )
    result = selector.select(
        candidates,
        {
            "two": SurrogateVector((2.0,), "integer-v1", "two"),
            "three": SurrogateVector((3.0,), "integer-v1", "three"),
        },
    )

    assert result.selected_candidate_ids[0] in {"two", "three"}
    assert {item.candidate_id for item in result.predictions} == {"two", "three"}
    assert result.metadata["surrogate"]["fit_status"] == "fitted"


def test_direct_emission_expander_separates_transport_from_task_parsing() -> None:
    client = CallableProposalClient(lambda request: '{"values": [1, 2]}')
    expander = DirectEmissionExpander(
        client=client,
        build_request=lambda request: ProposalRequest(
            messages=({"role": "user", "content": f"emit {request.reservoir_size}"},)
        ),
        parse_response=lambda response: json.loads(response.text)["values"],
        source="integer_llm",
    )

    result = expander.expand(ExpansionRequest(round_idx=0, reservoir_size=2))

    assert [item.payload for item in result.proposals] == [1, 2]
    assert {item.source for item in result.proposals} == {"integer_llm"}
    assert result.attempts[0].text == '{"values": [1, 2]}'


def integer_task_spec() -> LDMTaskSpec:
    return LDMTaskSpec(
        task="integer_search",
        candidate_domain=CandidateDomainSpec("integer", "integer", 1),
        objectives=(ObjectiveSpec("score", "maximize"),),
        response_spaces=(ResponseSpaceSpec("integers", "json"),),
        acquisition=AcquisitionSpec("reservoir_order", ("score",), "maximize", "first"),
        reservoir=ReservoirSpec(
            "integers",
            (
                ReservoirExpansionSpec(
                    "emit_integers",
                    "emit_candidate",
                    "integers",
                    True,
                ),
            ),
            "integer range",
            "integer string",
            max_size=3,
        ),
        surrogate=SurrogateSpaceSpec("none", "not used", "none"),
    )


def test_ldm_engine_runs_complete_lifecycle_and_persists_authoritative_state(
    tmp_path: Path,
) -> None:
    def expand(request: ExpansionRequest) -> ExpansionResult:
        start = request.round_idx * 2
        return ExpansionResult(
            proposals=(
                RawProposal(start, "mock"),
                RawProposal(start, "duplicate"),
                RawProposal(start + 1, "mock"),
            ),
            schema_update={"latest_round": request.round_idx},
        )

    runtime = CampaignRuntime.open(
        tmp_path / "engine-run",
        task="integer_search",
        budget_limits={"external_evaluations": 2},
    )
    engine = LDMEngine(
        task_spec=integer_task_spec(),
        expander=CallableReservoirExpander(expand),
        candidate_domain=IntegerDomain(maximum=9),
        evaluator=CallableCandidateEvaluator(
            lambda candidate: {"score": float(candidate.payload)}
        ),
        runtime=runtime,
    )

    result = engine.run(
        LDMEngineConfig(iterations=2, reservoir_size=3, evaluations_per_round=1)
    )

    assert result.stop_reason == "iteration_budget"
    assert [item.candidate.payload for item in result.state.observations] == [0, 2]
    assert result.summary["incumbent"]["candidate"]["candidate_id"] == "integer-2"
    assert result.summary["expansion_schema"] == {"latest_round": 1}
    assert runtime.load_checkpoint()["next_round"] == 2
    assert any(
        event["event_type"] == "reservoir_built"
        and event["payload"]["drop_counts"] == {"duplicate": 1}
        for event in runtime.events()
    )


def test_ldm_engine_classifies_evaluator_failures_and_stops_at_external_budget(
    tmp_path: Path,
) -> None:
    expander = CallableReservoirExpander(
        lambda request: ExpansionResult(
            proposals=(RawProposal(request.round_idx, "mock"),)
        )
    )
    runtime = CampaignRuntime.open(
        tmp_path / "limited",
        task="integer_search",
        budget_limits={"external_evaluations": 1},
    )

    def fail(candidate: Candidate):
        raise RuntimeError("domain evaluator failed")

    result = LDMEngine(
        task_spec=integer_task_spec(),
        expander=expander,
        candidate_domain=IntegerDomain(),
        evaluator=CallableCandidateEvaluator(fail),
        runtime=runtime,
    ).run(LDMEngineConfig(iterations=3, reservoir_size=1))

    assert result.stop_reason == "external_evaluation_budget"
    assert len(result.state.observations) == 1
    assert result.state.observations[0].evaluation.status == "failed"
    assert result.state.observations[0].evaluation.error == "domain evaluator failed"


def test_ldm_engine_enforces_task_contract_and_runs_encoded_selection(tmp_path: Path) -> None:
    spec = replace(integer_task_spec(), surrogate=IntegerEncoder().describe())
    runtime = CampaignRuntime.open(
        tmp_path / "selected",
        task="integer_search",
        budget_limits={"external_evaluations": 1},
    )
    engine = LDMEngine(
        task_spec=spec,
        expander=CallableReservoirExpander(
            lambda request: ExpansionResult(
                proposals=(RawProposal(1, "mock"), RawProposal(2, "mock"))
            )
        ),
        candidate_domain=IntegerDomain(),
        evaluator=CallableCandidateEvaluator(
            lambda candidate: {"score": float(candidate.payload)}
        ),
        runtime=runtime,
        selector=RBFGPUCBSelector(objective_name="score", feature_version="integer-v1"),
        surrogate_encoder=IntegerEncoder(),
    )

    result = engine.run(LDMEngineConfig(iterations=1, reservoir_size=2))

    assert result.state.observations[0].candidate.payload == 2
    selection_event = next(
        event for event in runtime.events() if event["event_type"] == "candidates_selected"
    )
    assert len(selection_event["payload"]["predictions"]) == 2

    wrong_runtime = CampaignRuntime.open(tmp_path / "wrong", task="other")
    with pytest.raises(ValueError, match="does not match"):
        LDMEngine(
            task_spec=spec,
            expander=CallableReservoirExpander(
                lambda request: ExpansionResult(proposals=(RawProposal(1, "mock"),))
            ),
            candidate_domain=IntegerDomain(),
            evaluator=CallableCandidateEvaluator(lambda candidate: {"score": 1.0}),
            runtime=wrong_runtime,
        )


def test_ldm_engine_resumes_from_shared_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "resumable-engine"
    spec = integer_task_spec()
    expander = CallableReservoirExpander(
        lambda request: ExpansionResult(
            proposals=(RawProposal(request.round_idx, "mock"),)
        )
    )
    evaluator = CallableCandidateEvaluator(
        lambda candidate: {"score": float(candidate.payload)}
    )
    first_runtime = CampaignRuntime.open(
        run_dir,
        task="integer_search",
        budget_limits={"external_evaluations": 2},
    )
    first = LDMEngine(
        task_spec=spec,
        expander=expander,
        candidate_domain=IntegerDomain(),
        evaluator=evaluator,
        runtime=first_runtime,
    ).run(LDMEngineConfig(iterations=1, reservoir_size=1))
    assert first.state.next_round == 1

    resumed_runtime = CampaignRuntime.open(
        run_dir,
        task="integer_search",
        budget_limits={"external_evaluations": 2},
        resume=True,
    )
    resumed_state = LDMEngineState.from_checkpoint(resumed_runtime.load_checkpoint())
    resumed = LDMEngine(
        task_spec=spec,
        expander=expander,
        candidate_domain=IntegerDomain(),
        evaluator=evaluator,
        runtime=resumed_runtime,
    ).run(
        LDMEngineConfig(iterations=2, reservoir_size=1),
        state=resumed_state,
    )

    assert [item.candidate.payload for item in resumed.state.observations] == [0, 1]
    assert resumed.state.next_round == 2
    assert resumed_runtime.budget.counters["external_evaluations"] == 2


def test_schema_only_expansion_does_not_count_as_an_empty_reservoir(tmp_path: Path) -> None:
    runtime = CampaignRuntime.open(
        tmp_path / "schema-only",
        task="integer_search",
        budget_limits={"external_evaluations": 1},
    )

    def expand(request: ExpansionRequest) -> ExpansionResult:
        if request.round_idx == 0:
            return ExpansionResult(schema_update={"new_parameter": {"type": "integer"}})
        return ExpansionResult(proposals=(RawProposal(1, "mock"),))

    result = LDMEngine(
        task_spec=integer_task_spec(),
        expander=CallableReservoirExpander(expand),
        candidate_domain=IntegerDomain(),
        evaluator=CallableCandidateEvaluator(lambda candidate: {"score": 1.0}),
        runtime=runtime,
    ).run(
        LDMEngineConfig(
            iterations=2,
            reservoir_size=1,
            max_empty_reservoir_rounds=1,
        )
    )

    assert result.stop_reason == "iteration_budget"
    assert len(result.state.observations) == 1
    assert result.state.expansion_schema["new_parameter"]["type"] == "integer"


def test_campaign_algorithm_enforces_an_exact_partial_final_batch(tmp_path: Path) -> None:
    recipe = CampaignRecipe(
        task_spec=integer_task_spec(),
        expander=CallableReservoirExpander(
            lambda request: ExpansionResult(
                proposals=tuple(
                    RawProposal(request.round_idx * 10 + offset, "mock")
                    for offset in range(3)
                )
            )
        ),
        candidate_domain=IntegerDomain(maximum=99),
        evaluator=CallableCandidateEvaluator(
            lambda candidate: {"score": float(candidate.payload)}
        ),
    )

    campaign = run_campaign(
        CampaignRequest(
            run_dir=tmp_path / "exact-budget",
            budget=CampaignBudget(
                rounds=2,
                reservoir_size=3,
                batch_size=2,
                target_observations=3,
                max_evaluation_attempts=3,
            ),
        ),
        recipe,
    )

    assert campaign.engine.stop_reason == "observation_target"
    assert [item.candidate.payload for item in campaign.engine.state.observations] == [
        0,
        1,
        10,
    ]
    assert campaign.runtime.budget.counters["external_evaluations"] == 3


def test_campaign_algorithm_replaces_failures_until_success_target(tmp_path: Path) -> None:
    recipe = CampaignRecipe(
        task_spec=integer_task_spec(),
        expander=CallableReservoirExpander(
            lambda _request: ExpansionResult(
                proposals=tuple(RawProposal(value, "mock") for value in range(3))
            )
        ),
        candidate_domain=IntegerDomain(),
        evaluator=CallableCandidateEvaluator(
            lambda candidate: (
                EvaluationResult(candidate.candidate_id, "failed", error="retry next")
                if candidate.payload == 0
                else {"score": float(candidate.payload)}
            )
        ),
    )

    campaign = run_campaign(
        CampaignRequest(
            run_dir=tmp_path / "replace-failures",
            budget=CampaignBudget(
                rounds=1,
                reservoir_size=3,
                batch_size=2,
                target_successful_evaluations=2,
                max_evaluation_attempts=3,
                max_evaluation_attempts_per_round=3,
                replace_failed_evaluations=True,
            ),
        ),
        recipe,
    )

    observations = campaign.engine.state.observations
    assert campaign.engine.stop_reason == "successful_evaluation_target"
    assert [item.evaluation.status for item in observations] == [
        "failed",
        "succeeded",
        "succeeded",
    ]
    assert campaign.runtime.budget.counters["external_evaluations"] == 3
    assert campaign.runtime.budget.counters["successful_evaluations"] == 2
