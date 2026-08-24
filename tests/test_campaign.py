"""Tests for the shared task-neutral campaign interface (:mod:`ldm_tts.campaign`)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from ldm_tts.campaign import (
    CampaignBudget,
    CampaignRecipe,
    CampaignRequest,
    InitializationExpander,
    InitializationOrderSelector,
    async_run_campaign,
    run_campaign,
)
from ldm_tts.contracts import (
    AcquisitionSpec,
    CallableCandidateEvaluator,
    Candidate,
    CandidateDomainSpec,
    CandidateRejection,
    EvaluationResult,
    LDMTaskSpec,
    ObjectiveSpec,
    Observation,
    RawProposal,
    ReservoirExpansionSpec,
    ReservoirSpec,
    ResponseSpaceSpec,
    SurrogateSpaceSpec,
)
from ldm_tts.engine import LDMEngineState
from ldm_tts.engine.expansion import (
    CallableReservoirExpander,
    ExpansionRequest,
    ExpansionResult,
)
from ldm_tts.optimization.records import BOObservation, BOSelectionResult


class IntegerDomain:
    def __init__(self, maximum: int = 9) -> None:
        self.maximum = maximum

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


def _recipe() -> CampaignRecipe:
    def expand(request: ExpansionRequest) -> ExpansionResult:
        return ExpansionResult(
            proposals=(RawProposal(request.round_idx, "deterministic_mock"),),
        )

    return CampaignRecipe(
        task_spec=integer_task_spec(),
        expander=CallableReservoirExpander(expand),
        candidate_domain=IntegerDomain(),
        evaluator=CallableCandidateEvaluator(
            lambda candidate: {"score": float(candidate.payload)}
        ),
    )


def _request(tmp_path: Path, budget: CampaignBudget, **overrides) -> CampaignRequest:
    values = {
        "run_dir": tmp_path / "campaign",
        "budget": budget,
        "config": {"mode": "mock"},
    }
    values.update(overrides)
    return CampaignRequest(**values)


def test_run_campaign_stops_at_observation_target_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    result = run_campaign(
        _request(
            tmp_path,
            CampaignBudget(
                rounds=3,
                reservoir_size=1,
                batch_size=1,
                target_observations=2,
            ),
        ),
        _recipe(),
    )

    assert result.engine.stop_reason == "observation_target"
    assert len(result.engine.state.observations) == 2
    assert result.engine.summary["successful_evaluation_count"] == 2
    run_dir = tmp_path / "campaign"
    for name in (
        "campaign.json",
        "events.jsonl",
        "checkpoint.json",
        "summary.json",
        "budget.json",
        "status.json",
    ):
        assert (run_dir / name).exists(), name
    limits = json.loads((run_dir / "budget.json").read_text())["limits"]
    assert limits["external_evaluations"] == 2
    assert limits["outer_iterations"] == 3


def test_run_campaign_stops_at_successful_evaluation_target(tmp_path: Path) -> None:
    def expand(request: ExpansionRequest) -> ExpansionResult:
        value = request.round_idx * 2
        return ExpansionResult(
            proposals=(RawProposal(value, "mock"), RawProposal(value + 1, "mock")),
        )

    def evaluate(candidate: Candidate):
        # Value 0 fails, others succeed.
        if candidate.payload == 0:
            return {"score": float("nan")}
        return {"score": float(candidate.payload)}

    recipe = CampaignRecipe(
        task_spec=integer_task_spec(),
        expander=CallableReservoirExpander(expand),
        candidate_domain=IntegerDomain(),
        evaluator=CallableCandidateEvaluator(evaluate),
    )
    result = run_campaign(
        _request(
            tmp_path,
            CampaignBudget(
                rounds=5,
                reservoir_size=2,
                batch_size=1,
                target_successful_evaluations=3,
                max_evaluation_attempts=5,
            ),
        ),
        recipe,
    )

    assert result.engine.stop_reason == "successful_evaluation_target"
    assert result.engine.summary["successful_evaluation_count"] == 3
    assert result.engine.summary["failed_evaluation_count"] == 1
    counters = json.loads(
        (tmp_path / "campaign" / "budget.json").read_text()
    )["counters"]
    assert counters["external_evaluations"] == 4


def test_run_campaign_resumes_from_checkpoint_and_extends_target(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "campaign"
    first = run_campaign(
        _request(
            tmp_path,
            CampaignBudget(rounds=2, reservoir_size=1, target_observations=2),
        ),
        _recipe(),
    )
    assert first.engine.stop_reason == "observation_target"
    assert len(first.engine.state.observations) == 2

    resumed = run_campaign(
        _request(
            tmp_path,
            CampaignBudget(rounds=4, reservoir_size=1, target_observations=4),
            resume=True,
        ),
        _recipe(),
    )
    assert resumed.engine.stop_reason == "observation_target"
    assert len(resumed.engine.state.observations) == 4
    events = [item["event_type"] for item in resumed.runtime.events()]
    assert "campaign_resumed" in events
    assert "campaign_finished" in events


def test_state_factory_seeds_initial_observations(tmp_path: Path) -> None:
    def state_factory(_runtime):
        candidate = Candidate("seed-7", 7, "7", source="seed")
        observation = Observation(
            candidate,
            EvaluationResult("seed-7", "succeeded", metrics={"score": 7.0}),
        )
        return LDMEngineState(observations=[observation])

    result = run_campaign(
        _request(
            tmp_path,
            CampaignBudget(rounds=1, reservoir_size=1, target_observations=2),
            state_factory=state_factory,
        ),
        _recipe(),
    )
    assert result.engine.stop_reason == "observation_target"
    assert len(result.engine.state.observations) == 2
    assert result.engine.state.observations[0].candidate.candidate_id == "seed-7"


def test_campaign_request_rejects_state_and_state_factory_together(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="both state and state_factory"):
        _request(
            tmp_path,
            CampaignBudget(rounds=1, reservoir_size=1),
            state=LDMEngineState(),
            state_factory=lambda _runtime: LDMEngineState(),
        )


def test_campaign_budget_validation() -> None:
    with pytest.raises(ValueError, match="not both"):
        CampaignBudget(
            rounds=1,
            reservoir_size=1,
            target_observations=1,
            target_successful_evaluations=1,
        )
    with pytest.raises(ValueError, match="replace_failed_evaluations"):
        CampaignBudget(rounds=1, reservoir_size=1, replace_failed_evaluations=True)
    with pytest.raises(ValueError, match="iterations must be non-negative"):
        CampaignBudget(rounds=-1, reservoir_size=1)


def test_initialization_expander_then_delegates_to_search(tmp_path: Path) -> None:
    def search_expand(request: ExpansionRequest) -> ExpansionResult:
        return ExpansionResult(
            proposals=(RawProposal(100 + request.round_idx, "search"),),
        )

    expander = InitializationExpander(
        (
            RawProposal(1, "seed"),
            RawProposal(2, "seed"),
        ),
        CallableReservoirExpander(search_expand),
        successful_target=2,
    )
    recipe = CampaignRecipe(
        task_spec=integer_task_spec(),
        expander=expander,
        candidate_domain=IntegerDomain(maximum=999),
        evaluator=CallableCandidateEvaluator(
            lambda candidate: {"score": float(candidate.payload)}
        ),
    )
    result = run_campaign(
        _request(
            tmp_path,
            CampaignBudget(
                rounds=3,
                reservoir_size=2,
                batch_size=2,
                target_observations=3,
            ),
        ),
        recipe,
    )

    assert result.engine.stop_reason == "observation_target"
    observations = result.engine.state.observations
    assert len(observations) == 3
    assert observations[0].candidate.source == "campaign_initialization"
    assert observations[1].candidate.source == "campaign_initialization"
    assert observations[2].candidate.source == "search"
    assert observations[2].candidate.payload == 101


def test_initialization_order_selector_delegates_after_target() -> None:
    class RecordingSelector:
        def __init__(self) -> None:
            self.fits = 0
            self.selects = 0

        def describe(self):
            return AcquisitionSpec("delegate", ("score",), "maximize", "delegate")

        def fit(self, history) -> None:
            self.fits += 1

        def select(self, candidates, representations, *, count=1) -> BOSelectionResult:
            self.selects += 1
            return BOSelectionResult(
                selected_candidate_ids=(candidates[-1].candidate_id,),
                metadata={"mode": "delegated"},
            )

    delegate = RecordingSelector()
    selector = InitializationOrderSelector(delegate, successful_target=1)
    candidates = (
        Candidate("first", 1, "1"),
        Candidate("second", 2, "2"),
    )

    initial = selector.select(candidates, {}, count=1)
    assert initial.selected_candidate_ids == ("first",)
    assert initial.metadata["mode"] == "initialization_order"
    assert delegate.selects == 0

    selector.fit([BOObservation.scalar("first", 1.0, (0.0,))])
    delegated = selector.select(candidates, {}, count=1)
    assert delegated.selected_candidate_ids == ("second",)
    assert delegated.metadata["mode"] == "delegated"
    assert delegate.fits == 1
    assert delegate.selects == 1


def test_runtime_hook_receives_open_runtime_before_state_factory(tmp_path: Path) -> None:
    calls: list[str] = []

    def runtime_hook(runtime):
        calls.append(f"hook:{runtime.task}")
        runtime.record("hook_ran")

    def state_factory(runtime):
        calls.append(f"factory:{len(runtime.events())}")
        return LDMEngineState()

    result = run_campaign(
        _request(
            tmp_path,
            CampaignBudget(rounds=1, reservoir_size=1, target_observations=1),
            runtime_hook=runtime_hook,
            state_factory=state_factory,
        ),
        _recipe(),
    )
    assert calls == ["hook:integer_search", "factory:2"]  # campaign_started + hook_ran
    event_types = [item["event_type"] for item in result.runtime.events()]
    assert "hook_ran" in event_types
    assert event_types.index("hook_ran") < event_types.index("reservoir_expanded")


def test_artifact_projector_receives_runtime_and_engine_result(tmp_path: Path) -> None:
    def projector(runtime, engine_result):
        return {
            "run_id": runtime.run_id,
            "stop_reason": engine_result.stop_reason,
            "observation_count": len(engine_result.state.observations),
        }

    result = run_campaign(
        _request(
            tmp_path,
            CampaignBudget(rounds=1, reservoir_size=1, target_observations=1),
            artifact_projector=projector,
        ),
        _recipe(),
    )
    assert result.projected == {
        "run_id": result.runtime.run_id,
        "stop_reason": "observation_target",
        "observation_count": 1,
    }


def test_async_run_campaign_bridge(tmp_path: Path) -> None:
    async def run():
        return await async_run_campaign(
            _request(
                tmp_path,
                CampaignBudget(rounds=1, reservoir_size=1, target_observations=1),
            ),
            _recipe(),
        )

    result = asyncio.run(run())
    assert result.engine.stop_reason == "observation_target"
    assert len(result.engine.state.observations) == 1
