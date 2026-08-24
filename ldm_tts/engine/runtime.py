"""Task-neutral Large Discovery Model campaign engine."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, MutableSequence, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar

from ldm_tts.contracts import (
    Candidate,
    CandidateDomainAdapter,
    CandidateEvaluator,
    EvaluationResult,
    LDMTaskSpec,
    ObjectiveSet,
    Observation,
    ReservoirBuilder,
)
from ldm_tts.engine.expansion import ExpansionRequest, ReservoirExpander
from ldm_tts.engine.run_store import BudgetExceededError, CampaignRuntime
from ldm_tts.optimization.records import (
    AcquisitionSelector,
    BOObservation,
    BOSelectionResult,
    SurrogateEncoder,
)


@dataclass(frozen=True)
class LDMEngineConfig:
    """Task-neutral lifecycle policy for one engine campaign."""

    iterations: int
    reservoir_size: int
    evaluations_per_round: int = 1
    max_empty_reservoir_rounds: int = 3
    target_observations: int | None = None
    target_successful_evaluations: int | None = None
    max_evaluation_attempts: int | None = None
    max_evaluation_attempts_per_round: int | None = None
    replace_failed_evaluations: bool = False

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise ValueError("engine iterations must be non-negative")
        if self.reservoir_size < 1:
            raise ValueError("engine reservoir_size must be positive")
        if self.evaluations_per_round < 1:
            raise ValueError("engine evaluations_per_round must be positive")
        if self.max_empty_reservoir_rounds < 1:
            raise ValueError("engine max_empty_reservoir_rounds must be positive")
        if self.target_observations is not None and self.target_observations < 0:
            raise ValueError("engine target_observations must be non-negative")
        if (
            self.target_successful_evaluations is not None
            and self.target_successful_evaluations < 0
        ):
            raise ValueError(
                "engine target_successful_evaluations must be non-negative"
            )
        if (
            self.target_observations is not None
            and self.target_successful_evaluations is not None
        ):
            raise ValueError(
                "engine must target observations or successful evaluations, not both"
            )
        if self.max_evaluation_attempts is not None and self.max_evaluation_attempts < 0:
            raise ValueError("engine max_evaluation_attempts must be non-negative")
        if (
            self.max_evaluation_attempts_per_round is not None
            and self.max_evaluation_attempts_per_round < 1
        ):
            raise ValueError(
                "engine max_evaluation_attempts_per_round must be positive"
            )
        if self.replace_failed_evaluations and self.target_successful_evaluations is None:
            raise ValueError(
                "replace_failed_evaluations requires target_successful_evaluations"
            )


@dataclass
class LDMEngineState:
    """Caller-resumable in-memory engine state."""

    observations: list[Observation] = field(default_factory=list)
    expansion_schema: dict[str, Any] = field(default_factory=dict)
    next_round: int = 0
    empty_reservoir_rounds: int = 0

    def __post_init__(self) -> None:
        if self.next_round < 0:
            raise ValueError("engine next_round must be non-negative")
        if self.empty_reservoir_rounds < 0:
            raise ValueError("engine empty_reservoir_rounds must be non-negative")

    def to_checkpoint(self) -> dict[str, Any]:
        return {
            "next_round": self.next_round,
            "empty_reservoir_rounds": self.empty_reservoir_rounds,
            "expansion_schema": _jsonable(self.expansion_schema),
            "observations": [_jsonable(item.to_dict()) for item in self.observations],
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> "LDMEngineState":
        observations: list[Observation] = []
        raw_observations = payload.get("observations", [])
        if not isinstance(raw_observations, list):
            raise ValueError("engine checkpoint observations must be a list")
        for raw in raw_observations:
            if not isinstance(raw, Mapping):
                raise ValueError("engine checkpoint observation must be an object")
            candidate_payload = raw.get("candidate")
            evaluation_payload = raw.get("evaluation")
            if not isinstance(candidate_payload, Mapping) or not isinstance(
                evaluation_payload, Mapping
            ):
                raise ValueError("engine checkpoint observation records are malformed")
            candidate = Candidate(
                candidate_id=str(candidate_payload.get("candidate_id", "")),
                payload=candidate_payload.get("payload"),
                canonical_key=str(candidate_payload.get("canonical_key", "")),
                source=str(candidate_payload.get("source", "")),
                metadata=dict(candidate_payload.get("metadata", {})),
            )
            evaluation = EvaluationResult(
                candidate_id=str(evaluation_payload.get("candidate_id", "")),
                status=str(evaluation_payload.get("status", "failed")),  # type: ignore[arg-type]
                metrics=dict(evaluation_payload.get("metrics", {})),
                artifacts=dict(evaluation_payload.get("artifacts", {})),
                resource_usage=dict(evaluation_payload.get("resource_usage", {})),
                error=str(evaluation_payload.get("error", "")),
                metadata=dict(evaluation_payload.get("metadata", {})),
            )
            round_idx = raw.get("round_idx")
            observations.append(
                Observation(
                    candidate=candidate,
                    evaluation=evaluation,
                    surrogate=None,
                    round_idx=None if round_idx is None else int(round_idx),
                    metadata=dict(raw.get("metadata", {})),
                )
            )
        expansion_schema = payload.get("expansion_schema", {})
        if not isinstance(expansion_schema, Mapping):
            raise ValueError("engine checkpoint expansion_schema must be an object")
        return cls(
            observations=observations,
            expansion_schema=dict(expansion_schema),
            next_round=int(payload.get("next_round", 0)),
            empty_reservoir_rounds=int(payload.get("empty_reservoir_rounds", 0)),
        )


@dataclass(frozen=True)
class LDMEngineResult:
    """Final engine state and publication-facing summary."""

    state: LDMEngineState
    rounds_run: int
    stop_reason: str
    summary: dict[str, Any]


ParentSelector = Callable[[Sequence[Observation], ObjectiveSet], Optional[Candidate]]


class LDMEngine:
    """Coordinate expansion, admission, selection, evaluation, and persistence."""

    def __init__(
        self,
        *,
        task_spec: LDMTaskSpec,
        expander: ReservoirExpander,
        candidate_domain: CandidateDomainAdapter,
        evaluator: CandidateEvaluator,
        runtime: CampaignRuntime,
        selector: AcquisitionSelector | None = None,
        surrogate_encoder: SurrogateEncoder | None = None,
        parent_selector: ParentSelector | None = None,
    ) -> None:
        if selector is None and surrogate_encoder is not None:
            raise ValueError("surrogate_encoder requires a selector")
        if runtime.task != task_spec.task:
            raise ValueError(
                f"campaign runtime task {runtime.task!r} does not match "
                f"task spec {task_spec.task!r}"
            )
        self.task_spec = task_spec
        self.expander = expander
        self.reservoir_builder = ReservoirBuilder(candidate_domain)
        self.evaluator = evaluator
        self.runtime = runtime
        self.selector = selector
        self.surrogate_encoder = surrogate_encoder
        self.parent_selector = parent_selector or _default_parent
        self.objectives = ObjectiveSet.from_specs(task_spec.objectives)
        if selector is not None:
            selector_spec = selector.describe()
            if tuple(selector_spec.objective_names) != self.objectives.names:
                raise ValueError(
                    "selector objectives do not match the task objective declaration"
                )
        if surrogate_encoder is not None:
            encoder_spec = surrogate_encoder.describe()
            if task_spec.surrogate.kind == "none":
                raise ValueError("task spec disables the surrogate used by the selector")
            comparable_fields = ("kind", "dimension_policy", "dimension", "version")
            mismatches = [
                name
                for name in comparable_fields
                if getattr(encoder_spec, name) != getattr(task_spec.surrogate, name)
            ]
            if mismatches:
                raise ValueError(
                    "surrogate encoder description does not match task spec field(s): "
                    + ", ".join(mismatches)
                )

    def run(
        self,
        config: LDMEngineConfig,
        *,
        state: LDMEngineState | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> LDMEngineResult:
        active = state or LDMEngineState()
        rounds_run = 0
        stop_reason = "iteration_budget"
        try:
            completed = _completion_reason(active, config)
            if completed is not None:
                stop_reason = completed
            for round_idx in range(active.next_round, config.iterations):
                completed = _completion_reason(active, config)
                if completed is not None:
                    stop_reason = completed
                    break
                remaining_attempts = _remaining_evaluation_attempts(
                    self.runtime, active, config
                )
                if remaining_attempts == 0:
                    stop_reason = "evaluation_attempt_budget"
                    break
                self.runtime.consume("outer_iterations")
                self.runtime.status.update(
                    "running",
                    phase="reservoir_expansion",
                    iteration=round_idx,
                    budget=self.runtime.budget,
                )
                parent = self.parent_selector(active.observations, self.objectives)
                expansion = self.expander.expand(
                    ExpansionRequest(
                        round_idx=round_idx,
                        reservoir_size=config.reservoir_size,
                        observations=tuple(active.observations),
                        parent=parent,
                        expansion_schema=dict(active.expansion_schema),
                        context=dict(context or {}),
                    )
                )
                if expansion.schema_update is not None:
                    active.expansion_schema.update(expansion.schema_update)
                self.runtime.record(
                    "reservoir_expanded",
                    {
                        "proposal_count": len(expansion.proposals),
                        "attempts": [item.to_dict() for item in expansion.attempts],
                        "schema_update": expansion.schema_update,
                        "metadata": expansion.metadata,
                    },
                    iteration=round_idx,
                )
                if expansion.attempts:
                    self.runtime.consume("proposal_attempts", len(expansion.attempts))

                reservoir_limit = config.reservoir_size
                if self.task_spec.reservoir.max_size is not None:
                    reservoir_limit = min(reservoir_limit, self.task_spec.reservoir.max_size)
                reservoir = self.reservoir_builder.build(
                    expansion.proposals,
                    evaluated_keys=(item.canonical_key for item in active.observations),
                    max_size=reservoir_limit,
                    metadata={"round_idx": round_idx},
                )
                self.runtime.record(
                    "reservoir_built",
                    {
                        "candidate_ids": [item.candidate_id for item in reservoir.candidates],
                        "drop_counts": reservoir.drop_counts,
                        "rejections": [item.to_dict() for item in reservoir.rejections],
                    },
                    iteration=round_idx,
                )
                if reservoir.candidates:
                    self.runtime.consume(
                        "valid_search_candidates", len(reservoir.candidates)
                    )

                if not reservoir.candidates:
                    schema_only = expansion.schema_update is not None and not expansion.proposals
                    active.empty_reservoir_rounds = (
                        0 if schema_only else active.empty_reservoir_rounds + 1
                    )
                    active.next_round = round_idx + 1
                    rounds_run += 1
                    self._checkpoint(active)
                    if (
                        not schema_only
                        and active.empty_reservoir_rounds
                        >= config.max_empty_reservoir_rounds
                    ):
                        stop_reason = "empty_reservoir_limit"
                        break
                    continue

                active.empty_reservoir_rounds = 0
                desired = _desired_round_results(active, config)
                selection_count = desired
                if config.replace_failed_evaluations:
                    selection_count = (
                        config.max_evaluation_attempts_per_round
                        or len(reservoir.candidates)
                    )
                selection_count = min(selection_count, len(reservoir.candidates))
                if remaining_attempts is not None:
                    selection_count = min(selection_count, remaining_attempts)
                selection = self._select(
                    active.observations,
                    reservoir.candidates,
                    selection_count,
                )
                selected = self._resolve_selection(reservoir.candidates, selection)
                self.runtime.record(
                    "candidates_selected",
                    selection.to_dict(),
                    iteration=round_idx,
                )
                if not selected:
                    stop_reason = "empty_selection"
                    active.next_round = round_idx + 1
                    rounds_run += 1
                    self._checkpoint(active)
                    break

                budget_exhausted = False
                round_observations = 0
                round_successes = 0
                for candidate in selected:
                    if config.target_successful_evaluations is not None:
                        if round_successes >= desired:
                            break
                    elif round_observations >= desired:
                        break
                    try:
                        self.runtime.consume_many(
                            {
                                "selected_candidates": 1,
                                "external_evaluations": 1,
                                "expensive_evaluation_attempts": 1,
                            }
                        )
                    except BudgetExceededError:
                        budget_exhausted = True
                        stop_reason = "external_evaluation_budget"
                        break
                    evaluation = self._evaluate(candidate)
                    benchmark_jobs = evaluation.resource_usage.get("benchmark_jobs", 0)
                    if benchmark_jobs:
                        self.runtime.consume("benchmark_jobs", benchmark_jobs)
                    if evaluation.succeeded:
                        self.runtime.consume("successful_evaluations")
                    representation = (
                        self.surrogate_encoder.encode(candidate)
                        if self.surrogate_encoder is not None and evaluation.succeeded
                        else None
                    )
                    observation = Observation(
                        candidate=candidate,
                        evaluation=evaluation,
                        surrogate=representation,
                        round_idx=round_idx,
                    )
                    active.observations.append(observation)
                    round_observations += 1
                    if evaluation.succeeded:
                        round_successes += 1
                    self.runtime.record(
                        "candidate_evaluated",
                        observation.to_dict(),
                        iteration=round_idx,
                        candidate_id=candidate.candidate_id,
                    )

                active.next_round = round_idx + 1
                rounds_run += 1
                self._checkpoint(active)
                if budget_exhausted:
                    break
                completed = _completion_reason(active, config)
                if completed is not None:
                    stop_reason = completed
                    break

            summary = self._summary(active, rounds_run, stop_reason)
            terminal = (
                "completed"
                if stop_reason
                in {
                    "iteration_budget",
                    "observation_target",
                    "successful_evaluation_target",
                }
                else "stopped"
            )
            self.runtime.finish(summary, status=terminal)
            return LDMEngineResult(active, rounds_run, stop_reason, summary)
        except Exception as exc:
            self.runtime.fail(exc)
            raise

    def _select(
        self,
        observations: Sequence[Observation],
        candidates: Sequence[Candidate],
        count: int,
    ) -> BOSelectionResult:
        if self.selector is None:
            return BOSelectionResult(
                selected_candidate_ids=tuple(item.candidate_id for item in candidates[:count]),
                metadata={"mode": "reservoir_order"},
            )
        history = [
            BOObservation.from_observation(
                observation,
                objective_names=self.objectives.names,
                feature=(
                    observation.surrogate
                    if observation.surrogate is not None
                    else (
                        self.surrogate_encoder.encode(observation.candidate)
                        if self.surrogate_encoder is not None
                        else None
                    )
                ),
            )
            for observation in observations
            if observation.evaluation.succeeded
        ]
        self.selector.fit(history)
        representations = (
            {
                candidate.candidate_id: self.surrogate_encoder.encode(candidate)
                for candidate in candidates
            }
            if self.surrogate_encoder is not None
            else {}
        )
        return self.selector.select(candidates, representations, count=count)

    def _resolve_selection(
        self,
        candidates: Sequence[Candidate],
        selection: BOSelectionResult,
    ) -> list[Candidate]:
        by_id = {item.candidate_id: item for item in candidates}
        unknown = [item for item in selection.selected_candidate_ids if item not in by_id]
        if unknown:
            raise ValueError(
                "selector returned candidate ids outside the active reservoir: "
                + ", ".join(unknown)
            )
        return [by_id[item] for item in selection.selected_candidate_ids]

    def _evaluate(self, candidate: Candidate) -> EvaluationResult:
        try:
            result = self.evaluator.evaluate(candidate)
            if result.candidate_id != candidate.candidate_id:
                raise ValueError("evaluator returned a mismatched candidate_id")
            return self.objectives.validate_result(result)
        except TimeoutError as exc:
            return EvaluationResult(candidate.candidate_id, "timed_out", error=str(exc))
        except Exception as exc:
            return EvaluationResult(candidate.candidate_id, "failed", error=str(exc))

    def _checkpoint(self, state: LDMEngineState) -> None:
        self.runtime.checkpoint(state.to_checkpoint())

    def _summary(
        self,
        state: LDMEngineState,
        rounds_run: int,
        stop_reason: str,
    ) -> dict[str, Any]:
        successful = [item for item in state.observations if item.evaluation.succeeded]
        summary: dict[str, Any] = {
            "task": self.task_spec.task,
            "rounds_run": rounds_run,
            "next_round": state.next_round,
            "observation_count": len(state.observations),
            "successful_evaluation_count": len(successful),
            "failed_evaluation_count": len(state.observations) - len(successful),
            "stop_reason": stop_reason,
            "expansion_schema": _jsonable(state.expansion_schema),
        }
        if len(self.objectives.specs) == 1:
            incumbent = self.objectives.incumbent(state.observations)
            summary["incumbent"] = None if incumbent is None else _jsonable(incumbent.to_dict())
        else:
            summary["pareto_candidate_ids"] = [
                item.candidate_id for item in self.objectives.pareto_front(state.observations)
            ]
        return summary


def _default_parent(
    observations: Sequence[Observation], objectives: ObjectiveSet
) -> Candidate | None:
    if not observations:
        return None
    if len(objectives.specs) == 1:
        incumbent = objectives.incumbent(observations)
        return None if incumbent is None else incumbent.candidate
    front = objectives.pareto_front(observations)
    return front[0].candidate if front else None


def _successful_evaluation_count(state: LDMEngineState) -> int:
    return sum(item.evaluation.succeeded for item in state.observations)


def _completion_reason(
    state: LDMEngineState,
    config: LDMEngineConfig,
) -> str | None:
    if (
        config.target_observations is not None
        and len(state.observations) >= config.target_observations
    ):
        return "observation_target"
    if (
        config.target_successful_evaluations is not None
        and _successful_evaluation_count(state)
        >= config.target_successful_evaluations
    ):
        return "successful_evaluation_target"
    return None


def _desired_round_results(
    state: LDMEngineState,
    config: LDMEngineConfig,
) -> int:
    if config.target_observations is not None:
        remaining = max(0, config.target_observations - len(state.observations))
        return min(config.evaluations_per_round, remaining)
    if config.target_successful_evaluations is not None:
        remaining = max(
            0,
            config.target_successful_evaluations
            - _successful_evaluation_count(state),
        )
        return min(config.evaluations_per_round, remaining)
    return config.evaluations_per_round


def _remaining_evaluation_attempts(
    runtime: CampaignRuntime,
    state: LDMEngineState,
    config: LDMEngineConfig,
) -> int | None:
    if config.max_evaluation_attempts is None:
        return None
    consumed = int(
        runtime.budget.counters.get(
            "external_evaluations",
            len(state.observations),
        )
    )
    return max(0, config.max_evaluation_attempts - consumed)


def _jsonable(value: Any) -> Any:
    """Normalize task metadata for durable engine artifacts."""

    return json.loads(json.dumps(value, default=str))


HistoryItem = TypeVar("HistoryItem")


@dataclass
class LDMSearchRoundResult(Generic[HistoryItem]):
    """Compatibility result emitted by one task-specific search round."""

    history_delta: Sequence[HistoryItem] = field(default_factory=tuple)
    record: dict[str, Any] | None = None
    empty_reservoir: bool = False
    stop_reason: str | None = None


@dataclass
class LDMSearchLoopResult(Generic[HistoryItem]):
    """Compatibility result from the legacy budget-controlled search loop."""

    history: list[HistoryItem]
    rounds_run: int
    early_stop_reason: str | None = None
    empty_reservoir_rounds: int = 0


def run_budgeted_search(
    history: MutableSequence[HistoryItem],
    *,
    budget: int,
    build_round: Callable[[int, list[HistoryItem]], LDMSearchRoundResult[HistoryItem]],
    record_round: Callable[[dict[str, Any]], None] | None = None,
    on_empty_reservoir: Callable[[int, int], None] | None = None,
    start_round: int = 0,
    max_empty_reservoir_rounds: int = 10,
    allow_early_stop: bool = True,
) -> LDMSearchLoopResult[HistoryItem]:
    """Run the legacy task-owned round adapter under shared budget policy."""

    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")
    if start_round < 0:
        raise ValueError(f"start_round must be non-negative, got {start_round}")
    if max_empty_reservoir_rounds < 1:
        raise ValueError(
            "max_empty_reservoir_rounds must be >= 1, "
            f"got {max_empty_reservoir_rounds}"
        )

    current_history = list(history)
    round_idx = int(start_round)
    rounds_run = 0
    empty_reservoir_rounds = 0
    early_stop_reason: str | None = None

    while len(current_history) < budget:
        result = build_round(round_idx, list(current_history))
        if result.record is not None and record_round is not None:
            record_round(result.record)
        rounds_run += 1

        if result.stop_reason:
            early_stop_reason = result.stop_reason
            break
        if result.empty_reservoir:
            empty_reservoir_rounds += 1
            if on_empty_reservoir is not None:
                on_empty_reservoir(round_idx, empty_reservoir_rounds)
            if empty_reservoir_rounds >= max_empty_reservoir_rounds and allow_early_stop:
                early_stop_reason = "empty_reservoir_limit"
                break
            round_idx += 1
            continue
        if not result.history_delta:
            if allow_early_stop:
                early_stop_reason = "empty_selection"
                break
            round_idx += 1
            continue

        current_history.extend(result.history_delta)
        empty_reservoir_rounds = 0
        round_idx += 1

    history[:] = current_history
    return LDMSearchLoopResult(
        history=current_history,
        rounds_run=rounds_run,
        early_stop_reason=early_stop_reason,
        empty_reservoir_rounds=empty_reservoir_rounds,
    )


__all__ = [
    "LDMEngine",
    "LDMEngineConfig",
    "LDMEngineResult",
    "LDMEngineState",
    "LDMSearchLoopResult",
    "LDMSearchRoundResult",
    "ParentSelector",
    "run_budgeted_search",
]
