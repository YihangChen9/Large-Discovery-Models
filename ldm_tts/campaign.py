"""Deep, task-neutral interface for running an LDM campaign.

Task packages provide scientific adapters in :class:`CampaignRecipe`; this
module owns runtime creation, absolute budgets, checkpoint restore, the shared
engine lifecycle, and optional legacy artifact projection.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ldm_tts.contracts import (
    AcquisitionSpec,
    Candidate,
    CandidateDomainAdapter,
    CandidateEvaluator,
    LDMTaskSpec,
    RawProposal,
)
from ldm_tts.engine import (
    LDMEngine,
    LDMEngineConfig,
    LDMEngineResult,
    LDMEngineState,
    ParentSelector,
)
from ldm_tts.engine.expansion import ExpansionRequest, ExpansionResult, ReservoirExpander
from ldm_tts.engine.run_store import CampaignRuntime
from ldm_tts.optimization.records import (
    AcquisitionSelector,
    BOObservation,
    BOSelectionResult,
    SurrogateEncoder,
    SurrogateVector,
)


class InitializationExpander:
    """Emit a fixed initialization reservoir before delegating to search."""

    def __init__(
        self,
        initializer: Sequence[RawProposal],
        search: ReservoirExpander,
        *,
        successful_target: int,
        source: str = "campaign_initialization",
    ) -> None:
        self.initializer = tuple(
            RawProposal(item.payload, source, metadata=dict(item.metadata))
            for item in initializer
        )
        self.search = search
        self.successful_target = max(0, int(successful_target))
        self.source = source

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        completed = sum(
            observation.evaluation.succeeded
            and observation.candidate.source == self.source
            for observation in request.observations
        )
        if completed < self.successful_target:
            return ExpansionResult(
                proposals=self.initializer,
                metadata={
                    "phase": "initialization",
                    "successful_target": self.successful_target,
                    "successful_completed": completed,
                },
            )
        return self.search.expand(request)


class InitializationOrderSelector:
    """Use reservoir order during initialization, then delegate selection."""

    def __init__(
        self,
        selector: AcquisitionSelector,
        *,
        successful_target: int,
    ) -> None:
        self.selector = selector
        self.successful_target = max(0, int(successful_target))
        self.history_size = 0

    def describe(self) -> AcquisitionSpec:
        return self.selector.describe()

    def fit(self, history: Sequence[BOObservation]) -> None:
        self.history_size = len(history)
        self.selector.fit(history)

    def select(
        self,
        candidates: Sequence[Candidate],
        representations: Mapping[str, SurrogateVector],
        *,
        count: int = 1,
    ) -> BOSelectionResult:
        if self.history_size < self.successful_target:
            return BOSelectionResult(
                selected_candidate_ids=tuple(
                    candidate.candidate_id for candidate in candidates[:count]
                ),
                metadata={"mode": "initialization_order"},
            )
        return self.selector.select(candidates, representations, count=count)


@dataclass(frozen=True)
class CampaignBudget:
    """Absolute completion and safety limits for one campaign.

    Exactly one of ``target_observations`` and
    ``target_successful_evaluations`` may be set. Successful-result campaigns
    can replace failed evaluations from the selected reservoir without
    consuming their scientific result target.
    """

    rounds: int
    reservoir_size: int
    batch_size: int = 1
    target_observations: int | None = None
    target_successful_evaluations: int | None = None
    max_evaluation_attempts: int | None = None
    max_evaluation_attempts_per_round: int | None = None
    replace_failed_evaluations: bool = False
    max_empty_reservoir_rounds: int = 3
    extra_limits: Mapping[str, int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Keep validation and error messages aligned with the engine's policy.
        self.engine_config()

    def engine_config(self) -> LDMEngineConfig:
        return LDMEngineConfig(
            iterations=int(self.rounds),
            reservoir_size=int(self.reservoir_size),
            evaluations_per_round=int(self.batch_size),
            max_empty_reservoir_rounds=int(self.max_empty_reservoir_rounds),
            target_observations=self.target_observations,
            target_successful_evaluations=self.target_successful_evaluations,
            max_evaluation_attempts=self.max_evaluation_attempts,
            max_evaluation_attempts_per_round=self.max_evaluation_attempts_per_round,
            replace_failed_evaluations=bool(self.replace_failed_evaluations),
        )

    def runtime_limits(self) -> dict[str, int | float]:
        attempt_limit = self.max_evaluation_attempts
        if attempt_limit is None:
            attempt_limit = self.target_observations
        if attempt_limit is None and self.target_successful_evaluations is not None:
            attempt_limit = self.target_successful_evaluations
        if attempt_limit is None:
            attempt_limit = self.rounds * self.batch_size
        success_limit = (
            self.target_successful_evaluations
            if self.target_successful_evaluations is not None
            else attempt_limit
        )
        limits: dict[str, int | float] = {
            "outer_iterations": self.rounds,
            "valid_search_candidates": self.rounds * self.reservoir_size,
            "selected_candidates": attempt_limit,
            "external_evaluations": attempt_limit,
            "expensive_evaluation_attempts": attempt_limit,
            "successful_evaluations": success_limit,
            "benchmark_jobs": attempt_limit,
        }
        limits.update({str(key): value for key, value in self.extra_limits.items()})
        return limits


@dataclass(frozen=True)
class CampaignRecipe:
    """Scientific behavior that varies at the shared campaign seam."""

    task_spec: LDMTaskSpec
    expander: ReservoirExpander
    candidate_domain: CandidateDomainAdapter
    evaluator: CandidateEvaluator
    selector: AcquisitionSelector | None = None
    surrogate_encoder: SurrogateEncoder | None = None
    parent_selector: ParentSelector | None = None


StateFactory = Callable[[CampaignRuntime], LDMEngineState]
ArtifactProjector = Callable[[CampaignRuntime, LDMEngineResult], Any]


@dataclass(frozen=True)
class CampaignRequest:
    """Run identity and lifecycle configuration for :func:`run_campaign`."""

    run_dir: Path
    budget: CampaignBudget
    config: Mapping[str, Any] = field(default_factory=dict)
    resume: bool = False
    state: LDMEngineState | None = None
    state_factory: StateFactory | None = None
    context: Mapping[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    contract_snapshot: Mapping[str, Any] | None = None
    contract_sha256: str = ""
    contract_profile: str = ""
    artifact_projector: ArtifactProjector | None = None
    runtime_hook: Callable[[CampaignRuntime], None] | None = None

    def __post_init__(self) -> None:
        if self.state is not None and self.state_factory is not None:
            raise ValueError("campaign request cannot set both state and state_factory")


@dataclass(frozen=True)
class CampaignResult:
    """Engine result, durable runtime, and optional projected task artifacts."""

    engine: LDMEngineResult
    runtime: CampaignRuntime
    projected: Any = None


def run_campaign(request: CampaignRequest, recipe: CampaignRecipe) -> CampaignResult:
    """Run one complete campaign through the shared LDM algorithm."""

    runtime = CampaignRuntime.open(
        Path(request.run_dir),
        task=recipe.task_spec.task,
        run_id=request.run_id,
        config=dict(request.config),
        task_spec=recipe.task_spec,
        # Resumed campaigns may intentionally extend absolute limits. Load the
        # persisted ledger first, then apply the new recipe below.
        budget_limits=(None if request.resume else request.budget.runtime_limits()),
        contract_snapshot=request.contract_snapshot,
        contract_sha256=request.contract_sha256,
        contract_profile=request.contract_profile,
        resume=bool(request.resume),
    )
    # A resumed runtime loads its previous ledger. The recipe is authoritative
    # for the new absolute limits, allowing a caller to extend a campaign while
    # preserving already-consumed counters.
    runtime.budget.limits = request.budget.runtime_limits()
    runtime.budget.write()

    if request.runtime_hook is not None:
        request.runtime_hook(runtime)

    state = request.state
    if state is None and request.state_factory is not None:
        state = request.state_factory(runtime)
    if state is None and request.resume:
        checkpoint = runtime.load_checkpoint()
        if checkpoint is not None:
            state = LDMEngineState.from_checkpoint(checkpoint)
    state = state or LDMEngineState()

    engine = LDMEngine(
        task_spec=recipe.task_spec,
        expander=recipe.expander,
        candidate_domain=recipe.candidate_domain,
        evaluator=recipe.evaluator,
        runtime=runtime,
        selector=recipe.selector,
        surrogate_encoder=recipe.surrogate_encoder,
        parent_selector=recipe.parent_selector,
    )
    engine_result = engine.run(
        request.budget.engine_config(),
        state=state,
        context=request.context,
    )
    projected = (
        request.artifact_projector(runtime, engine_result)
        if request.artifact_projector is not None
        else None
    )
    return CampaignResult(engine=engine_result, runtime=runtime, projected=projected)


async def async_run_campaign(
    request: CampaignRequest,
    recipe: CampaignRecipe,
) -> CampaignResult:
    """Async host bridge for the synchronous shared implementation.

    The whole campaign runs in one worker thread, so task adapters with local
    event loops (such as NanoGPT proposal traversal) remain isolated from the
    caller's running loop.
    """

    return await asyncio.to_thread(run_campaign, request, recipe)


__all__ = [
    "ArtifactProjector",
    "CampaignBudget",
    "CampaignRecipe",
    "CampaignRequest",
    "CampaignResult",
    "InitializationExpander",
    "InitializationOrderSelector",
    "StateFactory",
    "async_run_campaign",
    "run_campaign",
]
