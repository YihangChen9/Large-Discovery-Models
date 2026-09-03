"""Task-neutral LDM environment for reinforcement learning.

One LDM campaign becomes one RL episode. One environment step mirrors one
``LDMEngine`` round:

1. the policy emits raw candidate-proposal text (the action);
2. the task-declared parser turns it into payloads;
3. the task's ``CandidateDomainAdapter`` + the shared ``ReservoirBuilder``
   admit, deduplicate (against history and within the round) and cap the
   reservoir;
4. up to ``evaluations_per_round`` candidates are evaluated (reservoir order,
   matching the engine default) with engine-identical error wrapping;
5. the reward is the objective improvement over the previous incumbent, and
   the observation is a rendered feedback transcript.

The environment depends only on ``ldm_tts`` contracts — never on Slime or a
specific task. Task wiring happens in ``ldm_rl.factories``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ldm_tts.contracts import (
    Candidate,
    CandidateDomainAdapter,
    CandidateEvaluator,
    EvaluationResult,
    LDMTaskSpec,
    ObjectiveSet,
    Observation,
    RawProposal,
    ReservoirBuilder,
    ResponseSpaceSpec,
)
from ldm_tts.engine import LDMEngineState
from ldm_tts.optimization.records import (
    AcquisitionSelector,
    BOObservation,
    BOSelectionResult,
    SurrogateEncoder,
)

from ldm_rl.parsing import call_text_parser, load_declared_parser
from ldm_rl.prompts import render_reset_observation, render_step_observation

REWARD_POLICIES = ("improvement", "raw", "binary", "acquisition", "hypervolume")
ACQUISITION_AGGS = ("max", "mean")


@dataclass(frozen=True)
class EnvConfig:
    """Lifecycle and reward policy for one RL episode."""

    iterations: int
    reservoir_size: int = 1
    evaluations_per_round: int = 1
    max_empty_reservoir_rounds: int = 3
    target_observations: int | None = None
    target_successful_evaluations: int | None = None
    max_evaluation_attempts: int | None = None
    max_evaluation_attempts_per_round: int | None = None
    replace_failed_evaluations: bool = False
    reward: str = "improvement"
    reward_failure: float = 0.0
    reward_invalid: float = 0.0
    acquisition_agg: str = "max"
    reward_ref_point: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise ValueError("env iterations must be non-negative")
        if self.reservoir_size < 1:
            raise ValueError("env reservoir_size must be positive")
        if self.evaluations_per_round < 1:
            raise ValueError("env evaluations_per_round must be positive")
        if self.max_empty_reservoir_rounds < 1:
            raise ValueError("env max_empty_reservoir_rounds must be positive")
        if self.target_observations is not None and self.target_observations < 0:
            raise ValueError("env target_observations must be non-negative")
        if (
            self.target_successful_evaluations is not None
            and self.target_successful_evaluations < 0
        ):
            raise ValueError(
                "env target_successful_evaluations must be non-negative"
            )
        if (
            self.target_observations is not None
            and self.target_successful_evaluations is not None
        ):
            raise ValueError(
                "env must target observations or successful evaluations, not both"
            )
        if self.max_evaluation_attempts is not None and self.max_evaluation_attempts < 0:
            raise ValueError("env max_evaluation_attempts must be non-negative")
        if (
            self.max_evaluation_attempts_per_round is not None
            and self.max_evaluation_attempts_per_round < 1
        ):
            raise ValueError(
                "env max_evaluation_attempts_per_round must be positive"
            )
        if self.replace_failed_evaluations and self.target_successful_evaluations is None:
            raise ValueError(
                "env replace_failed_evaluations requires target_successful_evaluations"
            )
        if self.reward not in REWARD_POLICIES:
            raise ValueError(
                f"unknown reward policy {self.reward!r}; expected one of {REWARD_POLICIES}"
            )
        if self.acquisition_agg not in ACQUISITION_AGGS:
            raise ValueError(
                f"unknown acquisition_agg {self.acquisition_agg!r}; "
                f"expected one of {ACQUISITION_AGGS}"
            )
        if self.reward == "hypervolume" and self.reward_ref_point is None:
            raise ValueError(
                "reward 'hypervolume' requires a fixed reward_ref_point "
                "(oriented-space domain nadir); the moving nadir is disabled (PR #2)."
            )


@dataclass(frozen=True)
class EnvStep:
    """Outcome of one environment transition."""

    observation: str
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass
class EpisodeResult:
    """Summary of one finished episode."""

    steps: list[EnvStep]
    history: tuple[Observation, ...] = ()
    total_reward: float = 0.0
    best_metrics: dict[str, float] | None = None
    stop_reason: str = ""
    rounds: int = 0


class LDMEnv:
    """Reset/step environment over one task's campaign adapters."""

    def __init__(
        self,
        *,
        task_spec: LDMTaskSpec,
        domain: CandidateDomainAdapter,
        evaluator: CandidateEvaluator,
        config: EnvConfig | None = None,
        objectives: ObjectiveSet | None = None,
        parse_action: Callable[[str], list[Any]] | None = None,
        parser_parameters: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
        selector: AcquisitionSelector | None = None,
        surrogate_encoder: SurrogateEncoder | None = None,
    ) -> None:
        self.task_spec = task_spec
        self.config = config or EnvConfig(iterations=1)
        self.objectives = objectives or ObjectiveSet.from_specs(task_spec.objectives)
        self.context = dict(context or {})
        self._builder = ReservoirBuilder(domain)
        self._evaluator = evaluator
        self._parser_parameters = dict(parser_parameters or {})
        self._response_space = self._resolve_response_space()
        self._parse_action = parse_action or self._default_parse_action()
        self._state = LDMEngineState()
        self._done = False
        if (selector is None) != (surrogate_encoder is None):
            raise ValueError("selector and surrogate_encoder must be configured together")
        self.selector = selector
        self.surrogate_encoder = surrogate_encoder
        self._validate_optimizer_config()
        if self.config.reward == "acquisition" and self.selector is None:
            raise ValueError(
                "reward policy 'acquisition' requires a selector and surrogate_encoder"
            )

    # ------------------------------------------------------------------ setup

    def _validate_optimizer_config(self) -> None:
        """Mirror the engine's selector/encoder consistency checks."""

        if self.selector is None or self.surrogate_encoder is None:
            return
        selector_spec = self.selector.describe()
        if tuple(selector_spec.objective_names) != self.objectives.names:
            raise ValueError(
                "selector objectives do not match the task objective declaration"
            )
        encoder_spec = self.surrogate_encoder.describe()
        if self.task_spec.surrogate.kind == "none":
            raise ValueError("task spec disables the surrogate used by the selector")
        comparable_fields = ("kind", "dimension_policy", "dimension", "version")
        mismatches = [
            name
            for name in comparable_fields
            if getattr(encoder_spec, name) != getattr(self.task_spec.surrogate, name)
        ]
        if mismatches:
            raise ValueError(
                "surrogate encoder description does not match task spec field(s): "
                + ", ".join(mismatches)
            )

    def _resolve_response_space(self) -> ResponseSpaceSpec:
        producing = [
            expansion
            for expansion in self.task_spec.reservoir.expansions
            if expansion.produces_candidates
        ]
        names = sorted({expansion.response_space for expansion in producing})
        for space in self.task_spec.response_spaces:
            if space.name in names:
                return space
        if self.task_spec.response_spaces:
            return self.task_spec.response_spaces[0]
        raise ValueError(
            f"task {self.task_spec.task!r} declares no response space for parsing"
        )

    def _default_parse_action(self) -> Callable[[str], list[Any]]:
        parser = load_declared_parser(self._response_space)
        if parser is None:
            raise ValueError(
                f"task {self.task_spec.task!r} declares no parser for response "
                f"space {self._response_space.name!r}; supply parse_action explicitly"
            )

        def parse(text: str) -> list[Any]:
            return call_text_parser(
                parser,
                text,
                expected_count=self.config.reservoir_size,
                parameters=self._parser_parameters,
            )

        return parse

    # -------------------------------------------------------------- lifecycle

    @property
    def history(self) -> tuple[Observation, ...]:
        return tuple(self._state.observations)

    @property
    def next_round(self) -> int:
        return self._state.next_round

    @property
    def incumbent(self) -> Observation | None:
        if len(self.objectives.specs) != 1:
            return None
        return self.objectives.incumbent(self._state.observations)

    def reset(self) -> str:
        """Start a fresh episode and return the initial observation."""

        self._state = LDMEngineState()
        self._done = False
        return render_reset_observation(
            self.task_spec,
            reservoir_size=self.config.reservoir_size,
            context=self.context,
        )

    def _check_done(self) -> None:
        if self._done:
            raise RuntimeError("step() called after the episode finished")

    # ------------------------------------------------------- budget semantics
    #
    # These mirror ldm_tts.engine.runtime's budget helpers so an RL episode
    # stops and counts evaluations the same way a campaign does. The only
    # structural difference is that the env has no CampaignRuntime ledger:
    # ``max_evaluation_attempts`` consumption is derived from the observation
    # count (one observation per evaluation attempt, success or failure).

    def _successful_evaluation_count(self) -> int:
        return sum(item.evaluation.succeeded for item in self._state.observations)

    def _completion_reason(self) -> str | None:
        if (
            self.config.target_observations is not None
            and len(self._state.observations) >= self.config.target_observations
        ):
            return "observation_target"
        if (
            self.config.target_successful_evaluations is not None
            and self._successful_evaluation_count()
            >= self.config.target_successful_evaluations
        ):
            return "successful_evaluation_target"
        return None

    def _desired_round_results(self) -> int:
        if self.config.target_observations is not None:
            remaining = max(
                0, self.config.target_observations - len(self._state.observations)
            )
            return min(self.config.evaluations_per_round, remaining)
        if self.config.target_successful_evaluations is not None:
            remaining = max(
                0,
                self.config.target_successful_evaluations
                - self._successful_evaluation_count(),
            )
            return min(self.config.evaluations_per_round, remaining)
        return self.config.evaluations_per_round

    def _remaining_evaluation_attempts(self) -> int | None:
        if self.config.max_evaluation_attempts is None:
            return None
        return max(0, self.config.max_evaluation_attempts - len(self._state.observations))

    # ------------------------------------------------------------------ step

    def step(self, action_text: str) -> EnvStep:
        """Execute one engine round driven by the policy action text.

        Mirrors ``LDMEngine.run``'s per-round flow: loop-top completion and
        attempt-budget guards, reservoir build, target-aware selection count,
        failed-evaluation replacement, and a post-round completion check.
        """

        self._check_done()
        round_idx = self._state.next_round
        remaining = self.config.iterations - round_idx - 1
        baseline = self._componentwise_best(self._state.observations)
        terminated = False
        stop_reason = ""
        remaining_attempts: int | None = None
        parse_error: str | None = None
        payloads: list[Any] = []
        rejections: list[Any] = []
        new_observations: list[Observation] = []
        selection: BOSelectionResult | None = None

        completed = self._completion_reason()
        if completed is not None:
            terminated = True
            stop_reason = completed
        else:
            remaining_attempts = self._remaining_evaluation_attempts()
            if remaining_attempts == 0:
                terminated = True
                stop_reason = "evaluation_attempt_budget"

        if not terminated:
            try:
                payloads = self._parse_action(action_text)
            except Exception as exc:  # noqa: BLE001 - parser errors are env feedback
                payloads = []
                parse_error = str(exc)

            proposals = tuple(
                RawProposal(
                    payload,
                    source="policy",
                    metadata={"round_idx": round_idx, "policy": True},
                )
                for payload in payloads
            )
            reservoir_limit = self.config.reservoir_size
            if self.task_spec.reservoir.max_size is not None:
                reservoir_limit = min(reservoir_limit, self.task_spec.reservoir.max_size)
            build = self._builder.build(
                proposals,
                evaluated_keys=(
                    item.canonical_key for item in self._state.observations
                ),
                max_size=reservoir_limit,
                metadata={"round_idx": round_idx},
            )
            rejections = build.rejections

            if build.candidates:
                self._state.empty_reservoir_rounds = 0
                desired = self._desired_round_results()
                selection_count = desired
                if self.config.replace_failed_evaluations:
                    selection_count = (
                        self.config.max_evaluation_attempts_per_round
                        or len(build.candidates)
                    )
                selection_count = min(selection_count, len(build.candidates))
                if remaining_attempts is not None:
                    selection_count = min(selection_count, remaining_attempts)
                selection = self._select(build.candidates, selection_count)
                selected = self._resolve_selection(build.candidates, selection)
                round_observations = 0
                round_successes = 0
                for candidate in selected:
                    if self.config.target_successful_evaluations is not None:
                        if round_successes >= desired:
                            break
                    elif round_observations >= desired:
                        break
                    evaluation = self._evaluate(candidate)
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
                    self._state.observations.append(observation)
                    new_observations.append(observation)
                    round_observations += 1
                    if evaluation.succeeded:
                        round_successes += 1
                completed = self._completion_reason()
                if completed is not None:
                    terminated = True
                    stop_reason = completed
            else:
                self._state.empty_reservoir_rounds += 1
                if (
                    self._state.empty_reservoir_rounds
                    >= self.config.max_empty_reservoir_rounds
                ):
                    terminated = True
                    stop_reason = "empty_reservoir_limit"

        reward, reward_components = self._reward(
            new_observations, parse_error, baseline, selection
        )
        incumbent_after = self.objectives.incumbent(self._state.observations) if len(
            self.objectives.specs
        ) == 1 else None

        self._state.next_round = round_idx + 1
        truncated = not terminated and self._state.next_round >= self.config.iterations
        if truncated:
            stop_reason = "iteration_budget"
        self._done = terminated or truncated

        observation = render_step_observation(
            round_idx=round_idx,
            remaining_rounds=max(0, remaining),
            parse_error=parse_error,
            proposals_count=len(payloads),
            rejections=rejections,
            evaluations=new_observations,
            incumbent=incumbent_after,
        )
        info = _jsonable(
            {
                "task": self.task_spec.task,
                "round_idx": round_idx,
                "parse_error": parse_error,
                "proposal_count": len(payloads),
                "rejections": [item.to_dict() for item in rejections],
                "evaluated": [item.to_dict() for item in new_observations],
                "incumbent": None if incumbent_after is None else incumbent_after.to_dict(),
                "selection": None if selection is None else selection.to_dict(),
                "reward": reward,
                "reward_policy": self.config.reward,
                "reward_components": reward_components,
                "empty_reservoir_rounds": self._state.empty_reservoir_rounds,
                "terminated": terminated,
                "truncated": truncated,
                "stop_reason": stop_reason,
            }
        )
        return EnvStep(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _evaluate(self, candidate: Candidate) -> EvaluationResult:
        """Engine-identical evaluation wrapping."""

        try:
            result = self._evaluator.evaluate(candidate)
            if result.candidate_id != candidate.candidate_id:
                raise ValueError("evaluator returned a mismatched candidate_id")
            return self.objectives.validate_result(result)
        except TimeoutError as exc:
            return EvaluationResult(candidate.candidate_id, "timed_out", error=str(exc))
        except Exception as exc:  # noqa: BLE001 - engine evaluation semantics
            return EvaluationResult(candidate.candidate_id, "failed", error=str(exc))

    # ------------------------------------------------------------- selection

    def _select(
        self,
        candidates: Sequence[Candidate],
        count: int,
    ) -> BOSelectionResult:
        """Engine-identical selection: GP acquisition or reservoir order."""

        if self.selector is None or self.surrogate_encoder is None:
            return BOSelectionResult(
                selected_candidate_ids=tuple(
                    item.candidate_id for item in candidates[:count]
                ),
                metadata={"mode": "reservoir_order"},
            )
        history = [
            BOObservation.from_observation(
                observation,
                objective_names=self.objectives.names,
                feature=(
                    observation.surrogate
                    if observation.surrogate is not None
                    else self.surrogate_encoder.encode(observation.candidate)
                ),
            )
            for observation in self._state.observations
            if observation.evaluation.succeeded
        ]
        self.selector.fit(history)
        representations = {
            candidate.candidate_id: self.surrogate_encoder.encode(candidate)
            for candidate in candidates
        }
        return self.selector.select(candidates, representations, count=count)

    def _resolve_selection(
        self,
        candidates: Sequence[Candidate],
        selection: BOSelectionResult,
    ) -> list[Candidate]:
        by_id = {item.candidate_id: item for item in candidates}
        unknown = [
            item for item in selection.selected_candidate_ids if item not in by_id
        ]
        if unknown:
            raise ValueError(
                "selector returned candidate ids outside the active reservoir: "
                + ", ".join(unknown)
            )
        return [by_id[item] for item in selection.selected_candidate_ids]

    # ----------------------------------------------------------------- reward

    def _oriented(self, metrics: Mapping[str, Any]) -> tuple[float, ...]:
        return self.objectives.orient_for_maximization(metrics)

    def _componentwise_best(
        self, observations: Sequence[Observation]
    ) -> tuple[float, ...] | None:
        """Per-objective best oriented value across successful observations."""

        values = [
            self._oriented(item.metrics)
            for item in observations
            if item.evaluation.succeeded
        ]
        if not values:
            return None
        width = len(self.objectives.specs)
        return tuple(max(vector[i] for vector in values) for i in range(width))

    def _reward(
        self,
        new_observations: Sequence[Observation],
        parse_error: str | None,
        baseline: tuple[float, ...] | None,
        selection: BOSelectionResult | None,
    ) -> tuple[float, dict[str, Any]]:
        succeeded = [item for item in new_observations if item.evaluation.succeeded]
        components: dict[str, Any] = {"evaluated": len(new_observations), "succeeded": len(succeeded)}

        if not succeeded:
            if parse_error is not None:
                components["kind"] = "invalid_action"
                return self.config.reward_invalid, components
            if new_observations:
                components["kind"] = "evaluation_failure"
                return self.config.reward_failure, components
            components["kind"] = "all_rejected"
            return self.config.reward_invalid, components

        if self.config.reward == "acquisition":
            return self._acquisition_reward(new_observations, selection, components)

        if self.config.reward == "hypervolume":
            return self._hypervolume_reward(new_observations, components)

        new_values = [self._oriented(item.metrics) for item in succeeded]
        after = tuple(max(values[i] for values in new_values) for i in range(len(self.objectives.specs)))

        if self.config.reward == "raw":
            components["kind"] = "raw"
            reward = sum(after)
            return reward, components

        if baseline is None:
            baseline = tuple(0.0 for _ in self.objectives.specs)
        improvements = tuple(max(0.0, a - b) for a, b in zip(after, baseline, strict=True))
        improvement = sum(improvements)
        if self.config.reward == "binary":
            components["kind"] = "binary"
            return 1.0 if improvement > 0.0 else 0.0, components

        components.update(
            {
                "kind": "improvement",
                "baseline": baseline,
                "after": after,
                "improvements": improvements,
            }
        )
        return improvement, components

    def _acquisition_reward(
        self,
        new_observations: Sequence[Observation],
        selection: BOSelectionResult | None,
        components: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        """Reward from the GP acquisition scores of the evaluated candidates.

        The scores come from the selector run before evaluation, so the reward
        reflects the decision-time expected utility of the proposed candidate
        (mean + beta * std for GP-UCB), not its measured outcome.
        """

        evaluated_ids = {item.candidate_id for item in new_observations}
        if selection is None:
            raise RuntimeError(
                "reward policy 'acquisition' requires a selector-backed selection"
            )
        scores = [
            float(prediction.acquisition_score)
            for prediction in selection.predictions
            if prediction.candidate_id in evaluated_ids
            and prediction.acquisition_score is not None
        ]
        if not scores:
            components["kind"] = "acquisition"
            components["scores"] = []
            components["agg"] = self.config.acquisition_agg
            return 0.0, components
        agg = self.config.acquisition_agg
        value = sum(scores) / len(scores) if agg == "mean" else max(scores)
        components.update({"kind": "acquisition", "scores": scores, "agg": agg})
        return value, components

    @staticmethod
    def _hypervolume_2d(
        points: Sequence[tuple[float, float]], ref: tuple[float, float]
    ) -> float:
        """Dominated hypervolume of a 2-objective (maximise) point set vs ``ref``.

        ``ref`` is the lower-left reference corner in oriented space; only points
        that strictly dominate it contribute.
        """

        rx, ry = ref
        pts = [(a, b) for (a, b) in points if a > rx and b > ry]
        if not pts:
            return 0.0
        # Pareto front: scan by x descending, keep strictly increasing y.
        pts.sort(key=lambda p: (-p[0], -p[1]))
        front: list[tuple[float, float]] = []
        best_y = float("-inf")
        for a, b in pts:
            if b > best_y:
                front.append((a, b))
                best_y = b
        front.reverse()  # -> x ascending, y descending
        hv = 0.0
        prev_x = rx
        for a, b in front:
            hv += (a - prev_x) * (b - ry)
            prev_x = a
        return hv

    def _hypervolume_reward(
        self,
        new_observations: Sequence[Observation],
        components: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        """Per-round Pareto-front hypervolume improvement from real outcomes.

        Reward = HV(front after this round) - HV(front before), clipped at 0.
        Uses measured objective values (unlike ``acquisition``), so it directly
        rewards how much the round pushed the observed Pareto front. Two
        objectives only; requires a fixed ``reward_ref_point`` (oriented-space
        domain nadir) — the moving per-round nadir is disabled (see PR #2).
        """

        if len(self.objectives.specs) != 2:
            raise ValueError(
                "hypervolume reward supports exactly 2 objectives, got "
                f"{len(self.objectives.specs)}"
            )
        new_ids = {id(o) for o in new_observations}
        all_obs = list(self._state.observations)  # already includes this round
        prior = [o for o in all_obs if id(o) not in new_ids]

        def pts(observations: Sequence[Observation]) -> list[tuple[float, float]]:
            out: list[tuple[float, float]] = []
            for o in observations:
                if o.evaluation.succeeded:
                    v = self._oriented(o.metrics)
                    out.append((float(v[0]), float(v[1])))
            return out

        after_pts = pts(all_obs)
        before_pts = pts(prior)
        # A FIXED reference point is required. A per-round moving nadir (min of the
        # observed points) was tried and is broken: it lets evaluating a bad
        # molecule lower the reference and pay almost as much as finding a good one
        # (KangOxford PR #2 measured ΔHV total corr -0.19 with the worst evaluated
        # molecule), and its eps floor emits ~1e-13 "non-zero" rewards that starve
        # GRPO of gradient while slipping past the exact-equality zero-variance
        # counter. Callers must pass a fixed domain nadir via reward_ref_point.
        if self.config.reward_ref_point is None:
            raise ValueError(
                "hypervolume reward requires a fixed reward_ref_point (oriented-space "
                "domain nadir); the moving per-round nadir is disabled (see PR #2)."
            )
        ref = (
            float(self.config.reward_ref_point[0]),
            float(self.config.reward_ref_point[1]),
        )
        hv_after = self._hypervolume_2d(after_pts, ref)
        hv_before = self._hypervolume_2d(before_pts, ref)
        dhv = max(0.0, hv_after - hv_before)
        components.update(
            {
                "kind": "hypervolume",
                "hv_before": hv_before,
                "hv_after": hv_after,
                "ref_point": ref,
            }
        )
        return dhv, components

    # ------------------------------------------------------------------- run

    def run(
        self,
        policy: Callable[[str], str],
        *,
        max_steps: int | None = None,
    ) -> EpisodeResult:
        """Drive a full episode with a local synchronous policy."""

        observation = self.reset()
        steps: list[EnvStep] = []
        horizon = max_steps if max_steps is not None else self.config.iterations
        for _ in range(horizon):
            action = policy(observation)
            step = self.step(action)
            steps.append(step)
            observation = step.observation
            if step.done:
                break
        if steps and steps[-1].info.get("stop_reason"):
            stop_reason = str(steps[-1].info["stop_reason"])
        elif steps and steps[-1].terminated:
            stop_reason = "empty_reservoir_limit"
        else:
            stop_reason = "iteration_budget"
        incumbent = self.objectives.incumbent(self._state.observations) if len(
            self.objectives.specs
        ) == 1 else None
        return EpisodeResult(
            steps=steps,
            history=self.history,
            total_reward=sum(step.reward for step in steps),
            best_metrics=None if incumbent is None else dict(incumbent.metrics),
            stop_reason=stop_reason,
            rounds=len(steps),
        )


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


__all__ = [
    "EnvConfig",
    "EnvStep",
    "EpisodeResult",
    "LDMEnv",
    "REWARD_POLICIES",
]
