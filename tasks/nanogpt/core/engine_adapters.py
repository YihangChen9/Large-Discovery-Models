"""LDMEngine behavioral adapters for the nanoGPT model-based search task.

One nanoGPT model-based iteration becomes one engine round:

* :class:`NanogptIterationExpander` runs the iteration's inner surrogate
  search (root creation, breadth/depth generation, GP surrogate scoring, leaf
  selection) and emits the selected leaf -- plus the root when
  ``--evaluate-root`` applies -- as proposals. The depth traversal itself stays
  inside the expander because LDMEngine rounds model a flat reservoir, not a
  per-depth tree; the engine owns the real evaluations, budget, events, and
  checkpoints.
* :class:`NanogptCandidateDomain` admits state-backed proposals.
* :class:`NanogptEvaluator` executes the training command for one leaf and
  applies the task's buffer/feedback bookkeeping.
* :class:`NanogptCampaignTracker` is the shared mutable per-campaign state
  (best/latest state, previous best score) used by both adapters.

:func:`materialize_legacy_run` merges engine artifacts back into the task's
``model_based_summary.json`` / ``summary.json`` files.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any, Mapping, Optional, Sequence

from ldm_tts.contracts import (
    AcquisitionSpec,
    Candidate,
    CandidateRejection,
    EvaluationResult,
    RawProposal,
)
from ldm_tts.engine.expansion import ExpansionRequest, ExpansionResult
from ldm_tts.optimization.records import BOObservation, BOSelectionResult, SurrogateVector
from ldm_tts.transport import ProposalResponse


class NanogptCampaignTracker:
    """Shared best/latest/previous-score state across the nanogpt adapters."""

    def __init__(self, *, best_state=None, latest_state=None) -> None:
        self.best_state = best_state
        self.latest_state = latest_state
        self.previous_best_score: float | None = None

    def record(self, state, *, minimize: bool) -> None:
        from tasks.nanogpt.core.workflow import finite_score, is_better

        if state is None or state.score is None:
            return
        if not finite_score(state.score):
            return
        self.latest_state = state
        if self.best_state is None or self.best_state.score is None or is_better(
            state.score, self.best_state.score, minimize=minimize
        ):
            self.best_state = state


# ---------------------------------------------------------------------------
# Candidate domain
# ---------------------------------------------------------------------------


class NanogptCandidateDomain:
    """Admit state-backed nanoGPT proposals."""

    def admit(self, proposal: RawProposal) -> Candidate | CandidateRejection:
        payload = proposal.payload
        if not isinstance(payload, Mapping) or not payload.get("state_id"):
            return CandidateRejection(
                "invalid",
                "proposal payload must contain a state_id",
                proposal.source,
            )
        state_id = str(payload["state_id"])
        return Candidate(
            candidate_id=state_id,
            payload=dict(payload),
            canonical_key=f"state-{state_id}",
            source=proposal.source,
        )


# ---------------------------------------------------------------------------
# Initialization and reservoir expansion
# ---------------------------------------------------------------------------


class NanogptWarmupExpander:
    """Generate warm-up states; evaluation remains owned by the campaign."""

    def __init__(self, *, engine, args, logger, progress, requested: int) -> None:
        self.engine = engine
        self.args = args
        self.logger = logger
        self.progress = progress
        self.requested = max(0, int(requested))
        self.strategy = ""
        self.rng_seed = (
            int(args.warmup_seed)
            if int(args.warmup_seed) != 0
            else int(time.time_ns() % (2**32))
        )
        self.rng = random.Random(self.rng_seed)
        self.root = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(self._expand(request))

    async def _expand(self, request: ExpansionRequest) -> ExpansionResult:
        from tasks.nanogpt.core.workflow import (
            OperationSearchEngine,
            create_random_operation_warmup_state,
            resolve_warmup_strategy,
            write_state_update,
        )

        if self.root is None:
            self.root = self.engine.create_seed_state()
            self.root.metrics["warmup_root"] = True
            write_state_update(self.engine, self.root)
            self.strategy = resolve_warmup_strategy(self.args, self.engine)
            self.logger.write(
                f"warmup start requested={self.requested} strategy={self.strategy} "
                f"include_root={self.args.warmup_include_root} rng_seed={self.rng_seed}"
            )

        completed = sum(
            observation.evaluation.succeeded
            and str(observation.candidate.payload.get("kind", "")).startswith("warmup")
            for observation in request.observations
        )
        warmup_index = completed + 1
        self.progress.status(
            f"warmup generating {warmup_index}/{self.requested}"
        )
        if self.args.warmup_include_root and completed == 0:
            state = self.root
            kind = "warmup_root"
            state.metrics["warmup_index"] = warmup_index
            state.metrics["warmup_strategy"] = "root"
            write_state_update(self.engine, state)
        elif self.strategy == "random_operation":
            if not isinstance(self.engine, OperationSearchEngine):
                raise RuntimeError(
                    "random_operation warm-up requires OperationSearchEngine"
                )
            state = create_random_operation_warmup_state(
                self.engine,
                self.root,
                self.rng,
                warmup_index=warmup_index,
                total=self.requested,
            )
            kind = "warmup"
        else:
            children = await self.engine.expand_state(
                self.root,
                1,
                search_note=(
                    f"warm-up candidate {warmup_index}/{self.requested}: propose a "
                    "diverse candidate for GP initialization"
                ),
            )
            if not children:
                return ExpansionResult(
                    proposals=(RawProposal(None, "empty_warmup"),),
                    metadata={"phase": "warmup", "generation_failed": True},
                )
            state = children[0]
            state.metrics["warmup_index"] = warmup_index
            state.metrics["warmup_strategy"] = self.strategy
            write_state_update(self.engine, state)
            kind = "warmup"

        return ExpansionResult(
            proposals=(
                _state_proposal(
                    state,
                    kind=kind,
                    root_state_id=self.root.state_id,
                    iteration=0,
                    surrogate_metrics={},
                ),
            ),
            metadata={
                "phase": "warmup",
                "warmup_index": warmup_index,
                "strategy": self.strategy,
            },
        )


class NanogptCampaignExpander:
    """Route initialization and search through one engine expansion seam."""

    def __init__(
        self,
        warmup: NanogptWarmupExpander,
        search: "NanogptIterationExpander",
        *,
        warmup_target: int,
    ) -> None:
        self.warmup = warmup
        self.search = search
        self.warmup_target = max(0, int(warmup_target))

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        completed = sum(
            observation.evaluation.succeeded
            and str(observation.candidate.payload.get("kind", "")).startswith("warmup")
            for observation in request.observations
        )
        if completed < self.warmup_target:
            return self.warmup.expand(request)
        return self.search.expand(request)


class NanogptIterationExpander:
    """One model-based iteration as one engine reservoir expansion."""

    def __init__(
        self,
        *,
        engine,
        args,
        logger,
        progress,
        buffer_entries,
        feedback_memory,
        buffer_path,
        run_buffer_path,
        run_name,
        train_file,
        tracker: NanogptCampaignTracker,
    ) -> None:
        self.engine = engine
        self.args = args
        self.logger = logger
        self.progress = progress
        self.buffer_entries = buffer_entries
        self.feedback_memory = feedback_memory
        self.buffer_path = buffer_path
        self.run_buffer_path = run_buffer_path
        self.run_name = run_name
        self.train_file = train_file
        self.tracker = tracker
        self._loop: asyncio.AbstractEventLoop | None = None
        self.records: list[dict[str, Any]] = []

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(self._search(request))

    async def _search(self, request: ExpansionRequest) -> ExpansionResult:
        from tasks.nanogpt.core.workflow import (
            GPSurrogate,
            OperationSearchEngine,
            choose_seed_path,
            format_gp_progress,
            refresh_projected_buffer_entries,
            run_inner_surrogate_search,
        )

        engine = self.engine
        args = self.args
        completed_search_rounds = {
            observation.round_idx
            for observation in request.observations
            if observation.round_idx is not None
            and observation.candidate.payload.get("kind") in {"root", "selected"}
        }
        iteration = len(completed_search_rounds) + 1
        buffer_entries = self.buffer_entries
        if isinstance(engine, OperationSearchEngine):
            engine.current_iteration = iteration
            refresh_projected_buffer_entries(buffer_entries, engine.operation_schema, args)
        buffer_size_before = len(buffer_entries)
        surrogate = GPSurrogate(
            buffer_entries,
            lengthscale=args.gp_lengthscale,
            noise=args.gp_noise,
            prior_score=args.prior_score,
            prior_std=args.prior_std,
            minimize=engine.config.minimize,
        )
        gp_summary_before = surrogate.summary()
        self.progress.status(f"iteration {iteration} GP {format_gp_progress(gp_summary_before)}")
        self.logger.write(
            f"iteration={iteration} gp_before {format_gp_progress(gp_summary_before)}"
        )
        best_actual = self._best_actual(request)
        latest_actual = self._latest_actual(request)
        engine.config.seed_train_path = choose_seed_path(
            args, self.train_file, best_actual, latest_actual
        )
        root = engine.create_seed_state()
        self.logger.write(
            f"iteration={iteration} root={root.state_id} method={args.effective_method} "
            f"buffer_size_before={buffer_size_before}"
        )

        proposals: list[RawProposal] = []
        if iteration == 1 and args.evaluate_root and not args.skip_eval:
            proposals.append(
                _state_proposal(
                    root,
                    kind="root",
                    root_state_id=root.state_id,
                    iteration=iteration,
                    surrogate_metrics={},
                )
            )

        scored_states, leaves = await run_inner_surrogate_search(
            engine,
            args,
            root,
            surrogate,
            iteration=iteration,
            progress=self.progress,
        )
        if isinstance(engine, OperationSearchEngine):
            refresh_projected_buffer_entries(buffer_entries, engine.operation_schema, args)
        self.logger.write(
            f"iteration={iteration} generated={len(scored_states)} leaves={len(leaves)} "
            f"select_from={args.select_from}"
        )

        pool = leaves if args.select_from == "leaves" and leaves else scored_states
        selectable = [
            state
            for state in pool
            if state.metrics.get("surrogate_score") is not None
            and not state.metrics.get("feature_only_action")
        ]
        if not selectable and pool is not scored_states:
            selectable = [
                state
                for state in scored_states
                if state.metrics.get("surrogate_score") is not None
                and not state.metrics.get("feature_only_action")
            ]
        selected = None
        selected_surrogate_metrics: dict[str, Any] = {}
        if selectable and not args.skip_eval:
            for candidate_state in selectable:
                proposals.append(
                    _state_proposal(
                        candidate_state,
                        kind="selected",
                        root_state_id=root.state_id,
                        iteration=iteration,
                        surrogate_metrics=dict(candidate_state.metrics),
                    )
                )
        elif selectable:
            for candidate_state in selectable:
                engine.defer_evaluation(
                    candidate_state,
                    reason="Real evaluation skipped by --skip-eval.",
                )
        elif not scored_states:
            self.logger.write(f"iteration={iteration} selected=none reason=no_scored_states")
        else:
            self.logger.write(f"iteration={iteration} selected=none reason=no_selectable_states")

        if not proposals:
            proposals = [RawProposal(None, "empty_reservoir")]
        self.records.append({
            "iteration": iteration,
            "round_idx": int(request.round_idx),
            "method": args.effective_method,
            "root_state_id": root.state_id,
            "selected_state_id": None if selected is None else selected.state_id,
            "selected_surrogate_score": selected_surrogate_metrics.get("surrogate_score"),
            "selected_pred": selected_surrogate_metrics.get("surrogate_pred"),
            "selected_std": selected_surrogate_metrics.get("surrogate_std"),
            "selected_ei": selected_surrogate_metrics.get("surrogate_ei"),
            "score_key": engine.config.score_key,
            "previous_best_score": self.tracker.previous_best_score,
            "gp_before": gp_summary_before,
            "gp_after": None,
            "generated_count": len(scored_states),
            "buffer_size_before_iteration": buffer_size_before,
            "buffer_size_after_iteration": len(buffer_entries),
            "real_evaluations": 0,
            "actual_state_ids": [],
            "skip_eval": bool(args.skip_eval),
            "_selected_state": selected,
        })
        return ExpansionResult(
            proposals=tuple(proposals),
            attempts=(),
            metadata={
                "iteration": iteration,
                "root_state_id": root.state_id,
                "selected_state_id": None if selected is None else selected.state_id,
                "generated_count": len(scored_states),
                "skip_eval": bool(args.skip_eval),
            },
        )

    def _best_actual(self, request: ExpansionRequest):
        from tasks.nanogpt.core.workflow import is_better, state_from_id

        observed_ids = [
            str(observation.candidate.payload.get("state_id"))
            for observation in request.observations
            if observation.evaluation.succeeded
            and observation.candidate.payload.get("state_id")
            and (
                self.args.warmup_updates_seed
                or not str(observation.candidate.payload.get("kind", "")).startswith(
                    "warmup"
                )
            )
        ]
        best = self.tracker.best_state
        for state_id in reversed(observed_ids):
            state = state_from_id(self.engine, state_id)
            if state is not None and state.score is not None:
                if best is None or best.score is None or is_better(
                    state.score, best.score, minimize=self.engine.config.minimize
                ):
                    best = state
        return best

    def _latest_actual(self, request: ExpansionRequest):
        from tasks.nanogpt.core.workflow import state_from_id

        observed_ids = [
            str(observation.candidate.payload.get("state_id"))
            for observation in request.observations
            if observation.evaluation.succeeded
            and observation.candidate.payload.get("state_id")
            and (
                self.args.warmup_updates_seed
                or not str(observation.candidate.payload.get("kind", "")).startswith(
                    "warmup"
                )
            )
        ]
        if observed_ids:
            state = state_from_id(self.engine, observed_ids[-1])
            if state is not None:
                return state
        return self.tracker.latest_state


class NanogptPrecomputedSelector:
    """Choose the best surrogate-scored proposal inside the shared engine."""

    def __init__(self, expander: NanogptIterationExpander, *, objective_name: str) -> None:
        self.expander = expander
        self.objective_name = objective_name

    def describe(self) -> AcquisitionSpec:
        return AcquisitionSpec(
            name="precomputed_surrogate_score",
            objective_names=(self.objective_name,),
            score_direction="minimize",
            selection_rule="evaluate an optional root, then the minimum surrogate_score",
        )

    def fit(self, history: Sequence[BOObservation]) -> None:
        del history

    def select(
        self,
        candidates: Sequence[Candidate],
        representations: Mapping[str, SurrogateVector],
        *,
        count: int = 1,
    ) -> BOSelectionResult:
        del representations
        warmup = [
            candidate
            for candidate in candidates
            if str(candidate.payload.get("kind", "")).startswith("warmup")
        ]
        if warmup:
            return BOSelectionResult(
                selected_candidate_ids=(warmup[0].candidate_id,) if count > 0 else (),
                metadata={"mode": "warmup_order"},
            )
        roots = [candidate for candidate in candidates if candidate.payload.get("kind") == "root"]
        scored = [
            candidate
            for candidate in candidates
            if candidate.payload.get("kind") != "root"
            and candidate.payload.get("surrogate_metrics", {}).get("surrogate_score") is not None
        ]
        scored.sort(
            key=lambda candidate: (
                float(candidate.payload["surrogate_metrics"]["surrogate_score"]),
                candidate.candidate_id,
            )
        )
        chosen: list[Candidate] = []
        if roots and count > 0:
            chosen.append(roots[0])
        if scored and len(chosen) < count:
            chosen.append(scored[0])

        selected = next(
            (candidate for candidate in chosen if candidate.payload.get("kind") == "selected"),
            None,
        )
        if selected is not None and self.expander.records:
            metrics = dict(selected.payload.get("surrogate_metrics") or {})
            record = self.expander.records[-1]
            record["selected_state_id"] = selected.payload.get("state_id")
            record["selected_surrogate_score"] = metrics.get("surrogate_score")
            record["selected_pred"] = metrics.get("surrogate_pred")
            record["selected_std"] = metrics.get("surrogate_std")
            record["selected_ei"] = metrics.get("surrogate_ei")
            from tasks.nanogpt.core.workflow import state_from_id, write_state_update

            state = state_from_id(self.expander.engine, selected.payload.get("state_id"))
            if state is not None:
                state.metrics["model_based_selected_iteration"] = selected.payload.get(
                    "iteration"
                )
                write_state_update(self.expander.engine, state)
                record["_selected_state"] = state
            self.expander.logger.write(
                f"iteration={selected.payload.get('iteration')} "
                f"selected={selected.payload.get('state_id')} "
                f"surrogate_score={metrics.get('surrogate_score')} "
                f"pred={metrics.get('surrogate_pred')} std={metrics.get('surrogate_std')} "
                f"ei={metrics.get('surrogate_ei')}"
            )
        return BOSelectionResult(
            selected_candidate_ids=tuple(candidate.candidate_id for candidate in chosen),
            metadata={
                "mode": "precomputed_surrogate_score",
                "candidate_count": len(candidates),
            },
        )

def _state_proposal(
    state,
    *,
    kind: str,
    root_state_id: str,
    iteration: int,
    surrogate_metrics: Mapping[str, Any],
) -> RawProposal:
    payload = {
        "state_id": state.state_id,
        "kind": kind,
        "root_state_id": root_state_id,
        "iteration": iteration,
        "surrogate_metrics": dict(surrogate_metrics),
        "train_path": str(state.train_path),
    }
    return RawProposal(payload, f"model_based_{kind}", metadata={"state_id": state.state_id})


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class NanogptEvaluator:
    """Execute the training command for one state and update bookkeeping."""

    def __init__(
        self,
        *,
        engine,
        args,
        logger,
        progress,
        buffer_entries,
        feedback_memory,
        buffer_path,
        run_buffer_path,
        run_name,
        tracker: NanogptCampaignTracker,
    ) -> None:
        self.engine = engine
        self.args = args
        self.logger = logger
        self.progress = progress
        self.buffer_entries = buffer_entries
        self.feedback_memory = feedback_memory
        self.buffer_path = buffer_path
        self.run_buffer_path = run_buffer_path
        self.run_name = run_name
        self.tracker = tracker

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        from tasks.nanogpt.core.workflow import (
            append_buffer_entry,
            finite_score,
            make_buffer_entry,
            state_from_id,
            updated_best_score,
            write_state_update,
        )

        engine = self.engine
        state = state_from_id(engine, candidate.payload.get("state_id"))
        if state is None:
            return EvaluationResult(
                candidate.candidate_id,
                "failed",
                error=f"state not found: {candidate.payload.get('state_id')}",
            )
        engine.evaluate_state(state)
        self.progress.evaluated(state)
        score = state.score
        score_key = engine.config.score_key
        if score is None or not finite_score(score):
            return EvaluationResult(
                candidate.candidate_id,
                "failed",
                error=f"evaluation produced no finite {score_key}",
                metadata={"state_id": state.state_id},
            )
        kind = str(candidate.payload.get("kind", "selected"))
        iteration = int(candidate.payload.get("iteration", 0))
        root = state_from_id(engine, candidate.payload.get("root_state_id")) or state
        surrogate_metrics = dict(candidate.payload.get("surrogate_metrics") or {})
        provisional_best = updated_best_score(
            self.tracker.previous_best_score, score, minimize=engine.config.minimize
        )
        self.feedback_memory.record(
            kind=kind,
            iteration=iteration,
            state=state,
            root=root,
            selected_surrogate_metrics=surrogate_metrics,
            previous_best_score=self.tracker.previous_best_score,
            best_score_after=provisional_best,
        )
        engine.config.feedback_context = self.feedback_memory.prompt_context()
        updates_campaign_seed = (
            not kind.startswith("warmup") or bool(self.args.warmup_updates_seed)
        )
        if updates_campaign_seed:
            self.tracker.previous_best_score = provisional_best
            self.tracker.record(state, minimize=engine.config.minimize)
        state.metrics.update(
            {
                key: value
                for key, value in surrogate_metrics.items()
                if key.startswith("surrogate_")
                or key
                in {
                    "model_based_iteration",
                    "feature_version",
                    "feature_source_hash",
                    "extracted_params",
                    "operation_schema_version",
                    "operation_schema_feature_names",
                    "operation_schema_feature_count",
                    "operation_feature_expansions",
                    "operations",
                }
            }
        )
        if not kind.startswith("warmup"):
            state.metrics["model_based_selected_iteration"] = iteration
        write_state_update(engine, state)
        entry = make_buffer_entry(
            state,
            self.args,
            iteration=iteration,
            run_name=self.run_name,
            score_key=score_key,
        )
        if entry is not None:
            append_buffer_entry(self.buffer_path, entry, mirror_path=self.run_buffer_path)
            self.buffer_entries.append(entry)
            self.logger.write(
                f"iteration={iteration} evaluated_{kind}={state.state_id} "
                f"{score_key}={entry.score} buffer_size={len(self.buffer_entries)}"
            )
        return EvaluationResult(
            candidate.candidate_id,
            "succeeded",
            metrics={score_key: float(score)},
            resource_usage={"benchmark_jobs": 1},
            artifacts={"train_path": str(state.train_path)},
            metadata={"state_id": state.state_id, "kind": kind, "iteration": iteration},
        )


# ---------------------------------------------------------------------------
# Legacy run export
# ---------------------------------------------------------------------------


def merge_engine_summary(summary_path, engine_summary: Mapping[str, Any]) -> None:
    """Merge the engine summary fields into the task's summary.json."""
    payload: dict[str, Any] = {}
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("engine_summary", dict(engine_summary))
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
