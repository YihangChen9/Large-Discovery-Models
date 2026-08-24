"""LDMEngine behavioral adapters for the antibody task.

The native AntBO acquisition loop in ``ldm_light.ldm_acq.run_one`` is retired
as the campaign driver; the same scientific components now plug into
:class:`ldm_tts.engine.LDMEngine`:

* :class:`AntibodyCandidateDomain` -- CDRH3 length/alphabet/developability
  admission (``CandidateDomainAdapter``).
* :class:`AntibodyReservoirExpander` -- warmup/direct/policy/legacy proposal
  generation (``ReservoirExpander``). For policy and legacy-policy methods the
  DSL reservoir search (including its internal GP scoring) stays inside the
  expander and emits the final selected batch; for direct methods the expander
  only generates and the GP acquisition selection happens in
  :class:`AntibodyGPSelector`.
* :class:`AntibodyEvaluator` -- Absolut / random energy scoring
  (``CandidateEvaluator``).
* :class:`AntibodyGPSelector` -- GP fit + max/softmax acquisition selection for
  direct methods, with warmup rounds falling back to reservoir order
  (``AcquisitionSelector``).
* :class:`AntibodySurrogateEncoder` -- categorical CDRH3 index vectors
  (``SurrogateEncoder``).

:func:`materialize_legacy_run` re-exports ``results.csv`` and
``llm_acq_decisions.jsonl`` from the engine events.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ldm_tts.contracts import (
    AcquisitionSpec,
    Candidate,
    CandidateRejection,
    EvaluationResult,
    RawProposal,
    SurrogateSpaceSpec,
)
from ldm_tts.engine.expansion import ExpansionRequest, ExpansionResult
from ldm_tts.optimization.records import (
    BOObservation,
    BOSelectionResult,
    SurrogateVector,
)

from tasks.antibody.core.ldm_light.ldm_acq import (
    AA,
    AROMATIC,
    N_GLYCO,
    collect_direct_sequence_action,
    longest_hydrophobic_run,
    net_charge,
    passes_developability,
    propose,
    seqs_to_indices,
    valid_seq,
)
from tasks.antibody.core.ldm_light.selection import select_by_acquisition

SEQUENCE_CONSTRAINTS = {
    "max_cysteine": 1,
    "max_hydrophobic_run": 4,
    "max_aromatic_FWY": 2,
    "net_charge_range": [-1.0, 2.0],
    "forbid_n_glycosylation_NXS_or_NXT": True,
}


# ---------------------------------------------------------------------------
# Candidate domain
# ---------------------------------------------------------------------------


class AntibodyCandidateDomain:
    """Admit direct CDRH3 sequence proposals."""

    def __init__(self, seq_len: int) -> None:
        self.seq_len = int(seq_len)

    def admit(self, proposal: RawProposal) -> Candidate | CandidateRejection:
        payload = proposal.payload
        if not isinstance(payload, Mapping) or not payload.get("sequence"):
            return CandidateRejection(
                "invalid",
                "proposal payload must contain a sequence",
                proposal.source,
            )
        sequence = str(payload["sequence"]).strip().upper()
        reasons = _sequence_violations(sequence, self.seq_len)
        if reasons:
            return CandidateRejection(
                "invalid",
                "; ".join(reasons),
                proposal.source,
                metadata={"sequence": sequence, "reasons": reasons},
            )
        admitted_payload = dict(payload)
        admitted_payload["sequence"] = sequence
        return Candidate(
            candidate_id="ab-" + hashlib.sha256(sequence.encode()).hexdigest()[:12],
            payload=admitted_payload,
            canonical_key=sequence,
            source=proposal.source,
            metadata={
                "llm_score": payload.get("score"),
                "acquisition_score": payload.get("acquisition_score"),
                "selector_source": proposal.metadata.get("selector_source", proposal.source),
                "acquisition_used": bool(proposal.metadata.get("acquisition_used", False)),
                "phase": proposal.metadata.get("phase", ""),
            },
        )


def _sequence_violations(sequence: str, seq_len: int) -> list[str]:
    reasons: list[str] = []
    if len(sequence) != int(seq_len):
        reasons.append("length")
    if any(aa not in AA for aa in sequence):
        reasons.append("alphabet")
    if valid_seq(sequence, int(seq_len)):
        if sequence.count("C") > SEQUENCE_CONSTRAINTS["max_cysteine"]:
            reasons.append("max_cysteine")
        if longest_hydrophobic_run(sequence) > SEQUENCE_CONSTRAINTS["max_hydrophobic_run"]:
            reasons.append("max_hydrophobic_run")
        if sum(1 for aa in sequence if aa in AROMATIC) > SEQUENCE_CONSTRAINTS["max_aromatic_FWY"]:
            reasons.append("max_aromatic_FWY")
        low, high = SEQUENCE_CONSTRAINTS["net_charge_range"]
        if not low <= net_charge(sequence) <= high:
            reasons.append("net_charge_range")
        if N_GLYCO.search(sequence) is not None:
            reasons.append("n_glycosylation_NXS_or_NXT")
    if not reasons and not passes_developability(sequence):
        reasons.append("developability")
    return reasons


# ---------------------------------------------------------------------------
# Reservoir expander
# ---------------------------------------------------------------------------


class AntibodyReservoirExpander:
    """Generate antibody proposals per method, mirroring run_one's branches."""

    def __init__(
        self,
        *,
        method: str,
        method_spec: Mapping[str, Any],
        antigen: str,
        seed: int,
        seq_len: int,
        args: Any,
        llm,
        rng,
        acquisition_rng,
        antigen_context: Mapping[str, Any] | None = None,
        orchestrator=None,
        candidate_library: Sequence[str] = (),
        sink=None,
        run_dir: Any = None,
    ) -> None:
        self.method = method
        self.method_spec = dict(method_spec)
        self.antigen = antigen
        self.seed = int(seed)
        self.seq_len = int(seq_len)
        self.args = args
        self.llm = llm
        self.rng = rng
        self.acquisition_rng = acquisition_rng
        self.antigen_context = antigen_context
        self.orchestrator = orchestrator
        self.candidate_library = list(candidate_library)
        self.sink = sink
        self.run_dir = run_dir

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        rows = _rows_from_observations(request.observations)
        observed = {str(row["LastProtein"]) for row in rows}
        initialized = len(rows) >= int(self.args.n_init)
        using_acquisition = initialized and bool(self.method_spec["uses_acquisition"])
        batch_size = int(self.args.batch_size)

        decision: dict[str, Any] = {}
        selector_phase = "warmup"
        if initialized and self.method_spec["base_measure"] == "policy":
            from tasks.antibody.core.ldm_light.reservoir import select_with_policy_reservoir

            _selected, decision = select_with_policy_reservoir(
                llm=self.llm,
                rows=rows,
                antigen=self.antigen,
                seed=self.seed,
                iteration=len(rows),
                antigen_context=self.antigen_context,
                batch_size=batch_size,
                reduction=str(self.method_spec["reduction"]),
                args=self.args,
                select_candidates=False,
            )
            score_key = (
                str(self.args.acq).lower()
                if str(self.args.selection_score) == "acq"
                else f"bias+{str(self.args.acq).lower()}"
            )
            candidates = [
                {
                    "sequence": item["sequence"],
                    "score": None,
                    "acquisition_score": float(item[score_key]),
                    "acquisition_raw": float(item[str(self.args.acq).lower()]),
                    "mu": float(item["mu"]),
                    "sigma": float(item["sigma"]),
                    "source": f"policy_{item['strategy_index']}",
                }
                for item in decision["representatives"]
            ]
            decision["candidates"] = candidates
            selector_phase = "policy"
        elif (
            initialized
            and self.method_spec["base_measure"] == "direct"
            and self.method_spec["uses_acquisition"]
        ):
            from tasks.antibody.core.ldm_light.direct import propose_direct_batch

            candidates, generation = propose_direct_batch(
                llm=self.llm,
                rng=self.rng,
                antigen=self.antigen,
                seq_len=self.seq_len,
                n=int(self.args.gen_m),
                observed=observed,
                rows=rows,
                antigen_context=self.antigen_context,
                args=self.args,
                independent=True,
            )
            decision = {
                "source": f"direct_{self.method_spec['reduction']}",
                "generation": generation,
                "candidates": candidates,
            }
            selector_phase = "direct_acquisition"
        elif initialized and self.method_spec["base_measure"] == "legacy_policy":
            from tasks.antibody.core.ldm_light.ldm_acq import select_with_parallel_ldm

            _selected, decision = select_with_parallel_ldm(
                orchestrator=self.orchestrator,
                rows=rows,
                antigen=self.antigen,
                seed=self.seed,
                iteration=len(rows),
                antigen_context=self.antigen_context,
                batch_size=batch_size,
                args=self.args,
                select_candidates=False,
            )
            score_key = f"bias+{str(self.args.acq).lower()}"
            candidates = [
                {
                    **item,
                    "score": None,
                    "acquisition_score": float(item[score_key]),
                    "acquisition_raw": float(item[str(self.args.acq).lower()]),
                }
                for item in decision["parallel_results"]
            ]
            decision["candidates"] = candidates
            selector_phase = "legacy_policy"
        elif self.method == "legacy_policy_max":
            candidates, decision = propose(
                llm=self.llm,
                rng=self.rng,
                antigen=self.antigen,
                seq_len=self.seq_len,
                batch_size=batch_size,
                observed=observed,
                rows=rows,
                candidate_library=self.candidate_library,
                antigen_context=self.antigen_context,
                args=self.args,
            )
            selector_phase = "warmup_propose"
        else:
            from tasks.antibody.core.ldm_light.direct import propose_direct_batch

            candidates, decision = propose_direct_batch(
                llm=self.llm,
                rng=self.rng,
                antigen=self.antigen,
                seq_len=self.seq_len,
                n=batch_size,
                observed=observed,
                rows=rows,
                antigen_context=self.antigen_context,
                args=self.args,
                independent=False,
            )
            selector_phase = "llm_gen" if self.method == "llm_gen" else "warmup"

        if self.sink is not None:
            collect_direct_sequence_action(
                self.sink,
                decision=decision,
                selected_candidates=candidates,
                rows=rows,
                observed=observed,
                antigen=self.antigen,
                antigen_context=self.antigen_context,
                seq_len=self.seq_len,
                history_top_k=int(self.args.history_top_k),
                seed=self.seed,
                eval_start=len(rows),
                method=self.method,
                run_dir=self.run_dir,
            )

        round_source = str(decision.get("source") or f"warmup_{self.method}")
        proposals = tuple(
            RawProposal(
                dict(candidate),
                round_source,
                metadata={
                    "selector_source": round_source,
                    "acquisition_used": bool(using_acquisition),
                    "phase": selector_phase,
                },
            )
            for candidate in candidates
            if isinstance(candidate, Mapping) and candidate.get("sequence")
        )
        if not proposals:
            proposals = (RawProposal(None, "empty_reservoir"),)
        return ExpansionResult(
            proposals=proposals,
            attempts=(),
            metadata={
                "selector_phase": selector_phase,
                "using_acquisition": bool(using_acquisition),
                "parallel_budget": (
                    int(self.args.parallel_budget) if using_acquisition else batch_size
                ),
                "decision": _jsonable(decision),
                "eval_start": len(rows),
                "method": self.method,
            },
        )


def _jsonable(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


def _rows_from_observations(observations) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best: float | None = None
    for index, observation in enumerate(observations):
        payload = observation.candidate.payload
        sequence = payload.get("sequence") if isinstance(payload, Mapping) else None
        if not sequence:
            continue
        value = observation.evaluation.metrics.get("absolut_energy")
        if value is None:
            continue
        if best is None or float(value) < best:
            best = float(value)
        rows.append({
            "Index": index,
            "LastValue": float(value),
            "BestValue": best,
            "LLMScore": payload.get("score"),
            "AcquisitionScore": payload.get("acquisition_score"),
            "LastProtein": str(sequence),
            "BestProtein": str(sequence) if best == float(value) else rows[-1]["BestProtein"] if rows else str(sequence),
            "Source": observation.candidate.metadata.get("selector_source", observation.candidate.source),
            "AcquisitionUsed": observation.candidate.metadata.get("acquisition_used", False),
        })
    return rows


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class AntibodyEvaluator:
    """Score one CDRH3 candidate through the Absolut or random evaluator."""

    def __init__(self, energy_evaluator) -> None:
        self.energy_evaluator = energy_evaluator

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        sequence = candidate.payload["sequence"]
        values, _sequences = self.energy_evaluator.energy(seqs_to_indices([sequence]))
        value = float(np.asarray(values).ravel()[0])
        if not np.isfinite(value):
            return EvaluationResult(
                candidate.candidate_id,
                "failed",
                error="non-finite Absolut energy",
                metadata={"sequence": sequence},
            )
        return EvaluationResult(
            candidate.candidate_id,
            "succeeded",
            metrics={"absolut_energy": value},
            resource_usage={"benchmark_jobs": 1},
            metadata={"sequence": sequence},
        )


# ---------------------------------------------------------------------------
# Acquisition selector (direct methods)
# ---------------------------------------------------------------------------


class AntibodyGPSelector:
    """GP acquisition selection for direct proposals, with warmup ordering."""

    def __init__(self, *, args: Any, method_spec: Mapping[str, Any], rng) -> None:
        self.args = args
        self.method_spec = dict(method_spec)
        self.rng = rng
        self.rows: list[dict[str, Any]] = []

    def describe(self) -> AcquisitionSpec:
        return AcquisitionSpec(
            name=str(self.args.acq).lower(),
            objective_names=("absolut_energy",),
            score_direction="maximize",
            selection_rule=(
                f"{self.method_spec['reduction']} over direct reservoir acquisition scores"
            ),
            parameters={
                "beta": float(self.args.acq_beta),
                "xi": float(self.args.acq_xi),
                "n_init": int(self.args.n_init),
                "reduction": str(self.method_spec["reduction"]),
                "softmax_eta": float(getattr(self.args, "softmax_eta", 1.0)),
            },
        )

    def fit(self, history: Sequence[BOObservation]) -> None:
        rows: list[dict[str, Any]] = []
        for index, observation in enumerate(history):
            sequence = _sequence_from_bo_observation(observation)
            if sequence is None or not observation.objectives:
                continue
            value = observation.objectives[0]
            if value is None:
                continue
            rows.append({"LastProtein": sequence, "LastValue": float(value), "Index": index})
        self.rows = rows

    def select(
        self,
        candidates: Sequence[Candidate],
        representations: Mapping[str, SurrogateVector],
        *,
        count: int = 1,
    ) -> BOSelectionResult:
        del representations
        if len(self.rows) < int(self.args.n_init):
            selected = [item.candidate_id for item in candidates[: max(1, int(count))]]
            return BOSelectionResult(
                selected_candidate_ids=tuple(selected),
                metadata={"mode": "warmup_order", "rows": len(self.rows)},
            )
        precomputed = all(
            item.payload.get("acquisition_score") is not None for item in candidates
        )
        if precomputed:
            scored = [
                {
                    "sequence": item.payload["sequence"],
                    "acquisition_score": float(item.payload["acquisition_score"]),
                    "mu": item.payload.get("mu"),
                    "sigma": item.payload.get("sigma"),
                }
                for item in candidates
            ]
        else:
            from tasks.antibody.core.ldm_light.direct import score_direct_candidates

            scored = score_direct_candidates(
                [{"sequence": item.payload["sequence"]} for item in candidates],
                self.rows,
                args=self.args,
            )
        selected_indices, probabilities = select_by_acquisition(
            [item["acquisition_score"] for item in scored],
            batch_size=max(1, int(count)),
            reduction=str(self.method_spec["reduction"]),
            eta=float(getattr(self.args, "softmax_eta", 1.0)),
            rng=self.rng,
        )
        selected_ids = tuple(candidates[index].candidate_id for index in selected_indices)
        return BOSelectionResult(
            selected_candidate_ids=selected_ids,
            metadata={
                "mode": (
                    f"precomputed_{self.method_spec['reduction']}"
                    if precomputed
                    else f"direct_{self.method_spec['reduction']}"
                ),
                "reduction": str(self.method_spec["reduction"]),
                "softmax_eta": float(getattr(self.args, "softmax_eta", 1.0)),
                "selected_indices": list(selected_indices),
                "selection_probabilities": probabilities,
                "candidates_scored": scored,
                "selected_candidates": [
                    {
                        "sequence": candidates[index].payload["sequence"],
                        "acquisition_score": scored[index]["acquisition_score"],
                        "mu": scored[index]["mu"],
                        "sigma": scored[index]["sigma"],
                    }
                    for index in selected_indices
                ],
            },
        )


def _sequence_from_bo_observation(observation: BOObservation) -> Optional[str]:
    feature = observation.feature
    if feature is not None and feature.metadata.get("sequence"):
        return str(feature.metadata["sequence"])
    return None


# ---------------------------------------------------------------------------
# Surrogate encoder
# ---------------------------------------------------------------------------


class AntibodySurrogateEncoder:
    """Categorical CDRH3 index representation for the AntBO GP."""

    def __init__(self, seq_len: int) -> None:
        self.seq_len = int(seq_len)

    def describe(self) -> SurrogateSpaceSpec:
        return SurrogateSpaceSpec(
            kind="vector",
            representation="fixed-length categorical CDRH3 indices",
            dimension_policy="fixed",
            dimension=self.seq_len,
            encoder="tasks.antibody.core.engine_adapters.AntibodySurrogateEncoder",
            version="antbo_categorical_sequence_v1",
        )

    def encode(self, candidate: Candidate) -> SurrogateVector:
        sequence = candidate.payload["sequence"]
        indices = seqs_to_indices([sequence])
        row = indices[0].tolist() if len(indices) else []
        return SurrogateVector(
            tuple(float(value) for value in row),
            "antbo_categorical_sequence_v1",
            source_id=candidate.candidate_id,
            metadata={"sequence": sequence},
        )


# ---------------------------------------------------------------------------
# Legacy run export
# ---------------------------------------------------------------------------


def materialize_legacy_run(
    runtime,
    result,
    run_dir: Any,
    *,
    antigen: str,
    seed: int,
    method: str,
    method_spec: Mapping[str, Any],
    args: Any,
    decision_recorder=None,
) -> dict[str, Any]:
    """Export results.csv and llm_acq_decisions.jsonl from engine events."""
    import pandas as pd

    events = runtime.events()
    rows = _result_rows(result, events)
    results_path = run_dir / "results.csv"
    if rows:
        pd.DataFrame(rows).to_csv(results_path, index=False)

    round_records = _decision_records(events, antigen, seed, method, method_spec, args)
    for record in round_records:
        if decision_recorder is not None:
            decision_recorder.append_round(record)

    legacy_summary = {
        "antigen": antigen,
        "seed": seed,
        "method": method,
        "n_evals": int(args.n_evals),
        "evaluated": len(rows),
        "best_value": rows[-1]["BestValue"] if rows else None,
        "best_sequence": rows[-1]["BestProtein"] if rows else None,
        "engine": result.summary,
    }
    summary_payload = dict(result.summary)
    summary_payload.update(legacy_summary)
    from ldm_tts.engine.run_store import atomic_json_write

    atomic_json_write(run_dir / "summary.json", summary_payload)
    return legacy_summary


def _result_rows(result, events) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_value: float | None = None
    best_sequence: str | None = None
    event_times = {
        event.get("candidate_id"): float(event.get("timestamp_unix", 0.0))
        for event in events
        if event.get("event_type") == "candidate_evaluated"
    }
    acquisition_by_sequence: dict[str, float] = {}
    for event in events:
        if event.get("event_type") != "candidates_selected":
            continue
        metadata = (event.get("payload") or {}).get("metadata") or {}
        for item in metadata.get("selected_candidates", []):
            if isinstance(item, Mapping) and item.get("sequence"):
                acquisition_by_sequence[str(item["sequence"])] = float(
                    item.get("acquisition_score", 0.0)
                )
    previous_time: float | None = None
    for observation in result.state.observations:
        payload = observation.candidate.payload
        value = observation.evaluation.metrics.get("absolut_energy")
        if value is None:
            continue
        if best_value is None or float(value) < best_value:
            best_value = float(value)
            best_sequence = payload.get("sequence")
        timestamp = event_times.get(observation.candidate_id)
        elapsed = (
            0.0
            if previous_time is None or timestamp is None
            else max(0.0, float(timestamp) - previous_time)
        )
        previous_time = timestamp if timestamp is not None else previous_time
        acquisition_score = payload.get("acquisition_score")
        if acquisition_score is None:
            acquisition_score = acquisition_by_sequence.get(str(payload.get("sequence")))
        rows.append({
            "Index": len(rows),
            "LastValue": float(value),
            "BestValue": best_value,
            "LLMScore": payload.get("score"),
            "AcquisitionScore": acquisition_score,
            "Time": elapsed,
            "LastProtein": payload.get("sequence"),
            "BestProtein": best_sequence,
            "Source": observation.candidate.metadata.get("selector_source", observation.candidate.source),
            "AcquisitionUsed": observation.candidate.metadata.get("acquisition_used", False),
        })
    return rows


def _decision_records(
    events: Sequence[Mapping[str, Any]],
    antigen: str,
    seed: int,
    method: str,
    method_spec: Mapping[str, Any],
    args: Any,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    evaluated_ids: list[str] = []
    selection_payload: dict[str, Any] = {}
    for event in events:
        event_type = event.get("event_type")
        payload = event.get("payload") or {}
        if event_type == "reservoir_expanded":
            if pending is not None:
                records.append(_finalize_decision(
                    pending, evaluated_ids, selection_payload,
                    antigen, seed, method, method_spec, args,
                ))
                evaluated_ids = []
                selection_payload = {}
            metadata = payload.get("metadata") or {}
            pending = {
                "eval_start": int(metadata.get("eval_start", 0)),
                "method": method,
                "parallel_budget": int(metadata.get("parallel_budget", args.parallel_budget)),
                "using_acquisition": bool(metadata.get("using_acquisition", False)),
                "selector_phase": metadata.get("selector_phase", ""),
                "decision": metadata.get("decision", {}),
                "candidates": metadata.get("decision", {}).get("candidates", []),
            }
            continue
        if event_type == "candidates_selected" and pending is not None:
            selection_payload = dict(payload)
            continue
        if event_type == "candidate_evaluated":
            evaluated_ids.append(str(event.get("candidate_id", "")))
            continue
    if pending is not None:
        records.append(_finalize_decision(
            pending, evaluated_ids, selection_payload,
            antigen, seed, method, method_spec, args,
        ))
    return records


def _finalize_decision(
    pending: dict[str, Any],
    evaluated_ids: list[str],
    selection_payload: Mapping[str, Any],
    antigen: str,
    seed: int,
    method: str,
    method_spec: Mapping[str, Any],
    args: Any,
) -> dict[str, Any]:
    selection_metadata = selection_payload.get("metadata") or {}
    using_acquisition = bool(pending.get("using_acquisition", False))
    decision = pending.get("decision") or {}
    selected_candidates = selection_metadata.get("selected_candidates", [])
    if not selected_candidates:
        selected_candidates = decision.get("candidates", [])[: len(evaluated_ids)]
    return {
        "eval_start": int(pending.get("eval_start", 0)),
        "eval_end": int(pending.get("eval_start", 0)) + len(evaluated_ids),
        "antigen": antigen,
        "seed": seed,
        "method": method,
        "parallel_budget": int(pending.get("parallel_budget", args.parallel_budget)),
        "candidates": decision.get("candidates", pending.get("candidates", [])),
        "acquisition": {
            "enabled": bool(method_spec.get("uses_acquisition", False)),
            "used": using_acquisition,
            "name": str(getattr(args, "acq", "lcb")).lower(),
            "beta": float(getattr(args, "acq_beta", 1.0)),
            "xi": float(getattr(args, "acq_xi", 0.001)),
            "n_init": int(args.n_init),
            "reduction": method_spec.get("reduction"),
            "softmax_eta": float(getattr(args, "softmax_eta", 1.0)),
            "parallel_executor": (
                "tasks.antibody.core.ldm.acquisition.parallel_search.execute_atoms"
                if using_acquisition and method_spec.get("base_measure") in {"policy", "legacy_policy"}
                else None
            ),
            "scores": selection_metadata.get(
                "candidates_scored", decision.get("candidates", [])
            ),
            "selected_indices": selection_metadata.get("selected_indices", []),
            "selected_candidates": selected_candidates,
        },
        "decision": decision,
        "pool_csv": None,
    }
