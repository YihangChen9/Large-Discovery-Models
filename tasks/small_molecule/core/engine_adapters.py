"""LDMEngine behavioral adapters for the small-molecule task.

This module turns the tilted case2 scientific components into the task-owned
adapter seams required by :class:`ldm_tts.engine.LDMEngine`:

* :class:`SmilesCandidateDomain` -- SMILES canonicalization and hard filters
  (``CandidateDomainAdapter``).
* :class:`SmilesReservoirExpander` -- LLM reservoir expansion wrapping the
  existing ``DirectLLMReservoirBuilder`` / ``LLMSeedAnalogReservoirBuilder``,
  with the refill loop and q0-gumbel pool maintenance folded inside
  (``ReservoirExpander``).
* :class:`SmilesCandidateEvaluator` -- Vina + activity scoring per candidate
  (``CandidateEvaluator``).
* :class:`TiltedAcquisitionSelector` -- EHVI/mean posterior scores tilted by the
  q0 base measure and sampled with gumbel top-k (``AcquisitionSelector``).
* :class:`SmilesSurrogateEncoder` -- molecular fingerprint / SMILES string
  kernel representations (``SurrogateEncoder``).

:func:`materialize_legacy_trajectory` exports the engine events back into the
legacy trajectory files (``rounds.jsonl`` / ``history.json`` / legacy summary
fields) so downstream tooling keeps working.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from ldm_tts.contracts import (
    AcquisitionSpec,
    Candidate,
    CandidateRejection,
    EvaluationResult,
    Observation,
    RawProposal,
    SurrogateSpaceSpec,
)
from ldm_tts.engine.expansion import ExpansionRequest, ExpansionResult
from ldm_tts.optimization.records import (
    BOObservation,
    BOSelectionResult,
    SurrogateVector,
)
from ldm_tts.transport import ProposalResponse

from tasks.small_molecule.core.ldm_tilted_case2.base_measure import (
    apply_m1_base_measure,
    q0_effective_support,
    q0_entropy,
)
from tasks.small_molecule.core.ldm_tilted_case2.candidate_record import CandidateRecord
from tasks.small_molecule.core.ldm_tilted_case2.canonicalize import canonicalize_smiles
from tasks.small_molecule.core.ldm_tilted_case2.config import TiltedLDMCase2Config
from tasks.small_molecule.core.ldm_tilted_case2.ehvi_all import compute_ehvi_for_candidates
from tasks.small_molecule.core.ldm_tilted_case2.loop import _score_smiles_with_diagnostics
from tasks.small_molecule.core.ldm_tilted_case2.methods.direct_llm import (
    DirectLLMReservoirBuilder,
)
from tasks.small_molecule.core.ldm_tilted_case2.methods.llm_seed_analog import (
    LLMSeedAnalogReservoirBuilder,
)
from tasks.small_molecule.core.ldm_tilted_case2.pool_maintenance import (
    maintain_candidate_pool,
)
from tasks.small_molecule.core.ldm_tilted_case2.resampling import (
    effective_sample_size,
    gumbel_top_k,
    probability_entropy,
    robust_z,
    selected_rank_by_ehvi,
    tilted_logits,
    tilted_probabilities,
)
from tasks.small_molecule.core.rng import RNG, as_rng

DIRECT_ONLY_METHODS = {"m1_stratified_direct_llm_only", "m1_llm_one_step"}


# ---------------------------------------------------------------------------
# Candidate domain
# ---------------------------------------------------------------------------


class SmilesCandidateDomain:
    """Admit raw SMILES proposals as canonical molecule candidates."""

    def __init__(self, cfg: TiltedLDMCase2Config) -> None:
        self.cfg = cfg

    def admit(self, proposal: RawProposal) -> Candidate | CandidateRejection:
        payload = proposal.payload
        if isinstance(payload, str):
            smiles, rationale = payload, ""
        elif isinstance(payload, Mapping):
            smiles = payload.get("smiles")
            rationale = str(payload.get("rationale", ""))
        else:
            smiles = None
            rationale = ""
        raw_text = str(smiles or "").strip()
        if not raw_text:
            return CandidateRejection(
                "invalid",
                "proposal payload must contain a non-empty SMILES string",
                proposal.source,
            )
        canonical = canonicalize_smiles(raw_text)
        if canonical is None:
            return CandidateRejection(
                "invalid",
                "SMILES could not be canonicalized",
                proposal.source,
                metadata={"smiles": raw_text[:200]},
            )
        if len(canonical) > self.cfg.smiles_max_len:
            return CandidateRejection(
                "overlength",
                f"SMILES exceeds {self.cfg.smiles_max_len} characters",
                proposal.source,
                metadata={"smiles": canonical[:200]},
            )
        metadata: dict[str, Any] = {
            "rationale": rationale,
            "occurrence_by_source": dict(
                proposal.metadata.get("occurrence_by_source", {}) or {}
            ),
            "base_support_level": proposal.metadata.get("base_support_level"),
            "base_support_value": float(proposal.metadata.get("base_support_value", 0.0)),
            "q0_base_mass": float(proposal.metadata.get("q0_base_mass", 0.0)),
        }
        return Candidate(
            candidate_id="mol-" + hashlib.sha256(canonical.encode()).hexdigest()[:12],
            payload={"smiles": canonical, "rationale": rationale},
            canonical_key=canonical,
            source=proposal.source,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Reservoir expander
# ---------------------------------------------------------------------------


class SmilesReservoirExpander:
    """Expand the molecule reservoir through the tilted M1 LLM builders."""

    def __init__(
        self,
        cfg: TiltedLDMCase2Config,
        llm,
        analog_fn,
        *,
        rng: Optional[RNG] = None,
        budget_hook: Optional[Callable[[str, int], Any]] = None,
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.analog_fn = analog_fn
        self.rng = as_rng(rng or RNG(cfg.seed))
        self.budget_hook = budget_hook

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        history = _history_rows_from_observations(request.observations)
        if self.cfg.method == "m1_llm_seed_analog_oversample_sir":
            build_result = LLMSeedAnalogReservoirBuilder().build(
                history, self.cfg, self.llm, self.analog_fn, self.rng
            )
        else:
            build_result = DirectLLMReservoirBuilder().build(
                history, self.cfg, self.llm, self.rng
            )

        candidates = list(build_result.candidates)
        pool_maintenance: dict[str, Any] = {}
        if self.cfg.method not in DIRECT_ONLY_METHODS:
            candidates, pool_maintenance = maintain_candidate_pool(candidates, self.cfg, self.rng)
        evaluated = {row[0] for row in history}
        proposals = tuple(
            _candidate_to_proposal(candidate)
            for candidate in candidates
            if (candidate.canonical_smiles or candidate.raw_smiles) not in evaluated
        )
        attempts = tuple(
            _attempt_to_proposal_response(attempt)
            for attempt in build_result.llm_attempts
        )
        if self.budget_hook is not None:
            self.budget_hook("llm_requests", len(build_result.llm_attempts))
            self.budget_hook("proposal_attempts", len(build_result.llm_attempts))
        if not proposals:
            # Keep engine empty-reservoir accounting alive: emit a placeholder
            # proposal the domain adapter will reject.
            proposals = (RawProposal(None, "empty_reservoir"),)
        return ExpansionResult(
            proposals=proposals,
            attempts=attempts,
            metadata=_expansion_metadata(build_result, pool_maintenance, history),
        )


def _candidate_to_proposal(candidate: CandidateRecord) -> RawProposal:
    return RawProposal(
        {
            "smiles": candidate.canonical_smiles or candidate.raw_smiles,
            "rationale": candidate.metadata.get("rationale", ""),
        },
        candidate.sources[0] if candidate.sources else "direct_llm",
        metadata={
            "occurrence_by_source": dict(candidate.occurrence_by_source),
            "base_support_level": candidate.base_support_level,
            "base_support_value": float(candidate.base_support_value),
            "q0_base_mass": float(candidate.q0_base_mass),
            "canonical_smiles": candidate.canonical_smiles,
        },
    )


def _attempt_to_proposal_response(attempt: Mapping[str, Any]) -> ProposalResponse:
    text = attempt.get("raw_text") or attempt.get("raw_output") or "(attempt failed)"
    return ProposalResponse(
        text=str(text),
        metadata={
            key: attempt.get(key)
            for key in ("stage", "source_id", "error", "user_prompt")
            if attempt.get(key) is not None
        },
    )


def _expansion_metadata(build_result, pool_maintenance, history) -> dict[str, Any]:
    candidates = list(build_result.candidates)
    q0 = np.asarray([candidate.q0_base_mass for candidate in candidates], dtype=float)
    return {
        "phase": "llm_expansion",
        "raw_llm_text": build_result.raw_llm_text,
        "parsed_llm_json": build_result.parsed_llm_json,
        "llm_attempts": list(build_result.llm_attempts),
        "sources": [source.to_dict() for source in build_result.sources],
        "drop_counts": dict(build_result.drop_counts),
        "pool_maintenance": pool_maintenance,
        "refill_rounds": build_result.metadata.get("refill_rounds", 0),
        "q0_entropy": q0_entropy(q0),
        "q0_effective_support": q0_effective_support(q0),
        "history_size": len(history),
        "candidates": [candidate.to_dict() for candidate in candidates],
    }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class SmilesCandidateEvaluator:
    """Score one candidate with the Vina and activity objectives."""

    def __init__(self, vina_scorer, activity_scorer) -> None:
        self.scorers = (vina_scorer, activity_scorer)

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        smiles = candidate.payload["smiles"]
        scores, diagnostics = _score_smiles_with_diagnostics([smiles], self.scorers)
        pair = scores[0] if scores else (None, None)
        vina, activity = pair if len(pair) == 2 else (None, None)
        partial_metrics: dict[str, float] = {}
        if vina is not None:
            partial_metrics["vina"] = float(vina)
        if activity is not None:
            partial_metrics["activity"] = float(activity)
        if vina is None or activity is None:
            return EvaluationResult(
                candidate.candidate_id,
                "failed",
                metrics=partial_metrics,
                error="non-finite objective score after retries",
                metadata={"smiles": smiles, "diagnostics": diagnostics},
            )
        return EvaluationResult(
            candidate.candidate_id,
            "succeeded",
            metrics={"vina": float(vina), "activity": float(activity)},
            resource_usage={"benchmark_jobs": 1},
            metadata={"smiles": smiles, "diagnostics": diagnostics},
        )


# ---------------------------------------------------------------------------
# Acquisition selector
# ---------------------------------------------------------------------------


class TiltedAcquisitionSelector:
    """EHVI/mean posterior tilt sampling over the q0 base measure.

    ``fit`` receives engine BO observations whose surrogate features carry the
    canonical SMILES (see :class:`SmilesSurrogateEncoder`), so the GP history
    keeps its ``(smiles, (vina, activity))`` shape.
    """

    def __init__(self, cfg: TiltedLDMCase2Config, rng: Optional[RNG] = None) -> None:
        self.cfg = cfg
        self.rng = as_rng(rng or RNG(cfg.seed))
        self.history: list[tuple[str, tuple[float, float]]] = []

    def describe(self) -> AcquisitionSpec:
        return AcquisitionSpec(
            name=self.cfg.acquisition,
            objective_names=("vina", "activity"),
            score_direction="sample",
            selection_rule=(
                "sample candidates from the q0 base mass tilted by the robust-z "
                f"{self.cfg.acquisition} acquisition score"
            ),
            parameters={
                "alpha_base_measure": float(self.cfg.alpha_base_measure),
                "eta_acquisition_tilt": float(self.cfg.eta_ehvi_tilt),
                "acquisition_weights": list(self.cfg.acquisition_weights),
                "ehvi_n_samples": int(self.cfg.ehvi_n_samples),
            },
        )

    def fit(self, history: Sequence[BOObservation]) -> None:
        rows: list[tuple[str, tuple[float, float]]] = []
        for observation in history:
            smiles = _smiles_from_bo_observation(observation)
            if smiles is None or len(observation.objectives) != 2:
                continue
            first, second = observation.objectives
            if first is None or second is None:
                continue
            rows.append((smiles, (float(first), float(second))))
        self.history = rows

    def select(
        self,
        candidates: Sequence[Candidate],
        representations: Mapping[str, SurrogateVector],
        *,
        count: int = 1,
    ) -> BOSelectionResult:
        del representations  # SMILES kernel/features are recomputed from payloads.
        records = [_record_from_candidate(candidate, self.cfg) for candidate in candidates]
        if not any(record.q0_base_mass > 0 for record in records):
            apply_m1_base_measure(records, smoothing=self.cfg.m1_q0_smoothing)
        ehvi_result = compute_ehvi_for_candidates(self.history, records, self.cfg, self.rng)
        q0 = np.asarray([record.q0_base_mass for record in records], dtype=float)
        ehvi = ehvi_result.ehvi
        logits = tilted_logits(q0, ehvi, self.cfg)
        prob = tilted_probabilities(q0, ehvi, self.cfg)
        z = robust_z(ehvi, clip=self.cfg.z_clip, eps=self.cfg.eps)
        for record, logit, probability, z_value in zip(records, logits, prob, z):
            record.log_weight = float(logit)
            record.resampling_probability = float(probability)
            record.ehvi_z = float(z_value)
        needed = min(int(count), len(records))
        indices = gumbel_top_k(prob, needed, self.rng) if needed > 0 else []
        selected_ids = tuple(candidates[index].candidate_id for index in indices)
        return BOSelectionResult(
            selected_candidate_ids=selected_ids,
            fallback_reason=ehvi_result.fallback_reason,
            metadata={
                "selection_mode": "ehvi_sir",
                "acquisition_name": self.cfg.acquisition,
                "prob_entropy": probability_entropy(prob),
                "prob_effective_sample_size": effective_sample_size(prob),
                "selected_ehvi_rank": selected_rank_by_ehvi(records),
                "ehvi_fallback_reason": ehvi_result.fallback_reason,
                "selected_probabilities": [float(prob[index]) for index in indices],
                "selected_ehvi": [float(ehvi[index]) for index in indices],
                "selected_logits": [float(logits[index]) for index in indices],
            },
        )


def _smiles_from_bo_observation(observation: BOObservation) -> Optional[str]:
    feature = observation.feature
    if feature is not None and feature.metadata.get("smiles"):
        return str(feature.metadata["smiles"])
    return None


def _record_from_candidate(candidate: Candidate, cfg: TiltedLDMCase2Config) -> CandidateRecord:
    metadata = dict(candidate.metadata)
    occurrence = dict(metadata.get("occurrence_by_source", {}) or {})
    if not occurrence:
        occurrence = {candidate.source or "reservoir": 1}
    return CandidateRecord(
        raw_smiles=candidate.payload["smiles"],
        canonical_smiles=candidate.payload["smiles"],
        method=cfg.method,
        sources=list(occurrence),
        occurrence_by_source=occurrence,
        base_support_level=metadata.get("base_support_level"),
        base_support_value=float(metadata.get("base_support_value", 0.0)),
        q0_base_mass=float(metadata.get("q0_base_mass", 0.0)),
        metadata=dict(metadata),
    )


# ---------------------------------------------------------------------------
# Surrogate encoder
# ---------------------------------------------------------------------------


class SmilesSurrogateEncoder:
    """Molecular fingerprint or implicit SMILES string-kernel encoder."""

    def __init__(self, gp_config) -> None:
        self.gp_config = gp_config

    def describe(self) -> SurrogateSpaceSpec:
        if self.gp_config.impl == "smiles-strkernel":
            return SurrogateSpaceSpec(
                kind="kernel",
                representation="SMILES subsequence string kernel",
                dimension_policy="implicit",
                encoder="tasks.small_molecule.core.engine_adapters.SmilesSurrogateEncoder",
                version="smiles_strkernel_v1",
            )
        return SurrogateSpaceSpec(
            kind="vector",
            representation="fixed-length molecular fingerprint",
            dimension_policy="fixed",
            dimension=int(self.gp_config.fp_n_bits),
            encoder="tasks.small_molecule.core.engine_adapters.SmilesSurrogateEncoder",
            version=f"molecular_fingerprint_{int(self.gp_config.fp_n_bits)}_v1",
        )

    def encode(self, candidate: Candidate) -> SurrogateVector:
        smiles = candidate.payload["smiles"]
        if self.gp_config.impl == "smiles-strkernel":
            # The string kernel is computed lazily by the GP from the SMILES
            # payload; there is no standalone fixed vector representation.
            return SurrogateVector(
                (),
                "smiles_strkernel_v1",
                source_id=candidate.candidate_id,
                metadata={"smiles": smiles},
            )
        from tasks.small_molecule.core.gp import _smiles_to_fingerprints

        features = _smiles_to_fingerprints(
            [smiles],
            radius=self.gp_config.fp_radius,
            n_bits=self.gp_config.fp_n_bits,
        )
        row = features[0].tolist() if len(features) else []
        return SurrogateVector(
            tuple(float(value) for value in row),
            f"molecular_fingerprint_{int(self.gp_config.fp_n_bits)}_v1",
            source_id=candidate.candidate_id,
            metadata={"smiles": smiles},
        )


# ---------------------------------------------------------------------------
# History conversion helpers
# ---------------------------------------------------------------------------


def _history_rows_from_observations(observations) -> list[tuple[str, tuple]]:
    rows: list[tuple[str, tuple]] = []
    for observation in observations:
        payload = observation.candidate.payload
        smiles = payload.get("smiles") if isinstance(payload, Mapping) else None
        if not smiles:
            continue
        metrics = observation.evaluation.metrics
        rows.append((str(smiles), (metrics.get("vina"), metrics.get("activity"))))
    return rows


def observations_from_history_rows(history: Sequence[tuple[str, Sequence]]) -> list[Observation]:
    """Rebuild engine observations from a legacy history.json payload."""
    observations: list[Observation] = []
    for smiles, scores in history:
        if len(scores) != 2 or scores[0] is None or scores[1] is None:
            continue
        candidate = Candidate(
            candidate_id="mol-" + hashlib.sha256(str(smiles).encode()).hexdigest()[:12],
            payload={"smiles": str(smiles), "rationale": ""},
            canonical_key=str(smiles),
            source="legacy_resume",
        )
        evaluation = EvaluationResult(
            candidate.candidate_id,
            "succeeded",
            metrics={"vina": float(scores[0]), "activity": float(scores[1])},
        )
        observations.append(Observation(candidate=candidate, evaluation=evaluation))
    return observations


# ---------------------------------------------------------------------------
# Legacy trajectory export
# ---------------------------------------------------------------------------


def materialize_legacy_trajectory(
    runtime,
    result,
    cfg: TiltedLDMCase2Config,
    *,
    sink=None,
) -> dict[str, Any]:
    """Export engine events back into the legacy tilted-case2 files."""
    from tasks.small_molecule.core.acquisition import hypervolume

    events = runtime.events()
    rounds = _rounds_from_events(events, cfg)
    history = [
        (observation.candidate.payload["smiles"], (
            observation.evaluation.metrics.get("vina"),
            observation.evaluation.metrics.get("activity"),
        ))
        for observation in result.state.observations
    ]
    run_dir = runtime.run_dir
    if rounds:
        lines = "\n".join(
            json.dumps(record, sort_keys=True) for record in rounds
        ) + "\n"
        (run_dir / "rounds.jsonl").write_text(lines, encoding="utf-8")
    history_json = [
        {"smiles": smiles, "scores": list(scores)}
        for smiles, scores in history
    ]
    (run_dir / "history.json").write_text(
        json.dumps(history_json, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    finite_points = [
        scores for _smiles, scores in history
        if len(scores) == 2 and None not in scores
    ]
    drop_totals: dict[str, int] = {}
    llm_call_count = 0
    for record in rounds:
        llm_call_count += len(record.get("llm_attempts", []))
        for key, value in record.get("drop_counts", {}).items():
            drop_totals[key] = drop_totals.get(key, 0) + int(value)
    legacy_summary = {
        "method": cfg.method,
        "llm_call_count": llm_call_count,
        "round_count": len(rounds),
        "history_size": len(history),
        "final_hypervolume": (
            float(hypervolume(finite_points, cfg.ref_point, minimize=cfg.minimize))
            if finite_points
            else 0.0
        ),
        "drop_counts": drop_totals,
        "early_stop_reason": result.stop_reason,
        "q0_entropy": _last_round_metric(rounds, "q0_entropy"),
        "prob_effective_sample_size": _last_round_metric(
            rounds, "prob_effective_sample_size"
        ),
        "engine": result.summary,
    }
    summary_payload = dict(result.summary)
    summary_payload.update(legacy_summary)
    from ldm_tts.engine.run_store import atomic_json_write

    atomic_json_write(run_dir / "summary.json", summary_payload)

    if sink is not None and getattr(sink, "enabled", False):
        from ldm_tts.data import smallmol_irs_from_round_record

        for record in rounds:
            provenance = {
                "task": "small_molecule",
                "method": cfg.method,
                "round_idx": record.get("round_idx"),
                "trajectory_dir": str(run_dir),
            }
            outcome = {
                "selection_results": record.get("selection_results", {}),
                "drop_counts": record.get("drop_counts", {}),
            }
            for ir in smallmol_irs_from_round_record(record):
                sink.append(ir, provenance=provenance, outcome=outcome)
    return legacy_summary


def _rounds_from_events(
    events: Sequence[Mapping[str, Any]], cfg: TiltedLDMCase2Config
) -> list[dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    evaluated: list[dict[str, Any]] = []
    initialization_round = False
    for event in events:
        event_type = event.get("event_type")
        payload = event.get("payload") or {}
        if event_type == "reservoir_expanded":
            if pending is not None:
                _finalize_round(pending, evaluated, cfg)
                rounds.append(pending)
                evaluated = []
            metadata = payload.get("metadata") or {}
            initialization_round = metadata.get("phase") == "initialization"
            if initialization_round:
                pending = None
                evaluated = []
                continue
            pending = {
                "round_idx": event.get("iteration"),
                "method": cfg.method,
                "raw_llm_text": metadata.get("raw_llm_text", ""),
                "parsed_llm_json": metadata.get("parsed_llm_json", {}),
                "llm_attempts": metadata.get("llm_attempts", []),
                "sources": metadata.get("sources", []),
                "drop_counts": metadata.get("drop_counts", {}),
                "q0_entropy": metadata.get("q0_entropy"),
                "q0_effective_support": metadata.get("q0_effective_support"),
                "pool_maintenance": metadata.get("pool_maintenance", {}),
                "candidates": metadata.get("candidates", []),
                "phase": metadata.get("phase", "llm_expansion"),
            }
            continue
        if event_type == "reservoir_built" and pending is not None:
            engine_drops = payload.get("drop_counts", {})
            merged_drops = dict(pending.get("drop_counts", {}))
            for key, value in engine_drops.items():
                legacy_key = {
                    "already_evaluated": "evaluated",
                    "reservoir_capacity": "capacity",
                }.get(key, key)
                merged_drops[legacy_key] = merged_drops.get(legacy_key, 0) + int(value)
            pending["drop_counts"] = merged_drops
            continue
        if event_type == "candidates_selected" and pending is not None:
            metadata = payload.get("metadata") or {}
            pending["fallback_reason"] = (
                payload.get("fallback_reason") or metadata.get("ehvi_fallback_reason")
            )
            pending["prob_entropy"] = metadata.get("prob_entropy")
            pending["prob_effective_sample_size"] = metadata.get(
                "prob_effective_sample_size"
            )
            pending["selected_ehvi_rank"] = metadata.get("selected_ehvi_rank", [])
            selection_mode = metadata.get("selection_mode")
            if selection_mode is None:
                selection_mode = (
                    "llm_order" if metadata.get("mode") == "reservoir_order" else "ehvi_sir"
                )
            pending["selection_mode"] = selection_mode
            pending["selection_metadata"] = metadata
            pending["selected_candidate_ids"] = list(payload.get("selected_candidate_ids", []))
            continue
        if event_type == "candidate_evaluated":
            if initialization_round:
                continue
            observation = payload.get("candidate") or {}
            evaluation = payload.get("evaluation") or {}
            smiles = (
                observation.get("payload", {}).get("smiles")
                if isinstance(observation.get("payload"), dict)
                else None
            )
            evaluated.append({
                "candidate_id": observation.get("candidate_id", ""),
                "smiles": smiles,
                "scores": [
                    evaluation.get("metrics", {}).get("vina"),
                    evaluation.get("metrics", {}).get("activity"),
                ],
                "status": evaluation.get("status"),
            })
            continue
    if pending is not None:
        _finalize_round(pending, evaluated, cfg)
        rounds.append(pending)
    return rounds


def _finalize_round(
    record: dict[str, Any], evaluated: list[dict[str, Any]], cfg: TiltedLDMCase2Config
) -> None:
    del cfg
    selection_metadata = record.pop("selection_metadata", {})
    selected_ids = record.pop("selected_candidate_ids", [])
    if selected_ids:
        by_id = {item["candidate_id"]: item for item in evaluated}
        selected_rows = [
            by_id[candidate_id] for candidate_id in selected_ids if candidate_id in by_id
        ]
    else:
        selected_rows = evaluated
    failed = [
        {"smiles": item["smiles"], "scores": item["scores"]}
        for item in evaluated
        if item["status"] != "succeeded"
    ]
    record["selection_results"] = {
        "selected_smiles": [item["smiles"] for item in selected_rows],
        "selected_scores": [item["scores"] for item in selected_rows],
        "selected_probabilities": selection_metadata.get(
            "selected_probabilities", []
        )[: len(selected_rows)],
        "selected_ehvi": selection_metadata.get("selected_ehvi", [])[
            : len(selected_rows)
        ],
        "ehvi_fallback_reason": selection_metadata.get("ehvi_fallback_reason"),
        "selection_evaluation_attempts": len(evaluated),
        "failed_evaluations": failed,
    }
    if record.get("selection_mode") == "llm_order":
        record["selected_llm_rank"] = list(range(1, len(selected_rows) + 1))
    else:
        record["selected_llm_rank"] = []


def _last_round_metric(rounds: Sequence[Mapping[str, Any]], key: str) -> Any:
    if not rounds:
        return None
    return rounds[-1].get(key)
