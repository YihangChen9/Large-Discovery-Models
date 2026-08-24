"""Shared BO/SIR loop for tilted case2 M1 methods.

Deprecated as a campaign entrypoint: the small-molecule task now runs through
``ldm_tts.engine.LDMEngine`` with the adapters in
``tasks.small_molecule.core.engine_adapters``. This module retains the pure
scoring/resampling helpers used by those adapters and the legacy tests; do not
call :func:`run_tilted_case2_search` from new code.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from ldm_tts.engine.runtime import LDMSearchRoundResult, run_budgeted_search
from ldm_tts.contracts.evaluation import finite_or_none, is_finite_number
from ldm_tts.engine.run_store import load_jsonl
from tasks.small_molecule.core.ldm_tilted_case2.base_measure import q0_effective_support, q0_entropy
from tasks.small_molecule.core.ldm_tilted_case2.canonicalize import canonicalize_smiles
from tasks.small_molecule.core.ldm_tilted_case2.config import TiltedLDMCase2Config
from tasks.small_molecule.core.ldm_tilted_case2.ehvi_all import compute_ehvi_for_candidates
from tasks.small_molecule.core.ldm_tilted_case2.methods.direct_llm import DirectLLMReservoirBuilder
from tasks.small_molecule.core.ldm_tilted_case2.methods.llm_seed_analog import LLMSeedAnalogReservoirBuilder
from tasks.small_molecule.core.ldm_tilted_case2.pool_maintenance import maintain_candidate_pool
from tasks.small_molecule.core.ldm_tilted_case2.resampling import (
    effective_sample_size,
    gumbel_top_k,
    probability_entropy,
    robust_z,
    selected_rank_by_ehvi,
    tilted_logits,
    tilted_probabilities,
)
from tasks.small_molecule.core.ldm_tilted_case2.trace import TiltedTraceRecorder
from tasks.small_molecule.core.rng import RNG, as_rng
from tasks.small_molecule.core.scorer import Scorers, as_scorer_tuple

MAX_SCORE_ATTEMPTS = 2
MAX_ONE_STEP_EVALUATION_ATTEMPTS = 8
MAX_SELECTION_EVALUATION_ATTEMPTS = 8

LOGGER = logging.getLogger(__name__)


def run_tilted_case2_search(
    seed_smiles: Sequence[str],
    scorer: Scorers,
    analog_fn: Callable[[Sequence[str]], Sequence[str]],
    *,
    config: TiltedLDMCase2Config,
    llm,
    rng: Optional[RNG] = None,
) -> tuple[list[tuple[str, tuple[Optional[float], Optional[float]]]], dict | None]:
    rng_obj = as_rng(rng or RNG(config.seed))
    scorers = as_scorer_tuple(scorer)
    if len(scorers) != 2:
        raise ValueError("case2 requires exactly two scorers")
    resume_state = _load_resume_state(seed_smiles, scorers, config)
    if resume_state is None:
        history = _initial_history(seed_smiles, scorers, config)
        existing_rounds = None
    else:
        history, existing_rounds = resume_state
    recorder = TiltedTraceRecorder(
        config.trajectory_dir,
        config,
        existing_rounds=existing_rounds,
    )
    LOGGER.info(
        "tilted_case2 start method=%s init=%s history=%d/%d batch_size=%d allow_early_stop=%s trajectory=%s",
        config.method,
        config.init_strategy,
        len(history),
        config.budget,
        config.batch_size,
        config.allow_early_stop,
        config.trajectory_dir or "-",
    )
    loop_result = run_budgeted_search(
        history,
        budget=config.budget,
        build_round=lambda round_idx, round_history: _run_tilted_round(
            round_idx,
            round_history,
            config=config,
            llm=llm,
            analog_fn=analog_fn,
            scorers=scorers,
            rng_obj=rng_obj,
        ),
        record_round=recorder.record_round,
        on_empty_reservoir=lambda round_idx, empty_rounds: _log_empty_reservoir(
            round_idx,
            empty_rounds,
            config,
        ),
        start_round=len(recorder.rounds),
        max_empty_reservoir_rounds=config.max_empty_reservoir_rounds,
        allow_early_stop=config.allow_early_stop,
    )
    if (
        loop_result.empty_reservoir_rounds >= config.max_empty_reservoir_rounds
        and not config.allow_early_stop
    ):
        LOGGER.warning(
            "reservoir: empty limit reached, continuing was requested but the shared loop returned"
        )
    summary = recorder.finalize(history, early_stop_reason=loop_result.early_stop_reason)
    LOGGER.info(
        "tilted_case2 done history=%d/%d rounds=%d hv=%.6g early_stop=%s",
        len(history),
        config.budget,
        summary.get("round_count", len(recorder.rounds)),
        float(summary.get("final_hypervolume") or 0.0),
        loop_result.early_stop_reason,
    )
    return history, summary


def _run_tilted_round(
    round_idx: int,
    history,
    *,
    config: TiltedLDMCase2Config,
    llm,
    analog_fn,
    scorers,
    rng_obj,
) -> LDMSearchRoundResult:
    LOGGER.info(
        "round=%d history=%d/%d reservoir: building method=%s",
        round_idx,
        len(history),
        config.budget,
        config.method,
    )
    build_t0 = time.monotonic()
    build_result = _build_reservoir(history, config, llm, analog_fn, rng_obj)
    LOGGER.info(
        "round=%d reservoir: candidates=%d sources=%d drops=%s refills=%s elapsed=%.1fs",
        round_idx,
        len(build_result.candidates),
        len(build_result.sources),
        _compact_counts(build_result.drop_counts),
        build_result.metadata.get("refill_rounds", 0),
        time.monotonic() - build_t0,
    )
    if not build_result.candidates:
        return LDMSearchRoundResult(
            record=_empty_round(round_idx, config, build_result),
            empty_reservoir=True,
        )

    select_t0 = time.monotonic()
    if config.method in {"m1_stratified_direct_llm_only", "m1_llm_one_step"}:
        selected = _select_llm_order_and_score(build_result, scorers, config)
        if config.method == "m1_llm_one_step" and _has_nonfinite_scores(selected):
            build_result, selected = _retry_one_step_until_scored(
                history, config, llm, analog_fn, scorers, rng_obj, build_result, selected
            )
    else:
        selected = _select_and_score(build_result, history, scorers, config, rng_obj)
    LOGGER.info(
        "round=%d selection: mode=%s selected=%d smiles=%s scores=%s fallback=%s attempts=%s elapsed=%.1fs",
        round_idx,
        build_result.metadata.get("selection_mode", "ehvi_sir"),
        len(selected),
        _selected_smiles(selected),
        _selected_scores(selected),
        build_result.metadata.get("ehvi_fallback_reason"),
        build_result.metadata.get("selection_evaluation_attempts", len(selected)),
        time.monotonic() - select_t0,
    )
    if not selected and build_result.metadata.get("selection_failed_evaluations"):
        LOGGER.warning(
            "round=%d selection: all scoring attempts failed; first_failures=%s",
            round_idx,
            _compact_failed_for_log(build_result.metadata["selection_failed_evaluations"]),
        )
    history_delta = [
        (candidate.canonical_smiles or candidate.raw_smiles, tuple(candidate.true_scores))
        for candidate in selected
    ]
    LOGGER.info(
        "round=%d done history=%d/%d",
        round_idx,
        len(history) + len(history_delta),
        config.budget,
    )
    return LDMSearchRoundResult(
        history_delta=history_delta,
        record=_round_record(round_idx, config, build_result),
    )


def _log_empty_reservoir(round_idx: int, empty_rounds: int, config: TiltedLDMCase2Config) -> None:
    LOGGER.warning(
        "round=%d reservoir: empty empty_rounds=%d/%d",
        round_idx,
        empty_rounds,
        config.max_empty_reservoir_rounds,
    )
    if empty_rounds >= config.max_empty_reservoir_rounds and not config.allow_early_stop:
        LOGGER.warning(
            "round=%d reservoir: empty limit reached, continuing because allow_early_stop=False",
            round_idx,
        )


def _load_resume_state(seed_smiles, scorers, cfg):
    if not cfg.resume_from_trajectory:
        return None
    if not cfg.trajectory_dir:
        raise ValueError("resume_from_trajectory requires trajectory_dir")
    trajectory_dir = Path(cfg.trajectory_dir)
    rounds = _load_existing_rounds(trajectory_dir)
    history_path = trajectory_dir / "history.json"
    if history_path.exists():
        return _history_from_json(history_path), rounds
    history = _initial_history(seed_smiles, scorers, cfg)
    history.extend(_history_from_rounds(rounds))
    return history, rounds


def _load_existing_rounds(trajectory_dir: Path) -> list[dict]:
    return load_jsonl(trajectory_dir / "rounds.jsonl")


def _history_from_json(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        (str(row["smiles"]), tuple(row["scores"]))
        for row in rows
    ]


def _history_from_rounds(rounds: Sequence[dict]) -> list[tuple[str, tuple]]:
    history = []
    for record in rounds:
        selection = record.get("selection_results", {})
        smiles_list = selection.get("selected_smiles", [])
        scores_list = selection.get("selected_scores", [])
        for smiles, scores in zip(smiles_list, scores_list):
            history.append((str(smiles), tuple(scores)))
    return history


def _initial_history(seed_smiles, scorers, cfg):
    if cfg.init_strategy == "llm_cold_start":
        return []
    canonical = []
    seen = set()
    for smiles in seed_smiles:
        canon = canonicalize_smiles(smiles)
        if canon and canon not in seen:
            canonical.append(canon)
            seen.add(canon)
        if len(canonical) >= cfg.init_size:
            break
    scores = _score_smiles(canonical, scorers)
    return list(zip(canonical, scores))


def _retry_one_step_until_scored(history, cfg, llm, analog_fn, scorers, rng, build_result, selected):
    retry_history = list(history)
    failed = []
    current_result = build_result
    current_selected = selected
    for attempt in range(1, MAX_ONE_STEP_EVALUATION_ATTEMPTS):
        failed.extend(_failed_selected_rows(current_selected))
        if not failed:
            break
        retry_history = list(history) + failed
        current_result = _build_reservoir(retry_history, cfg, llm, analog_fn, rng)
        if not current_result.candidates:
            current_selected = []
            continue
        current_selected = _select_llm_order_and_score(current_result, scorers, cfg)
        if not _has_nonfinite_scores(current_selected):
            current_result.metadata["one_step_evaluation_refill_attempts"] = attempt
            current_result.metadata["one_step_failed_evaluations"] = _failed_metadata(failed)
            break
    if _has_nonfinite_scores(current_selected):
        return current_result, []
    return current_result, current_selected


def _failed_selected_rows(selected):
    rows = []
    for candidate in selected:
        if _finite_score_pair(candidate.true_scores):
            continue
        rows.append(
            (
                candidate.canonical_smiles or candidate.raw_smiles,
                tuple(candidate.true_scores or (None, None)),
            )
        )
    return rows


def _failed_metadata(failed):
    rows = []
    for item in failed:
        diagnostics = None
        if isinstance(item, dict):
            smiles = item.get("smiles")
            scores = item.get("scores", [])
            diagnostics = item.get("score_diagnostics")
        elif len(item) >= 3:
            smiles, scores, diagnostics = item[0], item[1], item[2]
        else:
            smiles, scores = item
        row = {"smiles": smiles, "scores": list(scores)}
        if diagnostics:
            row["score_diagnostics"] = diagnostics
        rows.append(row)
    return rows


def _has_nonfinite_scores(selected) -> bool:
    return any(not _finite_score_pair(candidate.true_scores) for candidate in selected)


def _finite_score_pair(scores) -> bool:
    return (
        scores is not None
        and len(scores) == 2
        and scores[0] is not None
        and scores[1] is not None
    )


def _build_reservoir(history, cfg, llm, analog_fn, rng):
    if cfg.method in {
        "m1_direct_llm_sir",
        "m1_stratified_direct_llm_sir",
        "m1_stratified_direct_llm_oversample_sir",
        "m1_stratified_direct_llm_only",
        "m1_llm_one_step",
    }:
        return DirectLLMReservoirBuilder().build(history, cfg, llm, rng)
    if cfg.method == "m1_llm_seed_analog_oversample_sir":
        return LLMSeedAnalogReservoirBuilder().build(history, cfg, llm, analog_fn, rng)
    raise ValueError(f"unsupported tilted case2 method: {cfg.method!r}")


def _select_and_score(build_result, history, scorers, cfg, rng):
    candidates, maintenance = maintain_candidate_pool(build_result.candidates, cfg, rng)
    build_result.candidates = candidates
    build_result.metadata["pool_maintenance"] = maintenance
    ehvi_result = compute_ehvi_for_candidates(history, candidates, cfg, rng)
    q0 = np.asarray([candidate.q0_base_mass for candidate in candidates], dtype=float)
    ehvi = ehvi_result.ehvi
    logits = tilted_logits(q0, ehvi, cfg)
    prob = tilted_probabilities(q0, ehvi, cfg)
    z = robust_z(ehvi, clip=cfg.z_clip, eps=cfg.eps)
    _write_selection_fields(candidates, logits, prob, z)
    selected, failed = _select_until_finite_scores(candidates, prob, scorers, cfg, rng)
    build_result.metadata["ehvi_fallback_reason"] = ehvi_result.fallback_reason
    build_result.metadata["acquisition_name"] = cfg.acquisition
    build_result.metadata["prob_entropy"] = probability_entropy(prob)
    build_result.metadata["prob_effective_sample_size"] = effective_sample_size(prob)
    build_result.metadata["selected_ehvi_rank"] = selected_rank_by_ehvi(candidates)
    build_result.metadata["selection_evaluation_attempts"] = len(failed) + len(selected)
    if failed:
        build_result.metadata["selection_failed_evaluations"] = _failed_metadata(failed)
    return selected


def _select_until_finite_scores(candidates, prob, scorers, cfg, rng):
    selected = []
    failed = []
    excluded_indices = set()
    while (
        len(selected) < cfg.batch_size
        and len(excluded_indices) < len(candidates)
        and len(failed) + len(selected) < MAX_SELECTION_EVALUATION_ATTEMPTS
    ):
        needed = cfg.batch_size - len(selected)
        indices = _sample_available_indices(prob, needed, excluded_indices, rng)
        if not indices:
            break
        smiles_batch = [
            candidates[idx].canonical_smiles or candidates[idx].raw_smiles for idx in indices
        ]
        scores, diagnostics = _score_smiles_with_diagnostics(smiles_batch, scorers)
        for local_idx, (idx, score_pair) in enumerate(zip(indices, scores)):
            candidate = candidates[idx]
            excluded_indices.add(idx)
            candidate.true_scores = list(score_pair)
            if _finite_score_pair(candidate.true_scores):
                candidate.selected = True
                selected.append(candidate)
                continue
            candidate.selected = False
            score_diagnostics = _meaningful_score_diagnostics(
                diagnostics[local_idx] if local_idx < len(diagnostics) else {}
            )
            failed.append(
                (
                    candidate.canonical_smiles or candidate.raw_smiles,
                    tuple(candidate.true_scores),
                    score_diagnostics,
                )
            )
            candidate.metadata["selection_failure_scores"] = candidate.true_scores
            if score_diagnostics:
                candidate.metadata["selection_failure_diagnostics"] = score_diagnostics
            candidate.true_scores = None
    return selected, failed


def _sample_available_indices(prob, k, excluded_indices, rng):
    available = [idx for idx in range(len(prob)) if idx not in excluded_indices]
    if not available or k <= 0:
        return []
    local_prob = np.asarray([prob[idx] for idx in available], dtype=float)
    local_indices = gumbel_top_k(local_prob, k, rng)
    return [available[idx] for idx in local_indices]


def _select_llm_order_and_score(build_result, scorers, cfg):
    candidates = build_result.candidates[: cfg.max_candidates_per_round]
    selected = candidates[: cfg.batch_size]
    if not selected:
        return []
    q0 = np.asarray([candidate.q0_base_mass for candidate in candidates], dtype=float)
    total = float(q0.sum())
    prob = q0 / total if total > 0.0 else np.full(len(candidates), 1.0 / len(candidates))
    for candidate, probability in zip(candidates, prob):
        candidate.resampling_probability = float(probability)
    scores = _score_smiles([c.canonical_smiles or c.raw_smiles for c in selected], scorers)
    for candidate, score_pair in zip(selected, scores):
        candidate.selected = True
        candidate.true_scores = list(score_pair)
    build_result.metadata["selection_mode"] = "llm_order"
    build_result.metadata["selected_llm_rank"] = list(range(1, len(selected) + 1))
    build_result.metadata["selected_ehvi_rank"] = []
    build_result.metadata["prob_entropy"] = probability_entropy(prob)
    build_result.metadata["prob_effective_sample_size"] = effective_sample_size(prob)
    return selected


def _write_selection_fields(candidates, logits, prob, z) -> None:
    for candidate, logit, probability, z_value in zip(candidates, logits, prob, z):
        candidate.log_weight = float(logit)
        candidate.resampling_probability = float(probability)
        candidate.ehvi_z = float(z_value)


def _score_smiles(smiles_list, scorers):
    scores, _diagnostics = _score_smiles_with_diagnostics(smiles_list, scorers)
    return scores


def _score_smiles_with_diagnostics(smiles_list, scorers):
    smiles_list = list(smiles_list)
    per_obj = []
    per_obj_diagnostics = []
    for scorer in scorers:
        values, diagnostics = _score_with_retries_with_diagnostics(smiles_list, scorer)
        per_obj.append([finite_or_none(value) for value in values])
        per_obj_diagnostics.append(diagnostics)
    scores = [tuple(values[i] for values in per_obj) for i in range(len(smiles_list))]
    diagnostics = [
        {
            "smiles": smiles,
            "objectives": [
                obj_diagnostics[i]
                for obj_diagnostics in per_obj_diagnostics
                if i < len(obj_diagnostics)
            ],
        }
        for i, smiles in enumerate(smiles_list)
    ]
    return scores, diagnostics


def _score_with_retries(smiles_list, scorer):
    values, _diagnostics = _score_with_retries_with_diagnostics(smiles_list, scorer)
    return values


def _score_with_retries_with_diagnostics(smiles_list, scorer):
    smiles_list = list(smiles_list)
    values, call_info = _call_scorer_with_info(smiles_list, scorer)
    diagnostics = [
        _score_diagnostic(value, _attempt_diagnostic(value, call_info, idx))
        for idx, value in enumerate(values)
    ]
    for _attempt in range(1, MAX_SCORE_ATTEMPTS):
        bad_indices = [
            idx for idx, value in enumerate(values) if not is_finite_number(value)
        ]
        if not bad_indices:
            break
        for idx in bad_indices:
            retry_values, retry_info = _call_scorer_with_info([smiles_list[idx]], scorer)
            diagnostics[idx]["attempts"].append(
                _attempt_diagnostic(
                    retry_values[0] if retry_values else float("nan"),
                    retry_info,
                    0,
                )
            )
            if retry_values and is_finite_number(retry_values[0]):
                values[idx] = retry_values[0]
            diagnostics[idx]["final_value"] = finite_or_none(values[idx])
            diagnostics[idx]["final_finite"] = is_finite_number(values[idx])
    return values, diagnostics


def _call_scorer(smiles_list, scorer):
    values, _call_info = _call_scorer_with_info(smiles_list, scorer)
    return values


def _call_scorer_with_info(smiles_list, scorer):
    smiles_list = list(smiles_list)
    try:
        values = list(scorer(smiles_list))
    except Exception as exc:
        raise RuntimeError(
            f"{_scorer_name(scorer)} failed while scoring {len(smiles_list)} "
            f"SMILES; sample={_sample_smiles_for_error(smiles_list)}: {exc}"
        ) from exc
    details = _scorer_last_results(scorer)
    if len(values) != len(smiles_list):
        raise RuntimeError(
            f"{_scorer_name(scorer)} returned {len(values)} values for "
            f"{len(smiles_list)} SMILES; sample={_sample_smiles_for_error(smiles_list)}"
        )
    return values, {"error": None, "details": details}


def _scorer_name(scorer) -> str:
    if hasattr(scorer, "__class__"):
        class_name = scorer.__class__.__name__
        if class_name and class_name != "function":
            return class_name
    return getattr(scorer, "__name__", type(scorer).__name__)


def _sample_smiles_for_error(smiles_list: Sequence[str], limit: int = 3) -> list[str]:
    return [_short_smiles(smiles) for smiles in smiles_list[:limit]]


def _scorer_last_results(scorer) -> list[Any]:
    details = getattr(scorer, "last_results", None)
    if not isinstance(details, list):
        return []
    return _json_safe(details)


def _score_diagnostic(value, first_attempt: dict[str, Any]) -> dict[str, Any]:
    return {
        "final_value": finite_or_none(value),
        "final_finite": is_finite_number(value),
        "attempts": [first_attempt],
    }


def _attempt_diagnostic(value, call_info: dict[str, Any], idx: int) -> dict[str, Any]:
    attempt = {
        "value": finite_or_none(value),
        "finite": is_finite_number(value),
    }
    if call_info.get("error"):
        attempt["error"] = call_info["error"]
    detail = _detail_at(call_info.get("details"), idx)
    if detail is not None:
        attempt["detail"] = detail
    return attempt


def _detail_at(details, idx: int):
    if not isinstance(details, list) or idx >= len(details):
        return None
    return details[idx]


def _meaningful_score_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any] | None:
    objectives = []
    for obj_idx, objective in enumerate(diagnostics.get("objectives", [])):
        attempts = objective.get("attempts", [])
        has_detail = any("detail" in attempt or "error" in attempt for attempt in attempts)
        if has_detail or not objective.get("final_finite", False):
            row = dict(objective)
            row["objective_index"] = obj_idx
            objectives.append(row)
    if not objectives:
        return None
    return {"objectives": objectives}


def _json_safe(value):
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


def _compact_counts(counts: dict) -> dict:
    return {str(key): int(value) for key, value in counts.items() if int(value) != 0}


def _compact_failed_for_log(rows: Sequence[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    compact = []
    for row in rows[:limit]:
        item = {
            "smiles": _short_smiles(row.get("smiles")),
            "scores": row.get("scores"),
        }
        reasons = _diagnostic_reasons(row.get("score_diagnostics"))
        if reasons:
            item["reasons"] = reasons
        compact.append(item)
    return compact


def _diagnostic_reasons(diagnostics: Any) -> list[str]:
    reasons = []
    if not isinstance(diagnostics, dict):
        return reasons
    for objective in diagnostics.get("objectives", []):
        obj_idx = objective.get("objective_index")
        for attempt in objective.get("attempts", []):
            if attempt.get("error"):
                reasons.append(f"obj{obj_idx}: {attempt['error']}")
                continue
            detail = attempt.get("detail")
            if isinstance(detail, dict):
                status = detail.get("status")
                message = detail.get("message")
                if status or message:
                    text = f"obj{obj_idx}: {status or 'failed'}"
                    if message:
                        text += f" ({str(message)[:160]})"
                    reasons.append(text)
                    continue
            if not attempt.get("finite", False):
                reasons.append(f"obj{obj_idx}: nonfinite")
    return reasons[:4]


def _selected_smiles(selected) -> list[str]:
    return [
        _short_smiles(candidate.canonical_smiles or candidate.raw_smiles)
        for candidate in selected
    ]


def _selected_scores(selected) -> list[list[float | None]]:
    return [list(candidate.true_scores or []) for candidate in selected]


def _short_smiles(smiles: str | None, max_len: int = 80) -> str:
    text = str(smiles or "")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _round_record(round_idx: int, cfg, build_result) -> dict:
    candidates = build_result.candidates
    prob = np.asarray([c.resampling_probability or 0.0 for c in candidates], dtype=float)
    q0 = np.asarray([c.q0_base_mass for c in candidates], dtype=float)
    return {
        "round_idx": round_idx,
        "method": cfg.method,
        "raw_llm_text": build_result.raw_llm_text,
        "parsed_llm_json": build_result.parsed_llm_json,
        "llm_attempts": build_result.llm_attempts,
        "sources": [source.to_dict() for source in build_result.sources],
        "drop_counts": build_result.drop_counts,
        "q0_entropy": q0_entropy(q0),
        "q0_effective_support": q0_effective_support(q0),
        "prob_entropy": probability_entropy(prob),
        "prob_effective_sample_size": effective_sample_size(prob),
        "selection_mode": build_result.metadata.get("selection_mode", "ehvi_sir"),
        "selected_llm_rank": build_result.metadata.get("selected_llm_rank", []),
        "selected_ehvi_rank": build_result.metadata.get(
            "selected_ehvi_rank", selected_rank_by_ehvi(candidates)
        ),
        "fallback_reason": build_result.metadata.get("ehvi_fallback_reason"),
        "selection_results": _selection_results(candidates, build_result),
        "pool_maintenance": build_result.metadata.get("pool_maintenance", {}),
        "candidates": [candidate.to_dict() for candidate in candidates],
    }


def _selection_results(candidates, build_result) -> dict:
    selected = [candidate for candidate in candidates if candidate.selected]
    return {
        "selected_smiles": [
            candidate.canonical_smiles or candidate.raw_smiles
            for candidate in selected
        ],
        "selected_scores": [candidate.true_scores for candidate in selected],
        "selected_probabilities": [
            candidate.resampling_probability for candidate in selected
        ],
        "selected_ehvi": [candidate.ehvi for candidate in selected],
        "ehvi_fallback_reason": build_result.metadata.get("ehvi_fallback_reason"),
        "selection_evaluation_attempts": build_result.metadata.get(
            "selection_evaluation_attempts"
        ),
        "failed_evaluations": build_result.metadata.get(
            "selection_failed_evaluations", []
        ),
    }


def _empty_round(round_idx: int, cfg, build_result) -> dict:
    record = _round_record(round_idx, cfg, build_result)
    record["fallback_reason"] = "empty_reservoir"
    return record
