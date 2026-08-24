"""Independent structured-policy reservoir for antibody LDM variants."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ldm_tts.transport.parsing import load_json_object
from tasks.antibody.core.ldm.dsl.bias import BiasAtom
from tasks.antibody.core.ldm.dsl.sandbox import safe_exec_dsl
from tasks.antibody.core.ldm.dsl.search_space import (
    LatinHyperCubeSampling,
    LocalSearch,
    NeighborSampling,
    Or,
    SearchSpaceAtom,
)
from tasks.antibody.core.ldm.dsl.validator import validate_bias_atom, validate_search_atom
from tasks.antibody.core.ldm_light.selection import select_by_acquisition


POLICY_ATOMS = (
    "LatinHyperCubeSampling",
    "NeighborSampling",
    "LocalSearch",
    "Or",
    "MaxCysteine",
    "MaxHydrophobicRun",
    "MaxAromatic",
    "NetChargeRange",
    "NoNGlycosylation",
    "BiasSum",
)


@dataclass
class PolicyStrategy:
    atom: SearchSpaceAtom
    bias: BiasAtom | None
    rationale: str | None
    raw: str
    fallback: bool = False


def build_policy_prompt(
    *,
    antigen: str,
    rows: list[dict[str, Any]],
    antigen_context: dict[str, Any] | None,
    history_top_k: int,
    strategy_budget: int,
) -> str:
    history = [
        {
            "iteration": int(row["Index"]),
            "sequence": row["LastProtein"],
            "score": float(row["LastValue"]),
            "best_so_far": float(row["BestValue"]),
        }
        for row in rows[-int(history_top_k):]
    ]
    best = min(rows, key=lambda row: float(row["LastValue"])) if rows else None
    payload = {
        "task": "sample_one_antibody_search_policy",
        "objective": "Minimize Absolut binding energy; lower is better.",
        "antigen": antigen,
        "best": None if best is None else {
            "sequence": best["LastProtein"],
            "score": float(best["LastValue"]),
        },
        "history": history,
        "antigen_context": antigen_context or {},
        "policy_budget": int(strategy_budget),
        "allowed_examples": [
            f"LatinHyperCubeSampling(num={int(strategy_budget)})",
            f"NeighborSampling('ADGHTKQNPRA', radius=3, mut_pr=0.45, budget={int(strategy_budget)})",
            "LocalSearch('ADGHTKQNPRA', radius=3, restart=3, steps=39)",
        ],
        "required_output": {
            "rationale": "short search-scope rationale",
            "trust_region": "one DSL expression",
            "update_bias": "optional bias DSL expression or null",
        },
        "output_rules": [
            "Return JSON only.",
            "Return exactly one independently sampled search strategy.",
            "Do not return antibody sequences as the final proposal.",
        ],
    }
    return json.dumps(payload, indent=2)


def parse_policy_strategy(raw: str, *, sample_timeout_s: float) -> PolicyStrategy:
    obj = load_json_object(raw)
    source = obj.get("trust_region", obj.get("update_trust_region", obj.get("strategy")))
    if isinstance(source, dict):
        source = source.get("trust_region", source.get("dsl"))
    if not isinstance(source, str) or not source.strip():
        raise ValueError("policy response must contain one trust_region DSL string")

    atom = safe_exec_dsl(source, whitelist=POLICY_ATOMS, expect_kind=SearchSpaceAtom)
    errors = validate_search_atom(atom, sample_timeout_s=float(sample_timeout_s))
    if errors:
        raise ValueError(f"invalid policy strategy: {errors}")

    bias_source = obj.get("update_bias", obj.get("bias"))
    bias = None
    if bias_source:
        if not isinstance(bias_source, str):
            raise ValueError("update_bias must be a DSL string or null")
        bias = safe_exec_dsl(bias_source, whitelist=POLICY_ATOMS, expect_kind=BiasAtom)
        bias_errors = validate_bias_atom(bias)
        if bias_errors:
            raise ValueError(f"invalid policy bias: {bias_errors}")
    return PolicyStrategy(
        atom=atom,
        bias=bias,
        rationale=obj.get("rationale"),
        raw=raw,
    )


def cap_strategy_budget(atom: SearchSpaceAtom, budget: int) -> SearchSpaceAtom:
    budget = max(1, int(budget))
    if atom.budget <= budget:
        return atom
    if isinstance(atom, Or):
        children = atom.children[:budget]
        if len(children) == 1:
            return cap_strategy_budget(children[0], budget)
        child_budget, remainder = divmod(budget, len(children))
        return Or(*[
            cap_strategy_budget(child, child_budget + int(index < remainder))
            for index, child in enumerate(children)
        ])
    if isinstance(atom, LatinHyperCubeSampling):
        return LatinHyperCubeSampling(num=budget)
    if isinstance(atom, NeighborSampling):
        return NeighborSampling(
            atom.center,
            fixed=atom.fixed,
            radius=atom.radius,
            mut_pr=atom.mut_pr,
            budget=budget,
        )
    if isinstance(atom, LocalSearch):
        if budget < 2:
            return NeighborSampling(
                atom.center,
                fixed=atom.fixed,
                radius=atom.radius,
                budget=budget,
            )
        restart = min(int(atom.restart), max(1, budget // 2))
        steps = max(1, budget // restart - 1)
        return LocalSearch(
            atom.center,
            fixed=atom.fixed,
            radius=atom.radius,
            restart=restart,
            steps=steps,
        )
    return atom


def _fallback_policy(rows: list[dict[str, Any]], budget: int, index: int) -> PolicyStrategy:
    if not rows:
        atom: SearchSpaceAtom = LatinHyperCubeSampling(num=budget)
    else:
        best = min(rows, key=lambda row: float(row["LastValue"]))["LastProtein"]
        atom = NeighborSampling(
            best,
            radius=1 + index % 4,
            mut_pr=min(0.8, 0.3 + 0.1 * index),
            budget=budget,
        )
    return PolicyStrategy(
        atom=atom,
        bias=None,
        rationale="deterministic fallback policy",
        raw="",
        fallback=True,
    )


def propose_policy_reservoir(
    *,
    llm: Any,
    antigen: str,
    rows: list[dict[str, Any]],
    antigen_context: dict[str, Any] | None,
    strategy_budget: int,
    args: Any,
) -> tuple[list[PolicyStrategy], dict[str, Any]]:
    n_strategies = int(args.n_strategies)
    prompt = build_policy_prompt(
        antigen=antigen,
        rows=rows,
        antigen_context=antigen_context,
        history_top_k=int(args.history_top_k),
        strategy_budget=strategy_budget,
    )
    strategies: list[PolicyStrategy] = []
    errors: list[dict[str, Any]] = []
    raw_outputs: list[str] = []

    for attempt in range(1, max(1, int(args.max_retries)) + 1):
        needed = n_strategies - len(strategies)
        if needed <= 0:
            break
        if str(args.planner_mode) == "choices":
            outputs = llm.call_many(
                prompt,
                temperature=float(args.temperature),
                timeout_s=int(args.timeout_s),
                n=needed,
            )
        else:
            outputs = [
                llm.call(
                    prompt,
                    temperature=float(args.temperature),
                    timeout_s=int(args.timeout_s),
                )
                for _ in range(needed)
            ]
        raw_outputs.extend(outputs)
        for output_index, raw in enumerate(outputs):
            try:
                strategies.append(parse_policy_strategy(
                    raw,
                    sample_timeout_s=float(args.sample_timeout_s),
                ))
            except Exception as exc:
                errors.append({
                    "attempt": attempt,
                    "output_index": output_index,
                    "error": str(exc),
                    "raw_response": raw,
                })

    if len(strategies) < n_strategies and not bool(args.fallback_random):
        raise RuntimeError(json.dumps(errors, indent=2))
    while len(strategies) < n_strategies:
        strategies.append(_fallback_policy(rows, strategy_budget, len(strategies)))

    strategies = strategies[:n_strategies]
    for strategy in strategies:
        strategy.atom = cap_strategy_budget(strategy.atom, strategy_budget)
    return strategies, {
        "planner_mode": str(args.planner_mode),
        "n_requested": n_strategies,
        "n_fallback": sum(strategy.fallback for strategy in strategies),
        "prompt": prompt,
        "raw_outputs": raw_outputs,
        "errors": errors,
    }


def policy_representatives(
    records_by_strategy: Iterable[list[dict[str, Any]]],
    *,
    score_key: str,
) -> list[dict[str, Any]]:
    """Keep one unique highest-scoring representative per policy pool."""
    representatives: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for strategy_index, records in enumerate(records_by_strategy):
        ranked = sorted(
            records,
            key=lambda record: float(record.get(score_key, -np.inf)),
            reverse=True,
        )
        representative = next(
            (record for record in ranked if tuple(record["seq"]) not in seen),
            None,
        )
        if representative is None:
            continue
        item = dict(representative)
        item["strategy_index"] = strategy_index
        representatives.append(item)
        seen.add(tuple(item["seq"]))
    return representatives


def select_with_policy_reservoir(
    *,
    llm: Any,
    rows: list[dict[str, Any]],
    antigen: str,
    seed: int,
    iteration: int,
    antigen_context: dict[str, Any] | None,
    batch_size: int,
    reduction: str,
    args: Any,
    select_candidates: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    from tasks.antibody.core.ldm.acquisition.parallel_search import execute_atoms
    from tasks.antibody.core.ldm_light.ldm_acq import (
        AA,
        IDX_TO_AA,
        fit_gp_and_make_acquisition,
        passes_developability,
        seqs_to_indices,
    )

    n_strategies = int(args.n_strategies)
    configured_budget = int(getattr(args, "per_strategy_budget", 0))
    strategy_budget = configured_budget or max(1, int(args.parallel_budget) // n_strategies)
    strategies, planner_record = propose_policy_reservoir(
        llm=llm,
        antigen=antigen,
        rows=rows,
        antigen_context=antigen_context,
        strategy_budget=strategy_budget,
        args=args,
    )
    acq_name = str(args.acq).lower()
    device = torch.device(getattr(args, "device", "cpu") or "cpu")
    gp, acquisition = fit_gp_and_make_acquisition(
        rows,
        acq_name=acq_name,
        beta=float(args.acq_beta),
        xi=float(args.acq_xi),
        device=device,
    )

    records_by_strategy: list[list[dict[str, Any]]] = []
    strategy_records: list[dict[str, Any]] = []
    observed_keys = {
        tuple(int(value) for value in encoded)
        for encoded in seqs_to_indices([row["LastProtein"] for row in rows])
    }
    for strategy_index, strategy in enumerate(strategies):
        try:
            records = execute_atoms(
                search_dsl=strategy.atom,
                gp=gp,
                f_acq=acquisition,
                bias_dsl=strategy.bias,
                bias_weight=float(args.bias_weight),
                config=np.array([len(AA)] * len(rows[0]["LastProtein"]), dtype=int),
                cdr_constraints=True,
                rng=np.random.default_rng(int(seed) + iteration * 1009 + strategy_index),
                timeout_s=float(args.sample_timeout_s),
                device=device,
                acq_name=acq_name,
            )
            records = [
                record for record in records
                if tuple(int(value) for value in record["seq"]) not in observed_keys
                and passes_developability(
                    "".join(IDX_TO_AA[int(value)] for value in record["seq"])
                )
            ]
            error = None
        except Exception as exc:
            records = []
            error = f"{type(exc).__name__}: {exc}"
        for record in records:
            record["strategy_index"] = strategy_index
            record["strategy_atom"] = repr(strategy.atom)
        records_by_strategy.append(records)
        strategy_records.append({
            "strategy_index": strategy_index,
            "atom": repr(strategy.atom),
            "bias": repr(strategy.bias) if strategy.bias is not None else None,
            "rationale": strategy.rationale,
            "fallback": strategy.fallback,
            "n_candidates": len(records),
            "error": error,
        })

    pool_score_key = acq_name if str(args.pool_score) == "acq" else f"bias+{acq_name}"
    representatives = policy_representatives(records_by_strategy, score_key=pool_score_key)
    if not representatives:
        raise RuntimeError("Policy reservoir produced no valid candidates")

    selection_score_key = acq_name if str(args.selection_score) == "acq" else f"bias+{acq_name}"
    selected_indices: list[int] = []
    probabilities: list[float] = []
    if select_candidates:
        selected_indices, probabilities = select_by_acquisition(
            [record[selection_score_key] for record in representatives],
            batch_size=batch_size,
            reduction=reduction,
            eta=float(args.softmax_eta),
            rng=np.random.default_rng(int(seed) + iteration),
        )
    selected: list[dict[str, Any]] = []
    for index in selected_indices:
        record = representatives[index]
        selected.append({
            "sequence": "".join(IDX_TO_AA[int(value)] for value in record["seq"]),
            "score": None,
            "acquisition_score": float(record[selection_score_key]),
            "acquisition_raw": float(record[acq_name]),
            "mu": float(record["mu"]),
            "sigma": float(record["sigma"]),
            "source": f"policy_{record['strategy_index']}",
        })

    public_representatives = []
    for record in representatives:
        item = {key: value for key, value in record.items() if key != "seq"}
        item["sequence"] = "".join(IDX_TO_AA[int(value)] for value in record["seq"])
        public_representatives.append(item)
    return selected, {
        "source": f"policy_{reduction}",
        "planner": planner_record,
        "reduction": reduction,
        "softmax_eta": float(args.softmax_eta),
        "strategy_budget": strategy_budget,
        "pool_score": str(args.pool_score),
        "selection_score": str(args.selection_score),
        "strategy_records": strategy_records,
        "representatives": public_representatives,
        "selected_indices": selected_indices,
        "selection_probabilities": probabilities,
        "selected_candidates": selected,
    }
