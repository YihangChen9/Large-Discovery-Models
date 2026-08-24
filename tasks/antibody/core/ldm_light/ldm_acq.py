#!/usr/bin/env python3
"""Antibody campaign through the shared LDM-TTS engine.

The warmup/proposal/acquisition/scoring components remain below (direct batch
proposal, policy reservoirs, GP acquisition, Absolut evaluation), but the
campaign loop now runs through ``ldm_tts.engine.LDMEngine`` with the adapters
in ``tasks.antibody.core.engine_adapters``. ``run_one`` assembles one engine
campaign per (antigen, seed) and re-exports ``results.csv`` and
``llm_acq_decisions.jsonl`` from the engine events.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

def find_repo_root(start: Path) -> Path:
    for path in [start.parent, *start.parents]:
        if (path / "core").is_dir() and (path / "core" / "__init__.py").exists():
            return path
    raise RuntimeError(f"Could not find AntBO repo root from {start}")


ROOT = find_repo_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WORKSPACE_ROOT = ROOT.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ldm_tts.campaign import (
    CampaignBudget,
    CampaignRecipe,
    CampaignRequest,
    run_campaign,
)
from ldm_tts.engine.run_store import JsonlTrajectoryRecorder
from ldm_tts.optimization.acquisition import SINGLE_OBJECTIVE_ACQUISITIONS, make_acquisition
from ldm_tts.data import DataCollectionSink, make_complete_design_ir
from ldm_tts.contracts import (
    AcquisitionSpec,
    CandidateDomainSpec,
    LDMTaskSpec,
    ObjectiveSpec,
    ReservoirExpansionSpec,
    ReservoirSpec,
    ResponseSpaceSpec,
    ProposalSearchSpec,
    SurrogateSpaceSpec,
)
from ldm_tts.transport.parsing import load_json_object
from tasks.antibody.core.ldm_light.methods import (
    METHOD_CHOICES,
    METHOD_SPECS,
    normalize_method,
)

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA)}
IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}
HYDROPHOBIC = set("AILMFWVY")
AROMATIC = set("FWY")
POSITIVE = set("RKH")
NEGATIVE = set("DE")
N_GLYCO = re.compile("N[^P][ST]")
ACQ_NAME = "ei"
ACQ_CHOICES = tuple(sorted(SINGLE_OBJECTIVE_ACQUISITIONS))
GP_TRAIN_STEPS = 300
WARMUP_POOL_SIZE = 1000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LLM warmup followed by LDM parallel acquisition selection.")
    p.add_argument("--config", default="resources/default_config.yaml")
    p.add_argument("--antigens_file", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_trials", type=int, default=1)
    p.add_argument("--n_evals", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--out_root", default="runs/llm_acq_baseline")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--timeout_s", type=int, default=120)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--history_top_k", type=int, default=10)
    p.add_argument("--parallel_budget", type=int, default=600,
                   help="Number of LDM-generated candidates scored by the GP acquisition after warmup.")
    p.add_argument("--n_init", type=int, default=20,
                   help="Number of initial oracle observations before fitting a GP acquisition.")
    p.add_argument("--method", type=normalize_method, choices=METHOD_CHOICES, default="policy_max")
    p.add_argument("--gen_m", type=int, default=5)
    p.add_argument("--n_strategies", type=int, default=5)
    p.add_argument("--planner_mode", choices=("choices", "independent"), default="choices")
    p.add_argument("--softmax_eta", type=float, default=1.0)
    p.add_argument("--per_strategy_budget", type=int, default=0)
    p.add_argument("--pool_score", choices=("acq", "combined"), default="acq")
    p.add_argument("--selection_score", choices=("acq", "combined"), default="acq")
    p.add_argument("--bias_weight", type=float, default=0.05)
    p.add_argument("--sample_timeout_s", type=float, default=5.0)
    p.add_argument("--device", choices=("cpu", "cuda"), default="")
    p.add_argument(
        "--acq",
        "--acquisition",
        dest="acq",
        choices=ACQ_CHOICES,
        default=ACQ_NAME,
        help=(
            "Shared GP posterior acquisition used after warmup. All modes are "
            "internally scored so larger is better."
        ),
    )
    p.add_argument(
        "--acq_beta",
        "--acq-beta",
        dest="acq_beta",
        type=float,
        default=1.0,
        help="Exploration coefficient for LCB/UCB acquisitions.",
    )
    p.add_argument(
        "--acq_xi",
        "--acq-xi",
        dest="acq_xi",
        type=float,
        default=0.001,
        help="Expected-improvement margin used by EI.",
    )
    p.add_argument("--include_antigen_context", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fallback_random", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def read_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_antigens(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_candidate_library(path: str | None, seq_len: int) -> list[str]:
    if not path:
        return []
    data = pd.read_csv(path, index_col=None)
    seqs = data.iloc[:, 0].astype(str).str.strip().str.upper().tolist()
    out: list[str] = []
    seen: set[str] = set()
    for seq in seqs:
        if valid_seq(seq, seq_len) and passes_developability(seq) and seq not in seen:
            out.append(seq)
            seen.add(seq)
    return out


def make_llm_client():
    try:
        from tasks.antibody.core.ldm import OpenAIClient
        return OpenAIClient()
    except Exception as exc:
        raise RuntimeError(
            "Could not create tasks.antibody.core.ldm.OpenAIClient. Check .env, openai package, "
            f"and project dependencies. Original error: {exc}"
        ) from exc


def seqs_to_indices(seqs: list[str]) -> np.ndarray:
    return np.array([[AA_TO_IDX[aa] for aa in seq] for seq in seqs], dtype=np.int32)


def indices_to_seqs(x: np.ndarray) -> list[str]:
    return ["".join(IDX_TO_AA[int(i)] for i in row) for row in np.asarray(x)]


def valid_seq(seq: str, seq_len: int) -> bool:
    return len(seq) == seq_len and all(aa in AA for aa in seq)


def longest_hydrophobic_run(seq: str) -> int:
    best = cur = 0
    for aa in seq:
        cur = cur + 1 if aa in HYDROPHOBIC else 0
        best = max(best, cur)
    return best


def net_charge(seq: str) -> float:
    total = 0.0
    for aa in seq:
        if aa == "H":
            total += 0.1
        elif aa in POSITIVE:
            total += 1.0
        elif aa in NEGATIVE:
            total -= 1.0
    return total


def passes_developability(seq: str) -> bool:
    return (
        seq.count("C") <= 1
        and longest_hydrophobic_run(seq) <= 4
        and sum(1 for aa in seq if aa in AROMATIC) <= 2
        and -1.0 <= net_charge(seq) <= 2.0
        and N_GLYCO.search(seq) is None
    )


def extract_json(raw: str) -> dict[str, Any]:
    try:
        return load_json_object(raw)
    except ValueError as exc:
        if str(exc) == "no JSON object found":
            raise ValueError("No JSON object found in LLM response") from exc
        raise


def best_history(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    return [
        {"sequence": row["LastProtein"], "score": row["LastValue"]}
        for row in sorted(rows, key=lambda r: r["LastValue"])[:top_k]
    ]


def build_prompt(
    antigen: str,
    seq_len: int,
    batch_size: int,
    observed: set[str],
    rows: list[dict[str, Any]],
    candidate_pool: list[dict[str, Any]],
    antigen_context: dict[str, Any] | None,
    top_k: int,
) -> str:
    payload = {
        "task": "Pure LLM baseline for CDRH3 sequence proposal.",
        "objective": "Minimize Absolut energy. Lower true score is better.",
        "mode": (
            "LLM proposal stage. Return diverse candidates; a Bayesian "
            "acquisition function may select which candidate(s) are evaluated."
        ),
        "reasoning": "Reason internally from history and antigen context, but do not output reasoning.",
        "antigen": antigen,
        "constraints": {
            "length": seq_len,
            "alphabet": AA,
            "num_sequences": batch_size,
            "choose_only_from_candidate_pool": True,
            "do_not_repeat": sorted(observed)[-200:],
            "candidate_pool_developability_filter": {
                "max_cysteine": 1,
                "max_hydrophobic_run": 4,
                "max_aromatic_FWY": 2,
                "net_charge_range": [-1.0, 2.0],
                "forbid_n_glycosylation_NXS_or_NXT": True,
            },
        },
        "history": {
            "num_observed": len(rows),
            "best": best_history(rows, top_k),
            "recent": rows[-top_k:],
        },
        "candidate_pool": candidate_pool,
        "antigen_context": antigen_context or {},
        "required_output": {
            "selected": [
                {
                    "id": 0,
                    "sequence": "A" * seq_len,
                    "score": "numeric LLM priority score; higher is better",
                }
            ]
        },
        "output_rules": [
            "Return JSON only.",
            "The only top-level key must be selected.",
            "Each selected item must contain only id, sequence, and score.",
            "Select only sequences present in candidate_pool.",
            "Do not include rationale or explanation.",
        ],
    }
    return json.dumps(payload, indent=2)


def parse_selected(
    obj: dict[str, Any],
    seq_len: int,
    observed: set[str],
    candidate_pool: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw = obj.get("selected", obj.get("candidates", obj.get("sequences", [])))
    if not isinstance(raw, list):
        raise ValueError("LLM JSON must contain a selected list")

    pool_by_id = {int(item["id"]): item["sequence"] for item in candidate_pool}
    pool_seqs = {item["sequence"] for item in candidate_pool}
    candidates: list[dict[str, Any]] = []
    used: set[str] = set()
    for item in raw:
        if isinstance(item, dict):
            raw_id = item.get("id")
            seq = str(item.get("sequence", "")).strip().upper()
            score = item.get("score")
            if raw_id is not None:
                try:
                    seq = pool_by_id[int(raw_id)]
                except (KeyError, TypeError, ValueError):
                    continue
        else:
            seq, score = str(item).strip().upper(), None

        if (
            seq not in pool_seqs
            or not valid_seq(seq, seq_len)
            or not passes_developability(seq)
            or seq in observed
            or seq in used
        ):
            continue
        try:
            score = None if score is None else float(score)
        except (TypeError, ValueError):
            score = None
        candidates.append({"sequence": seq, "score": score})
        used.add(seq)
    return candidates


def random_candidates(rng: random.Random, n: int, seq_len: int, observed: set[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    used: set[str] = set()
    attempts = 0
    max_attempts = max(10000, n * 200)
    while len(candidates) < n and attempts < max_attempts:
        attempts += 1
        seq = "".join(rng.choice(AA) for _ in range(seq_len))
        if seq not in observed and seq not in used and passes_developability(seq):
            candidates.append({"sequence": seq, "score": None})
            used.add(seq)
    if len(candidates) < n:
        raise RuntimeError(
            f"Could only generate {len(candidates)} developability-filtered random candidates "
            f"after {attempts} attempts; requested {n}."
        )
    return candidates


def make_candidate_pool(
    rng: random.Random,
    library: list[str],
    observed: set[str],
    seq_len: int,
    pool_size: int,
) -> list[dict[str, Any]]:
    if library:
        available = [seq for seq in library if seq not in observed]
        rng.shuffle(available)
        seqs = available[:pool_size]
    else:
        seqs = [item["sequence"] for item in random_candidates(rng, pool_size, seq_len, observed)]
    return [{"id": i, "sequence": seq} for i, seq in enumerate(seqs)]


def propose(
    llm: Any,
    rng: random.Random,
    antigen: str,
    seq_len: int,
    batch_size: int,
    observed: set[str],
    rows: list[dict[str, Any]],
    candidate_library: list[str],
    antigen_context: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    errors = []
    for attempt in range(1, args.max_retries + 1):
        candidate_pool = make_candidate_pool(
            rng=rng,
            library=candidate_library,
            observed=observed,
            seq_len=seq_len,
            pool_size=max(batch_size, WARMUP_POOL_SIZE),
        )
        if len(candidate_pool) < batch_size:
            raise RuntimeError(f"Candidate pool exhausted: {len(candidate_pool)} available, need {batch_size}")
        prompt = build_prompt(
            antigen=antigen,
            seq_len=seq_len,
            batch_size=batch_size,
            observed=observed,
            rows=rows,
            candidate_pool=candidate_pool,
            antigen_context=antigen_context,
            top_k=args.history_top_k,
        )
        raw = llm.call(prompt, temperature=args.temperature, timeout_s=args.timeout_s)
        try:
            parsed = extract_json(raw)
            candidates = parse_selected(parsed, seq_len, observed, candidate_pool)
            if len(candidates) >= batch_size:
                return candidates[:batch_size], {
                    "source": "llm",
                    "attempt": attempt,
                    "prompt": prompt,
                    "candidate_pool": candidate_pool,
                    "raw_response": raw,
                    "parsed": parsed,
                }
            raise ValueError(f"Only {len(candidates)} valid novel candidates returned")
        except Exception as exc:
            errors.append({"attempt": attempt, "error": str(exc), "raw_response": raw})

    if args.fallback_random:
        candidate_pool = make_candidate_pool(
            rng=rng,
            library=candidate_library,
            observed=observed,
            seq_len=seq_len,
            pool_size=max(batch_size, WARMUP_POOL_SIZE),
        )
        return [{"sequence": item["sequence"], "score": None} for item in candidate_pool[:batch_size]], {
            "source": "fallback_random",
            "errors": errors,
        }
    raise RuntimeError(json.dumps(errors, indent=2))


def build_status_from_rows(
    rows: list[dict[str, Any]],
    antigen: str,
    seed: int,
    iteration: int,
    antigen_context: dict[str, Any] | None,
    orchestrator,
):
    from tasks.antibody.core.ldm import OrchestratorStatus

    full_history = [
        (row["LastProtein"], float(row["LastValue"]), int(row["Index"]))
        for row in rows
    ]
    best_row = min(rows, key=lambda row: float(row["BestValue"]))
    best_seq = best_row["BestProtein"]
    best_idx = min(range(len(rows)), key=lambda i: float(rows[i]["LastValue"]))
    n_iters_without_improvement = len(rows) - 1 - best_idx
    return OrchestratorStatus(
        iteration=iteration,
        antigen_id=antigen,
        antigen_seed=seed,
        iter_seed=iteration,
        current_search_dsl=orchestrator.current_search_dsl,
        current_bias_dsl=orchestrator.current_bias_dsl,
        full_history=full_history,
        best_value=float(best_row["BestValue"]),
        best_sequence=seqs_to_indices([best_seq])[0].tolist(),
        n_evals=len(rows),
        n_iters_without_improvement=max(0, n_iters_without_improvement),
        antigen_context=antigen_context or {},
    )


def fit_gp_and_make_acquisition(
    rows: list[dict[str, Any]],
    *,
    acq_name: str = ACQ_NAME,
    beta: float = 1.0,
    xi: float = 0.001,
    gp_train_steps: int = GP_TRAIN_STEPS,
    device: Any = "cpu",
):
    """Fit a GP and return a larger-is-better minimization acquisition."""
    import torch
    from tasks.antibody.core.gp import train_gp

    acq_name = str(acq_name).lower()
    if acq_name not in ACQ_CHOICES:
        raise ValueError(f"Unsupported acquisition {acq_name!r}; choose one of {ACQ_CHOICES}.")
    beta = float(beta)
    acquisition = make_acquisition(
        acq_name,
        minimize=(True,),
        beta=beta,
        xi=float(xi),
    )
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA acquisition device, but torch.cuda.is_available() is False.")

    train_seqs = [row["LastProtein"] for row in rows]
    train_x = torch.tensor(seqs_to_indices(train_seqs), dtype=torch.float32, device=device)
    y_raw = np.array([float(row["LastValue"]) for row in rows], dtype=np.float32)
    y_mean = float(y_raw.mean())
    y_std = float(y_raw.std() + 1e-8)
    train_y = torch.tensor((y_raw - y_mean) / y_std, dtype=torch.float32, device=device).view(-1)

    gp = train_gp(
        train_x=train_x,
        train_y=train_y,
        use_ard=True,
        num_steps=max(30, int(gp_train_steps)),
        kern="transformed_overlap",
        noise_variance=1e-6,
        alphabet_size=len(AA),
        search_strategy="local",
    )
    gp.eval()
    best = train_y.min()

    def f_acq(x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        else:
            x = x.to(device=device, dtype=torch.float32)
        if x.dim() == 1:
            x = x.reshape(1, -1)
        posterior = gp(x)
        mu = posterior.mean.view(-1)
        sigma = posterior.stddev.view(-1).clamp_min(1e-9)
        return acquisition.score(mu, sigma, best=best.to(mu))

    return gp, f_acq


def fallback_search_dsl(rows: list[dict[str, Any]], budget: int):
    from tasks.antibody.core.ldm.dsl.search_space import LatinHyperCubeSampling, LocalSearch

    if not rows:
        return LatinHyperCubeSampling(num=max(1, budget))
    best_row = min(rows, key=lambda row: float(row["BestValue"]))
    restart = 3
    steps = max(1, budget // restart - 1)
    return LocalSearch(best_row["BestProtein"], radius=3, restart=restart, steps=steps)


def select_with_parallel_ldm(
    *,
    orchestrator,
    rows: list[dict[str, Any]],
    antigen: str,
    seed: int,
    iteration: int,
    antigen_context: dict[str, Any] | None,
    batch_size: int,
    args: argparse.Namespace,
    select_candidates: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use LDM's parallel executor: LLM DSL -> execute_atoms -> acquisition argmax."""
    import torch
    from tasks.antibody.core.ldm.acquisition.parallel_search import execute_atoms

    acq_name = str(getattr(args, "acq", ACQ_NAME)).lower()
    acq_beta = float(getattr(args, "acq_beta", 1.0))
    acq_xi = float(getattr(args, "acq_xi", 0.001))
    gp, f_acq = fit_gp_and_make_acquisition(
        rows,
        acq_name=acq_name,
        beta=acq_beta,
        xi=acq_xi,
    )
    status = build_status_from_rows(
        rows=rows,
        antigen=antigen,
        seed=seed,
        iteration=iteration,
        antigen_context=antigen_context,
        orchestrator=orchestrator,
    )
    decision = orchestrator.step(status)
    search_dsl = decision.search_dsl
    if search_dsl is None:
        search_dsl = fallback_search_dsl(rows, int(args.parallel_budget))
    bias_dsl = decision.bias_dsl

    results = execute_atoms(
        search_dsl=search_dsl,
        gp=gp,
        f_acq=f_acq,
        bias_dsl=bias_dsl,
        bias_weight=0.0,
        config=np.array([len(AA)] * len(rows[0]["LastProtein"]), dtype=int),
        cdr_constraints=True,
        rng=np.random.default_rng(seed + iteration),
        timeout_s=5.0,
        device=torch.device("cpu"),
        acq_name=acq_name,
    )
    if not results:
        raise RuntimeError(f"Parallel LDM search produced no candidates from {search_dsl!r}")

    score_key = f"bias+{acq_name}"
    ranked = sorted(range(len(results)), key=lambda i: results[i][score_key], reverse=True)
    selected_indices = ranked[:batch_size] if select_candidates else []
    selected_candidates: list[dict[str, Any]] = []
    for idx in selected_indices:
        item = results[idx]
        seq = "".join(IDX_TO_AA[int(v)] for v in item["seq"])
        selected_candidates.append({
            "sequence": seq,
            "score": None,
            "acquisition_score": float(item[score_key]),
            "acquisition_raw": float(item[acq_name]),
            "mu": float(item["mu"]),
            "sigma": float(item["sigma"]),
            "source": item.get("source", repr(search_dsl)),
        })

    parallel_results_json: list[dict[str, Any]] = []
    for item in results:
        parallel_results_json.append({
            "sequence": "".join(IDX_TO_AA[int(v)] for v in item["seq"]),
            acq_name: float(item[acq_name]),
            "mu": float(item["mu"]),
            "sigma": float(item["sigma"]),
            "bias": float(item.get("bias", 0.0)),
            score_key: float(item[score_key]),
            "source": item.get("source", repr(search_dsl)),
        })

    return selected_candidates, {
        "source": f"ldm_parallel_{acq_name}_argmax",
        "acq_name": acq_name,
        "acq_beta": acq_beta,
        "acq_xi": acq_xi,
        "decision": {
            "search_dsl": repr(search_dsl),
            "bias_dsl": repr(bias_dsl) if bias_dsl is not None else None,
            "fallback_used": bool(decision.fallback_used or decision.search_dsl is None),
            "rationale": decision.rationale,
        },
        "parallel_results": parallel_results_json,
        "selected_indices": selected_indices,
        "selected_candidates": selected_candidates,
    }


def run_absolut_info(absolut_path: str, command: str, antigen: str, timeout_s: int) -> str:
    exe = Path(absolut_path).resolve() / "src/bin/Absolut"
    if not exe.exists():
        return ""
    proc = subprocess.run(
        [str(exe), command, antigen],
        cwd=str(Path(absolut_path).resolve()),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return (proc.stdout or "")[:2000]


def collect_antigen_context(config: dict[str, Any], antigen: str) -> dict[str, Any]:
    bbox = dict(config["bbox"])
    bbox["antigen"] = antigen
    timeout_s = int(config.get("llm_antigen_context_timeout_s", 30))
    return {
        "antigen": antigen,
        "source": "Absolut",
        "info_antigen": run_absolut_info(bbox["path"], "info_antigen", antigen, timeout_s),
        "info_filenames": run_absolut_info(bbox["path"], "info_filenames", antigen, timeout_s),
    }


class RandomEvaluator:
    def energy(self, x: np.ndarray) -> tuple[np.ndarray, list[str]]:
        seqs = indices_to_seqs(x)
        return np.random.random(len(seqs)), seqs


class AbsolutEvaluator:
    def __init__(self, bbox: dict[str, Any], run_id: str) -> None:
        self.bbox = bbox
        self.run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)

    def energy(self, x: np.ndarray) -> tuple[np.ndarray, list[str]]:
        x = np.asarray(x, dtype=np.int32)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        antigen = self.bbox["antigen"]
        absolut_path = self.bbox["path"]
        n_proc = int(self.bbox["process"])
        start_task = int(self.bbox["startTask"])
        tmp = f"TempCDR3_{antigen}_{self.run_id}.txt"
        out = f"{antigen}FinalBindings_Process_1_Of_1.txt"
        cwd = os.getcwd()

        os.chdir(absolut_path)
        lock_dir = f".antbo_llm_acq_{antigen}.lock"
        self._acquire_lock(lock_dir)
        try:
            self._remove_files(antigen, n_proc, tmp, out)
            seqs = indices_to_seqs(x)
            with open(tmp, "w", encoding="utf-8") as handle:
                for i, seq in enumerate(seqs, start=1):
                    handle.write(f"{i}\t{seq}\n")

            proc = subprocess.run(
                [
                    "taskset",
                    "-c",
                    f"{start_task}-{start_task + n_proc}",
                    "./src/bin/Absolut",
                    "repertoire",
                    antigen,
                    tmp,
                    str(n_proc),
                ],
                capture_output=True,
                text=False,
            )
            if proc.returncode != 0:
                stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else "(empty)"
                raise RuntimeError(f"Absolut failed with returncode={proc.returncode}:\n{stderr}")
            if not os.path.exists(out):
                raise FileNotFoundError(f"Absolut output file not found: {out}")

            data = pd.read_csv(out, sep="\t", skiprows=1)
            data["sequence_idx"] = data.ID_slide_Variant.map(lambda value: int(str(value).split("_")[0]))
            values = data.groupby("sequence_idx")[["Energy"]].min()["Energy"].values
            return values, seqs
        finally:
            self._remove_files(antigen, n_proc, tmp, out)
            self._release_lock(lock_dir)
            os.chdir(cwd)

    @staticmethod
    def _remove_files(antigen: str, n_proc: int, tmp: str, out: str) -> None:
        for path in [tmp, out]:
            if os.path.exists(path):
                os.remove(path)
        for i in range(n_proc):
            part = f"TempBindingsFor{antigen}_t{i}_Part1_of_1.txt"
            if os.path.exists(part):
                os.remove(part)

    @staticmethod
    def _acquire_lock(lock_dir: str, timeout_s: int = 600) -> None:
        start = time.time()
        while True:
            try:
                os.mkdir(lock_dir)
                return
            except FileExistsError:
                if time.time() - start > timeout_s:
                    raise TimeoutError(f"Timed out waiting for Absolut lock: {lock_dir}")
                time.sleep(1.0)

    @staticmethod
    def _release_lock(lock_dir: str) -> None:
        try:
            os.rmdir(lock_dir)
        except FileNotFoundError:
            pass


def make_evaluator(config: dict[str, Any], antigen: str, run_id: str):
    bbox = dict(config["bbox"])
    bbox["antigen"] = antigen
    if bbox.get("tool", "Absolut") == "random":
        return RandomEvaluator(), bbox
    return AbsolutEvaluator(bbox, run_id), bbox


def describe_ldm_task(
    args: argparse.Namespace,
    config: dict[str, Any],
    antigen: str,
) -> LDMTaskSpec:
    seq_len = int(config.get("seq_len", 11))
    acq_name = str(getattr(args, "acq", ACQ_NAME)).lower()
    method = normalize_method(getattr(args, "method", "policy_max"))
    method_spec = METHOD_SPECS[method]
    return LDMTaskSpec(
        task="antibody",
        candidate_domain=CandidateDomainSpec(
            name="cdrh3_sequence",
            kind="categorical_sequence",
            dimension=seq_len,
            representation="fixed-length amino-acid sequence encoded as categorical indices",
            constraints={
                "alphabet": AA,
                "alphabet_size": len(AA),
                "max_cysteine": 1,
                "max_hydrophobic_run": 4,
                "max_aromatic_FWY": 2,
                "net_charge_range": [-1.0, 2.0],
                "forbid_n_glycosylation_NXS_or_NXT": True,
            },
            metadata={
                "antigen": antigen,
                "n_init": int(args.n_init),
                "parallel_budget": int(args.parallel_budget),
                "method": method,
                "base_measure": method_spec["base_measure"],
            },
        ),
        objectives=(
            ObjectiveSpec(
                name="absolut_energy",
                direction="minimize",
                description="Absolut binding energy; lower is better.",
            ),
        ),
        response_spaces=(
            ResponseSpaceSpec(
                name="candidate_pool_selection",
                output_kind="json",
                parser="tasks.antibody.core.ldm_light.ldm_acq.parse_selected",
                description="Warmup LLM selects sequence ids from a supplied candidate reservoir.",
            ),
            ResponseSpaceSpec(
                name="direct_sequence_generation",
                output_kind="json",
                parser="tasks.antibody.core.ldm_light.direct.parse_direct_sequences",
                description="Direct variants emit a JSON list of CDRH3 sequences.",
            ),
            ResponseSpaceSpec(
                name="dsl_update",
                output_kind="json",
                parser="tasks.antibody.core.ldm.llm.response_parser.parse_response",
                description="Post-warmup LLM updates search-space and optional bias DSL atoms.",
            ),
        ),
        acquisition=AcquisitionSpec(
            name=acq_name,
            objective_names=("absolut_energy",),
            score_direction="maximize",
            selection_rule=(
                "no acquisition; evaluate direct LLM generation"
                if not method_spec["uses_acquisition"]
                else f"{method_spec['reduction']} over {method_spec['base_measure']} reservoir acquisition scores"
            ),
            parameters={
                "beta": float(getattr(args, "acq_beta", 1.0)),
                "xi": float(getattr(args, "acq_xi", 0.001)),
                "n_init": int(args.n_init),
                "parallel_budget": int(args.parallel_budget),
                "batch_size": int(args.batch_size),
                "softmax_eta": float(getattr(args, "softmax_eta", 1.0)),
                "gen_m": int(getattr(args, "gen_m", 5)),
                "n_strategies": int(getattr(args, "n_strategies", 5)),
            },
        ),
        reservoir=ReservoirSpec(
            name="cdrh3_candidate_reservoir",
            expansions=(
                ReservoirExpansionSpec(
                    name="direct_sequence_generation",
                    action_kind="emit_candidate",
                    response_space="direct_sequence_generation",
                    produces_candidates=True,
                    description="Emit valid CDRH3 candidates directly.",
                ),
                ReservoirExpansionSpec(
                    name="policy_guided_generation",
                    action_kind="configure_generator",
                    response_space="dsl_update",
                    produces_candidates=True,
                    description="Update the DSL policy used to generate a candidate reservoir.",
                ),
            ),
            candidate_validator="CDRH3 length, alphabet, and biochemical constraint checks",
            deduplication_key="amino-acid sequence",
            max_size=int(args.parallel_budget),
            metadata={"base_measure": method_spec["base_measure"]},
        ),
        surrogate=SurrogateSpaceSpec(
            kind="vector" if method_spec["uses_acquisition"] else "none",
            representation=(
                "fixed-length categorical CDRH3 indices"
                if method_spec["uses_acquisition"]
                else "not used by direct LLM selection"
            ),
            dimension_policy="fixed" if method_spec["uses_acquisition"] else "none",
            dimension=seq_len if method_spec["uses_acquisition"] else None,
            encoder=(
                "tasks.antibody.core.antbo.bo.custom_init.StandardTransform"
                if method_spec["uses_acquisition"]
                else ""
            ),
            version="antbo_categorical_sequence_v1" if method_spec["uses_acquisition"] else "",
        ),
        proposal_search=ProposalSearchSpec(
            name="single_turn",
            evaluation_policy="outer_loop_acquisition_selection",
            parameters={
                "method": method,
                "proposal_mode": method_spec["base_measure"],
                "reduction": method_spec["reduction"],
            },
        ),
        metadata={"seed": int(args.seed), "n_evals": int(args.n_evals)},
    )


def append_results(
    rows: list[dict[str, Any]],
    values: np.ndarray,
    seqs: list[str],
    llm_scores: list[float | None],
    acquisition_scores: list[float | None],
    elapsed_s: float,
    source: str,
    acquisition_used: bool,
    start_idx: int,
) -> tuple[int, float, str]:
    best_value = min((row["BestValue"] for row in rows), default=float("inf"))
    best_seq = rows[-1]["BestProtein"] if rows else ""
    idx = start_idx
    for seq, llm_score, acq_score, value in zip(seqs, llm_scores, acquisition_scores, values):
        value = float(value)
        if value < best_value:
            best_value = value
            best_seq = seq
        rows.append({
            "Index": idx,
            "LastValue": value,
            "BestValue": best_value,
            "LLMScore": llm_score,
            "AcquisitionScore": acq_score,
            "Time": elapsed_s,
            "LastProtein": seq,
            "BestProtein": best_seq,
            "Source": source,
            "AcquisitionUsed": bool(acquisition_used),
        })
        idx += 1
    return idx, best_value, best_seq


def collect_direct_sequence_action(
    sink: DataCollectionSink,
    *,
    decision: dict[str, Any],
    selected_candidates: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    observed: set[str],
    antigen: str,
    antigen_context: dict[str, Any] | None,
    seq_len: int,
    history_top_k: int,
    seed: int,
    eval_start: int,
    method: str,
    run_dir: Path,
) -> bool:
    """Collect a validated direct LLM action and reject fallback/DSL decisions."""

    if not sink.enabled:
        return False
    source = str(decision.get("source") or "")
    candidate_pool = None
    if source == "llm":
        candidate_pool = decision.get("candidate_pool")
        teacher_candidates = selected_candidates
    elif source == "llm_direct":
        teacher_candidates = selected_candidates
    elif source.startswith("direct_"):
        generation = decision.get("generation")
        if not isinstance(generation, dict):
            return False
        generation_source = str(generation.get("source") or "")
        if generation_source != "llm_direct":
            return False
        raw_candidates = decision.get("candidates")
        if not isinstance(raw_candidates, list):
            return False
        teacher_candidates = raw_candidates
    else:
        return False

    candidates = [
        {"design": candidate["sequence"], "rationale": None}
        for candidate in teacher_candidates
        if isinstance(candidate, dict) and candidate.get("sequence")
    ]
    if not candidates:
        return False

    best_row = min(rows, key=lambda row: float(row["LastValue"])) if rows else None
    observations = []
    for row in rows[-max(1, int(history_top_k)) :]:
        roles = ["recent"]
        if best_row is row:
            roles.append("best")
        observations.append(
            {
                "design": row["LastProtein"],
                "results": {"absolut_energy": float(row["LastValue"])},
                "roles": roles,
                "round": int(row["Index"]),
            }
        )
    best_so_far = (
        None
        if best_row is None
        else {
            "design": best_row["LastProtein"],
            "results": {"absolut_energy": float(best_row["LastValue"])},
            "round": int(best_row["Index"]),
        }
    )
    raw_context: dict[str, Any] = {
        "target_id": antigen,
        "target_context": antigen_context or {},
    }
    if isinstance(candidate_pool, list) and candidate_pool:
        raw_context["candidate_pool"] = candidate_pool

    ir = make_complete_design_ir(
        task_id="protein",
        domain="antibody_sequence",
        task_description=(
            f"Direct CDRH3 antibody sequence generation for antigen {antigen}. "
            "Generate developable antibody strings directly."
        ),
        objectives=[
            {
                "name": "absolut_energy",
                "direction": "minimize",
                "description": "Absolut binding energy; lower is better.",
            }
        ],
        design_space_description=(
            "Fixed-length CDRH3 sequence over the standard amino-acid alphabet "
            "with developability constraints."
        ),
        active_parameters=[
            {
                "name": "sequence",
                "type": "string",
                "domain": {"length": int(seq_len), "alphabet": list(AA)},
                "edit_op": None,
            }
        ],
        observations=observations,
        best_so_far=best_so_far,
        candidates=candidates,
        request_description=(
            f"Propose {len(candidates)} CDRH3 sequence(s) of length {seq_len} "
            f"over alphabet {AA}; do not repeat observed sequences."
        ),
        num_candidates=len(candidates),
        round_idx=eval_start,
        num_evaluated=len(rows),
        do_not_repeat=sorted(observed)[-200:],
        allows_new_parameters=False,
        reasoning_available=False,
        raw_context=raw_context,
    )
    sink.append(
        ir,
        provenance={
            "task": "protein",
            "run_dir": str(run_dir),
            "antigen": antigen,
            "seed": seed,
            "eval_start": eval_start,
            "method": method,
            "source": source,
        },
    )
    return True


def run_one(config: dict[str, Any], antigen: str, seed: int, args: argparse.Namespace) -> Path:
    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed)
    acquisition_rng = np.random.default_rng(seed)

    run_id = f"{antigen}_seed{seed}_pid{os.getpid()}"
    seq_len = int(config.get("seq_len", 11))
    method = normalize_method(getattr(args, "method", "policy_max"))
    method_spec = METHOD_SPECS[method]
    pool_csv = config.get("tabular_search_csv")
    candidate_library = read_candidate_library(pool_csv, seq_len)
    if method == "legacy_policy_max" and candidate_library:
        print(f"[llm-acq] Loaded candidate library: {len(candidate_library)} sequences from {pool_csv}")
    elif method == "legacy_policy_max":
        print("[llm-acq] No candidate library provided; using random temporary candidate reservoirs.")
    llm = make_llm_client()
    evaluator, bbox = make_evaluator(config, antigen, run_id)

    acq_name = str(getattr(args, "acq", ACQ_NAME)).lower()
    acq_beta = float(getattr(args, "acq_beta", 1.0))
    acq_xi = float(getattr(args, "acq_xi", 0.001))
    mode = f"{method}_{acq_name}_budget{args.parallel_budget}"
    run_dir = Path(args.out_root) / f"{mode}_antigen_{antigen}_seed_{seed}_n{args.n_evals}_batch{args.batch_size}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.csv"
    data_sink = DataCollectionSink.from_env(default_root=run_dir / "ldm_data")
    decision_recorder = JsonlTrajectoryRecorder(
        run_dir,
        config_snapshot={
            "algorithm": "LDM-TTS",
            "task": "antibody",
            "antigen": antigen,
            "seed": seed,
            "n_evals": int(args.n_evals),
            "batch_size": int(args.batch_size),
            "parallel_budget": int(args.parallel_budget),
            "method": method,
            "method_spec": method_spec,
            "gen_m": int(getattr(args, "gen_m", 5)),
            "n_strategies": int(getattr(args, "n_strategies", 5)),
            "planner_mode": str(getattr(args, "planner_mode", "choices")),
            "softmax_eta": float(getattr(args, "softmax_eta", 1.0)),
            "per_strategy_budget": int(getattr(args, "per_strategy_budget", 0)),
            "pool_score": str(getattr(args, "pool_score", "acq")),
            "selection_score": str(getattr(args, "selection_score", "acq")),
            "bias_weight": float(getattr(args, "bias_weight", 0.05)),
            "acquisition": acq_name,
            "acq_beta": acq_beta,
            "acq_xi": acq_xi,
            "ldm_task_spec": describe_ldm_task(args, config, antigen).to_dict(),
        },
        rounds_filename="llm_acq_decisions.jsonl",
        reset_rounds_file=True,
    )
    orchestrator = None
    if method == "legacy_policy_max":
        from tasks.antibody.core.ldm import DSLConfig, Orchestrator

        ldm_cfg = DSLConfig(
            llm_init_enabled=False,
            llm_loop_enabled=True,
            llm_temperature=float(args.temperature),
            max_retries=int(args.max_retries),
            llm_call_timeout_s=int(args.timeout_s),
            history_max_in_prompt=int(args.history_top_k),
            bias_weight=0.0,
            sample_timeout_s=float(getattr(args, "sample_timeout_s", 5.0)),
            batch_size=int(args.batch_size),
            acq_search_budget=int(args.parallel_budget),
            acq_max_rounds=1,
            num_llm_review=max(1, min(int(args.parallel_budget), 10)),
            strategy="ldm-default",
        )
        orchestrator = Orchestrator(
            config=ldm_cfg,
            llm_client=llm,
            decision_log_path=run_dir / "ldm_parallel_decisions.json",
        )

    antigen_context = None
    if args.include_antigen_context and bbox.get("tool", "Absolut") == "Absolut":
        antigen_context = collect_antigen_context(config, antigen)
        with open(run_dir / "llm_antigen_context.json", "w", encoding="utf-8") as f:
            json.dump(antigen_context, f, indent=2)

    observed: set[str] = set()
    rows: list[dict[str, Any]] = []
    eval_idx = 0
    del observed, rows, eval_idx  # history is rebuilt from engine observations

    from tasks.antibody.core import engine_adapters

    sink = DataCollectionSink.from_env(default_root=run_dir / "ldm_data")
    domain = engine_adapters.AntibodyCandidateDomain(seq_len)
    expander = engine_adapters.AntibodyReservoirExpander(
        method=method,
        method_spec=method_spec,
        antigen=antigen,
        seed=seed,
        seq_len=seq_len,
        args=args,
        llm=llm,
        rng=rng,
        acquisition_rng=acquisition_rng,
        antigen_context=antigen_context,
        orchestrator=orchestrator,
        candidate_library=candidate_library,
        sink=sink,
        run_dir=run_dir,
    )
    evaluator_adapter = engine_adapters.AntibodyEvaluator(evaluator)
    encoder = None
    selector = None
    if bool(method_spec["uses_acquisition"]):
        encoder = engine_adapters.AntibodySurrogateEncoder(seq_len)
        selector = engine_adapters.AntibodyGPSelector(
            args=args, method_spec=method_spec, rng=acquisition_rng
        )
    task_spec = describe_ldm_task(args, config, antigen)
    recipe = CampaignRecipe(
        task_spec=task_spec,
        expander=expander,
        candidate_domain=domain,
        evaluator=evaluator_adapter,
        surrogate_encoder=encoder,
        selector=selector,
    )
    iterations = -(-int(args.n_evals) // int(args.batch_size))
    campaign = run_campaign(
        CampaignRequest(
            run_dir=run_dir,
            config=_jsonable_args(args),
            budget=CampaignBudget(
                rounds=iterations,
                reservoir_size=max(1, int(args.parallel_budget)),
                batch_size=int(args.batch_size),
                target_observations=int(args.n_evals),
                max_evaluation_attempts=int(args.n_evals),
                max_empty_reservoir_rounds=max(iterations, 1),
            ),
            artifact_projector=lambda runtime, result: engine_adapters.materialize_legacy_run(
                runtime,
                result,
                run_dir,
                antigen=antigen,
                seed=seed,
                method=method,
                method_spec=method_spec,
                args=args,
                decision_recorder=decision_recorder,
            ),
        ),
        recipe,
    )
    result = campaign.engine

    if hasattr(llm, "close"):
        llm.close()
    return run_dir


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def main() -> None:
    args = parse_args()
    if args.parallel_budget <= 0:
        raise ValueError("--parallel_budget must be positive")
    if args.n_init < 0:
        raise ValueError("--n_init must be non-negative")
    args.acq = str(args.acq).lower()
    if args.acq not in ACQ_CHOICES:
        raise ValueError(f"--acq must be one of {ACQ_CHOICES}")
    if args.acq_beta < 0:
        raise ValueError("--acq_beta must be non-negative")
    if args.acq_xi < 0:
        raise ValueError("--acq_xi must be non-negative")
    if args.gen_m <= 0 or args.n_strategies <= 0:
        raise ValueError("--gen_m and --n_strategies must be positive")
    if args.softmax_eta < 0 or np.isnan(args.softmax_eta):
        raise ValueError("--softmax_eta must be non-negative or positive infinity")
    if args.per_strategy_budget < 0:
        raise ValueError("--per_strategy_budget must be non-negative")
    if args.sample_timeout_s <= 0:
        raise ValueError("--sample_timeout_s must be positive")
    if METHOD_SPECS[args.method]["uses_acquisition"] and args.n_init <= 0:
        raise ValueError("Acquisition-guided methods require --n_init to be positive")
    if (
        str(args.method).startswith("policy_")
        and args.per_strategy_budget == 0
        and args.parallel_budget < args.n_strategies
    ):
        raise ValueError(
            "Policy methods require --parallel_budget >= --n_strategies "
            "when --per_strategy_budget is zero"
        )
    config = read_yaml(os.path.abspath(args.config))
    args.device = args.device or str(config.get("device") or "cpu")
    antigens = read_antigens(args.antigens_file)
    print(f"LLM + parallel {args.acq.upper()} baseline antigens: {antigens}")

    for antigen in antigens:
        for seed in range(args.seed, args.seed + args.n_trials):
            run_dir = run_one(config, antigen, seed, args)
            print(f"Saved LLM + parallel {args.acq.upper()} baseline run to {run_dir}")


if __name__ == "__main__":
    main()
