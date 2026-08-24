#!/usr/bin/env python3
"""Small-molecule task workflow for the shared LDM-TTS config runner.

This module wires the small-molecule scientific components into the shared
``ldm_tts.engine.LDMEngine`` campaign runtime:

    LLM proposal reservoir -> shared acquisition-tilted selection -> environment scoring

The task-owned adapters (candidate domain, reservoir expander, evaluator,
acquisition selector, and surrogate encoder) live in
``tasks.small_molecule.core.engine_adapters``. The previous task-local
``ldm_tilted_case2.loop.run_tilted_case2_search`` loop is retired; its pure
helpers remain importable and its trajectory files are re-exported from the
engine events.

Example smoke run without external services:

    python -m tasks.small_molecule.ldm_task.procedure \
        --mock \
        --method m1_stratified_direct_llm_oversample_sir \
        --budget 8 \
        --m1-k-direct-llm 16 \
        --trajectory-dir runs/case2_mock

Example real run:

    python -m tasks.small_molecule.ldm_task.procedure \
        --method m1_stratified_direct_llm_oversample_sir \
        --init-strategy llm_cold_start \
        --budget 80 \
        --m1-k-direct-llm 512 \
        --max-candidates-per-round 256 \
        --kernel sk \
        --gp-device cpu \
        --llm-url http://127.0.0.1:52307/v1 \
        --llm-model-name Qwen3-Coder-30B-A3B-Instruct \
        --llm-max-retries 20 \
        --llm-retry-wait-seconds 10 \
        --vina-bin /path/to/vina \
        --trajectory-dir runs/case2_real

Resume an interrupted run:

    python -m tasks.small_molecule.ldm_task.procedure \
        --resume-from runs/case2_real \
        --budget 160
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ldm_tts.registration.registry import REPOSITORY_RELATIVE_PREFIXES
from ldm_tts.campaign import (
    CampaignBudget,
    CampaignRecipe,
    CampaignRequest,
    InitializationExpander,
    InitializationOrderSelector,
    run_campaign,
)
from ldm_tts.engine import LDMEngineState
from ldm_tts.engine.run_store import CampaignRuntime, unique_run_dir
from ldm_tts.data import DataCollectionSink
from ldm_tts.contracts import (
    AcquisitionSpec,
    CandidateDomainSpec,
    LDMTaskSpec,
    ObjectiveSpec,
    RawProposal,
    ReservoirExpansionSpec,
    ReservoirSpec,
    ResponseSpaceSpec,
    ProposalSearchSpec,
    SurrogateSpaceSpec,
)
from ldm_tts.registration.dependencies import format_checks, has_failures
from tasks.small_molecule.core.dependencies import check_small_molecule

DEFAULT_NN_MODEL_PATH = ""
QWEN35_DEFAULT_SAMPLING = {
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}
VALID_METHODS = {
    "m1_direct_llm_sir",
    "m1_stratified_direct_llm_sir",
    "m1_stratified_direct_llm_oversample_sir",
    "m1_stratified_direct_llm_only",
    "m1_llm_one_step",
    "m1_llm_seed_analog_oversample_sir",
}


class ExpandingMockCase2LLM:
    """Deterministic molecule-emitting mock for local loop smoke tests."""

    def __init__(self) -> None:
        self.model_name = "mock-case2-tts"
        self.call_log: list[dict[str, object]] = []
        self._counter = 0

    def chat(self, system: str, user: str, *, json_mode: bool = True) -> str:
        self._counter += 1
        base_len = 3 + (self._counter % 10)
        base = "C" * base_len
        if '"seeds"' in user:
            payload = {
                "seeds": [
                    {"smiles": base, "budget": 8, "intent": "mock local seed"},
                    {"smiles": base + "N", "budget": 8, "intent": "mock polar seed"},
                ]
            }
        else:
            payload = {
                "direct_smiles": [
                    {"smiles": base, "rationale": "alkyl"},
                    {"smiles": base + "N", "rationale": "amine"},
                    {"smiles": base + "O", "rationale": "alcohol"},
                    {"smiles": "N" + base + "N", "rationale": "diamine"},
                    {"smiles": "O" + base + "N", "rationale": "hetero"},
                ]
            }
        text = json.dumps(payload)
        self.call_log.append({
            "system": system,
            "user": user,
            "response": text,
            "idx": self._counter - 1,
        })
        return text


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args)
    output_dir = resolve_output_dir(args)
    resume_requested = bool(args.resume)
    if resume_requested and not output_dir.exists():
        raise SystemExit(f"--resume target does not exist: {output_dir}")
    run_dir = output_dir if resume_requested else unique_run_dir(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(json.dumps({
            "config": planned_config_json(args, output_dir),
            "ldm_task_spec": describe_ldm_task(args).to_dict(),
            "output_dir": str(output_dir),
            "mock": bool(args.mock),
        }, indent=2, sort_keys=True))
        return 0

    if not args.mock:
        preflight_real_dependencies(args)

    try:
        cfg = build_config(args, output_dir)
        from tasks.small_molecule.core import engine_adapters

        llm = build_llm(args)
        vina_fn, activity_fn = (
            build_mock_scorers() if args.mock else build_real_scorers(args, output_dir)
        )
        analog_fn = build_mock_analog_fn() if args.mock else build_real_analog_fn(args, output_dir)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Could not import the tilted case2 runtime. Install the project GP "
            "dependencies first, especially torch, gpytorch, and gauche. "
            f"Original import error: {exc}"
        ) from exc

    legacy_resume = resume_requested and not (output_dir / "campaign.json").exists()
    spec = describe_ldm_task(args)

    evaluator = engine_adapters.SmilesCandidateEvaluator(vina_fn, activity_fn)
    sink = DataCollectionSink.from_env(default_root=run_dir / "ldm_data")

    domain = engine_adapters.SmilesCandidateDomain(cfg)
    expander = engine_adapters.SmilesReservoirExpander(
        cfg,
        llm,
        analog_fn,
        rng=None,
        budget_hook=None,
    )
    initialization_target = 0
    if not resume_requested and cfg.init_strategy == "seed_smiles":
        initial_proposals = _initial_seed_proposals(args, cfg)
        initialization_target = len(initial_proposals)
        expander = InitializationExpander(
            initial_proposals,
            expander,
            successful_target=initialization_target,
            source="seed_smiles",
        )
    encoder = None
    selector = None
    if cfg.method not in engine_adapters.DIRECT_ONLY_METHODS:
        encoder = engine_adapters.SmilesSurrogateEncoder(cfg.gp_config)
        selector = engine_adapters.TiltedAcquisitionSelector(cfg)
        if initialization_target:
            selector = InitializationOrderSelector(
                selector,
                successful_target=initialization_target,
            )
    recipe = CampaignRecipe(
        task_spec=spec,
        expander=expander,
        candidate_domain=domain,
        evaluator=evaluator,
        surrogate_encoder=encoder,
        selector=selector,
    )
    minimum_rounds = -(-int(cfg.budget) // int(cfg.batch_size))
    max_rounds = minimum_rounds + max(1, int(cfg.max_empty_reservoir_rounds))
    max_attempts = max(int(cfg.budget), int(cfg.budget) * 8)
    campaign = run_campaign(
        CampaignRequest(
            run_dir=run_dir,
            config=_jsonable_args(args),
            resume=resume_requested and not legacy_resume,
            state_factory=lambda runtime: _resolve_engine_state(
                runtime,
                engine_adapters,
                resume=resume_requested,
                legacy_resume=legacy_resume,
            ),
            budget=CampaignBudget(
                rounds=max_rounds,
                reservoir_size=cfg.max_candidates_per_round,
                batch_size=cfg.batch_size,
                target_successful_evaluations=cfg.budget,
                max_evaluation_attempts=max_attempts,
                max_evaluation_attempts_per_round=min(
                    cfg.max_candidates_per_round, max(8, cfg.batch_size)
                ),
                replace_failed_evaluations=True,
                max_empty_reservoir_rounds=(
                    cfg.max_empty_reservoir_rounds
                    if cfg.allow_early_stop
                    else max_rounds
                ),
                extra_limits={
                    "llm_requests": max_rounds * max(1, cfg.llm_max_retries + 2),
                    # The stratified direct-LLM reservoir expander issues up to
                    # `max_candidates_per_round` proposals per round across a
                    # reservoir expansion plus refill loops, each chunk being a
                    # separate LLM call. `(llm_max_retries + 2)` only accounts
                    # for one expansion's worth of calls, so scale by the
                    # number of chunks (LLM_DIRECT_CHUNK_SIZE == 8) to avoid
                    # exhausting `proposal_attempts` before the target
                    # evaluations complete.
                    "proposal_attempts": max_rounds
                    * max(1, cfg.llm_max_retries + 2)
                    * max(1, int(cfg.max_candidates_per_round) // 8),
                },
            ),
            artifact_projector=lambda runtime, result: engine_adapters.materialize_legacy_trajectory(
                runtime, result, cfg, sink=sink
            ),
        ),
        recipe,
    )
    engine_result = campaign.engine

    history = [
        (observation.candidate.payload["smiles"], (
            observation.evaluation.metrics.get("vina"),
            observation.evaluation.metrics.get("activity"),
        ))
        for observation in engine_result.state.observations
    ]
    legacy_summary = campaign.projected

    result = {
        "output_dir": str(run_dir.resolve()),
        "history_size": len(history),
        "best": best_observed(history, cfg.minimize),
        "summary": legacy_summary,
        "history_path": str((run_dir / "history.json").resolve()),
        "summary_path": str((run_dir / "summary.json").resolve()),
        "rounds_path": str((run_dir / "rounds.jsonl").resolve()),
        "engine_summary": engine_result.summary,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "api_key"
    }


def _resolve_engine_state(
    runtime: CampaignRuntime,
    engine_adapters,
    *,
    resume: bool,
    legacy_resume: bool,
) -> LDMEngineState:
    if resume:
        if not legacy_resume:
            checkpoint = runtime.load_checkpoint()
            if checkpoint is not None:
                return LDMEngineState.from_checkpoint(checkpoint)
        history_rows = _load_legacy_history(Path(runtime.run_dir))
        return LDMEngineState(
            observations=engine_adapters.observations_from_history_rows(history_rows)
        )
    return LDMEngineState()


def _initial_seed_proposals(args, cfg) -> tuple[RawProposal, ...]:
    from tasks.small_molecule.core.ldm_tilted_case2.canonicalize import canonicalize_smiles

    canonical: list[str] = []
    seen: set[str] = set()
    for smiles in parse_seed_smiles(args.seed_smiles):
        canon = canonicalize_smiles(smiles)
        if canon and canon not in seen:
            canonical.append(canon)
            seen.add(canon)
        if len(canonical) >= cfg.init_size:
            break
    return tuple(
        RawProposal(
            {"smiles": smiles, "rationale": ""},
            "seed_smiles",
        )
        for smiles in canonical
    )


def _load_legacy_history(run_dir: Path) -> list[tuple[str, Sequence[object]]]:
    history_path = run_dir / "history.json"
    if history_path.exists():
        rows = json.loads(history_path.read_text(encoding="utf-8"))
        return [
            (str(row["smiles"]), tuple(row["scores"]))
            for row in rows
        ]
    from ldm_tts.engine.run_store import load_jsonl

    history: list[tuple[str, Sequence[object]]] = []
    for record in load_jsonl(run_dir / "rounds.jsonl"):
        selection = record.get("selection_results", {})
        for smiles, scores in zip(
            selection.get("selected_smiles", []),
            selection.get("selected_scores", []),
        ):
            history.append((str(smiles), tuple(scores)))
    return history


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tilted case2 molecule test-time search."
    )
    parser.add_argument("--method", choices=sorted(VALID_METHODS), default="m1_stratified_direct_llm_oversample_sir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-smiles", default="CCO,CCN,CCC,CCCN,CCCC")
    parser.add_argument("--init-strategy", choices=["seed_smiles", "llm_cold_start"], default="llm_cold_start")
    parser.add_argument("--init-size", type=int, default=5)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--smiles-max-len", type=int, default=80)
    parser.add_argument("--max-candidates-per-round", type=int, default=256)
    parser.add_argument("--max-empty-reservoir-rounds", type=int, default=10)
    parser.add_argument(
        "--allow-early-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow the search to stop before budget when repeated empty reservoirs hit "
            "--max-empty-reservoir-rounds. Use --no-allow-early-stop to keep retrying."
        ),
    )
    parser.add_argument("--kernel", choices=["fp", "sk"], default="sk")
    parser.add_argument("--gp-device", default="cpu")
    parser.add_argument("--gp-fit-itersteps", type=int, default=20)
    parser.add_argument("--gp-fp-n-bits", type=int, default=2048)
    parser.add_argument(
        "--acq",
        "--acquisition",
        dest="acq",
        choices=["ehvi", "mean"],
        default="ehvi",
        help="Shared two-objective posterior acquisition used to tilt candidate sampling.",
    )
    parser.add_argument(
        "--acq-weights",
        default="0.5,0.5",
        metavar="VINA,ACTIVITY",
        help="Comma-separated objective weights used by posterior-mean acquisition.",
    )
    parser.add_argument("--ehvi-n-samples", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight on log q0 base measure.")
    parser.add_argument("--eta", type=float, default=3.0, help="Weight on robust-z acquisition tilt.")
    parser.add_argument("--m1-k-direct-llm", type=int, default=128)
    parser.add_argument("--m1-q0-smoothing", type=float, default=None)
    parser.add_argument("--m1-analog-n-llm-seeds", type=int, default=8)
    parser.add_argument("--m1-analog-k-total", type=int, default=1024)
    parser.add_argument("--llm-url", default=os.environ.get("LLM_BASE_URL", ""))
    parser.add_argument(
        "--llm-model-name",
        default=(
            os.environ.get("LLM_MODEL_NAME")
            or os.environ.get("LLM_MODEL")
            or "DeepSeek-V4-Flash"
        ),
    )
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""))
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-max-tokens", type=int, default=4096)
    parser.add_argument("--llm-top-p", type=float, default=None)
    parser.add_argument("--llm-top-k", type=int, default=None)
    parser.add_argument("--llm-min-p", type=float, default=None)
    parser.add_argument("--llm-presence-penalty", type=float, default=None)
    parser.add_argument("--llm-repetition-penalty", type=float, default=None)
    parser.add_argument(
        "--qwen35-sampling-defaults",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For Qwen3.5 non-coder reasoning models, apply Qwen-style defaults "
            "for unset sampling passthroughs: top_p=0.95, top_k=20, min_p=0, "
            "presence_penalty=1.5, repetition_penalty=1.0."
        ),
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help=(
            "Pass chat_template_kwargs.enable_thinking=false through extra_body. "
            "Useful for Qwen3.5 reasoning models served by vLLM/SGLang."
        ),
    )
    parser.add_argument(
        "--llm-extra-body-json",
        default="",
        help=(
            "Raw JSON object merged into the OpenAI SDK extra_body request field "
            "for provider-specific parameters."
        ),
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=2,
        help="Retry attempts after the initial LLM JSON attempt fails.",
    )
    parser.add_argument(
        "--llm-retry-wait-seconds",
        type=float,
        default=0.0,
        help="Seconds to wait between failed LLM JSON attempts.",
    )
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock LLM/scorers/analog generator.")
    parser.add_argument("--trajectory-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from the selected trajectory directory. Existing history.json is used "
            "when present; otherwise rounds.jsonl is replayed."
        ),
    )
    parser.add_argument(
        "--resume-from",
        default="",
        help=(
            "Resume from an existing trajectory directory, or from a file inside it "
            "(summary.json, history.json, rounds.jsonl, or config.json). Implies --resume."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Console progress verbosity. Default hides per-chunk LLM details.",
    )
    parser.add_argument(
        "--debug-llm-chunks",
        action="store_true",
        help="Show individual LLM chunk start/success logs.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--vina-bin", default=os.environ.get("VINA_BIN", ""))
    parser.add_argument("--vina-cache-dir", default="")
    parser.add_argument("--vina-pdb-id", default="8UN5")
    parser.add_argument("--vina-chain-id", default="A")
    parser.add_argument("--vina-ligand-resname", default="")
    parser.add_argument("--vina-exhaustiveness", type=int, default=4)
    parser.add_argument("--vina-n-poses", type=int, default=3)
    parser.add_argument("--vina-seed", type=int, default=42)
    parser.add_argument("--vina-max-workers", type=int, default=1)
    parser.add_argument(
        "--vina-no-cache",
        action="store_true",
        help="Disable reuse of cached docking results inside the Vina cache directory.",
    )
    parser.add_argument(
        "--vina-allow-zero-charge-fallback",
        action="store_true",
        help="Allow debug receptor preparation when Meeko assigns zero receptor charges.",
    )
    parser.add_argument(
        "--vina-allow-debug-receptor",
        action="store_true",
        help="Allow docking against a receptor marked as debug/non-production.",
    )
    parser.add_argument(
        "--nn-model-path",
        default=os.environ.get("G12D", DEFAULT_NN_MODEL_PATH),
        help="Trusted joblib activity-model path. Required for real runs; may also be set with G12D.",
    )
    parser.add_argument("--reasyn-repo", default=os.environ.get("REASYN_HOME", os.environ.get("REASYN_REPO", "")))
    parser.add_argument("--reasyn-python", default=os.environ.get("REASYN_PYTHON", os.environ.get("REASYN_BIN", "")))
    parser.add_argument("--reasyn-model-path", default=os.environ.get("REASYN_MODEL_PATH", ""))
    parser.add_argument("--reasyn-devices", default="0")
    parser.add_argument("--reasyn-time-limit", type=int, default=1800)
    return parser.parse_args(argv)


def preflight_real_dependencies(args: argparse.Namespace) -> None:
    dep_args = {key.replace("_", "-"): value for key, value in vars(args).items()}
    checks = check_small_molecule(
        dep_args,
        os.environ.copy(),
        REPO_ROOT,
        mode="real",
        include_optional="analog" in str(args.method),
    )
    if has_failures(checks):
        raise SystemExit(
            "Small-molecule dependency preflight failed. Fix the failed items "
            "or run `python scripts/check_task_dependencies.py <config>` for "
            "the full config-aware report.\n" + format_checks(checks)
        )


def configure_logging(args: argparse.Namespace) -> None:
    root_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=root_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    logging.captureWarnings(True)
    direct_logger = logging.getLogger("tasks.small_molecule.core.ldm_tilted_case2.methods.direct_llm")
    if args.debug_llm_chunks:
        direct_logger.setLevel(logging.DEBUG)
    elif root_level <= logging.DEBUG:
        direct_logger.setLevel(logging.INFO)
    else:
        direct_logger.setLevel(logging.NOTSET)
    logging.getLogger("numexpr").setLevel(logging.WARNING)


def build_config(args: argparse.Namespace, output_dir: Path) -> TiltedLDMCase2Config:
    from tasks.small_molecule.core.gp import GPConfig
    from tasks.small_molecule.core.ldm_tilted_case2.config import TiltedLDMCase2Config

    return TiltedLDMCase2Config(
        method=args.method,
        init_size=args.init_size,
        init_strategy=args.init_strategy,
        budget=args.budget,
        batch_size=args.batch_size,
        smiles_max_len=args.smiles_max_len,
        max_candidates_per_round=args.max_candidates_per_round,
        max_empty_reservoir_rounds=args.max_empty_reservoir_rounds,
        allow_early_stop=bool(args.allow_early_stop),
        gp_config=GPConfig(
            impl="smiles-strkernel" if args.kernel == "sk" else "fingerprint+tanimoto",
            device=args.gp_device,
            fit_n_itersteps=args.gp_fit_itersteps,
            fp_n_bits=args.gp_fp_n_bits,
            smiles_maxlen=args.smiles_max_len,
        ),
        acquisition=args.acq,
        acquisition_weights=resolve_acquisition_weights(args.acq_weights),
        ehvi_n_samples=args.ehvi_n_samples,
        alpha_base_measure=args.alpha,
        eta_ehvi_tilt=args.eta,
        m1_k_direct_llm=args.m1_k_direct_llm,
        m1_q0_smoothing=resolve_q0_smoothing(args),
        m1_analog_n_llm_seeds=args.m1_analog_n_llm_seeds,
        m1_analog_k_total=args.m1_analog_k_total,
        llm_max_retries=args.llm_max_retries,
        llm_retry_wait_seconds=args.llm_retry_wait_seconds,
        trajectory_dir=args.trajectory_dir or str(output_dir),
        resume_from_trajectory=bool(args.resume),
        seed=args.seed,
        verbose=bool(args.verbose),
    )


def describe_ldm_task(args: argparse.Namespace) -> LDMTaskSpec:
    kernel_impl = "smiles-strkernel" if args.kernel == "sk" else "fingerprint+tanimoto"
    gp_feature_dimension = None if args.kernel == "sk" else int(args.gp_fp_n_bits)
    if args.method in {"m1_stratified_direct_llm_only", "m1_llm_one_step"}:
        acquisition = AcquisitionSpec(
            name="llm_order",
            objective_names=("vina", "activity"),
            score_direction="rank",
            selection_rule="evaluate candidates in LLM/reservoir order",
            parameters={"batch_size": int(args.batch_size)},
        )
    else:
        acquisition = AcquisitionSpec(
            name=str(args.acq),
            objective_names=("vina", "activity"),
            score_direction="sample",
            selection_rule=(
                "sample candidates from q0 base mass tilted by the robust-z shared "
                f"{args.acq} acquisition score"
            ),
            parameters={
                "alpha_base_measure": float(args.alpha),
                "eta_acquisition_tilt": float(args.eta),
                "acquisition_weights": list(resolve_acquisition_weights(args.acq_weights)),
                "ehvi_n_samples": int(args.ehvi_n_samples),
                "batch_size": int(args.batch_size),
            },
        )
    direct_only = args.method in {"m1_stratified_direct_llm_only", "m1_llm_one_step"}
    if direct_only:
        surrogate = SurrogateSpaceSpec(
            kind="none",
            representation="not used by direct LLM ordering",
            dimension_policy="none",
        )
    elif args.kernel == "sk":
        surrogate = SurrogateSpaceSpec(
            kind="kernel",
            representation="SMILES subsequence string kernel",
            dimension_policy="implicit",
            encoder="tasks.small_molecule.core.engine_adapters.SmilesSurrogateEncoder",
            version="smiles_strkernel_v1",
        )
    else:
        surrogate = SurrogateSpaceSpec(
            kind="vector",
            representation="fixed-length molecular fingerprint",
            dimension_policy="fixed",
            dimension=int(args.gp_fp_n_bits),
            encoder="tasks.small_molecule.core.engine_adapters.SmilesSurrogateEncoder",
            version=f"molecular_fingerprint_{int(args.gp_fp_n_bits)}_v1",
        )
    return LDMTaskSpec(
        task="small_molecule",
        candidate_domain=CandidateDomainSpec(
            name="smiles",
            kind="string",
            dimension=None,
            representation="canonical SMILES string",
            constraints={"max_smiles_len": int(args.smiles_max_len)},
            metadata={
                "gp_kernel": kernel_impl,
                "gp_feature_dimension": gp_feature_dimension,
                "max_candidates_per_round": int(args.max_candidates_per_round),
            },
        ),
        objectives=(
            ObjectiveSpec(
                name="vina",
                direction="minimize",
                description="AutoDock Vina binding score; lower is better.",
            ),
            ObjectiveSpec(
                name="activity",
                direction="maximize",
                description="KRAS G12D activity model score; higher is better.",
            ),
        ),
        response_spaces=(
            ResponseSpaceSpec(
                name="direct_smiles",
                output_kind="json",
                parser="tasks.small_molecule.core.ldm_tilted_case2.schemas.parse_m1_direct_smiles",
                description="LLM emits direct candidate SMILES without objective scores.",
                schema={
                    "type": "object",
                    "required": ["direct_smiles"],
                    "properties": {
                        "direct_smiles": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["smiles"],
                                "properties": {
                                    "smiles": {"type": "string"},
                                    "rationale": {"type": "string"},
                                },
                            },
                        },
                    },
                },
                metadata={
                    "banned_score_keys": [
                        "score",
                        "objective_score",
                        "constraint_score",
                        "acquisition_score",
                        "uncertainty",
                        "proxy_value",
                    ]
                },
            ),
            ResponseSpaceSpec(
                name="seed_plan",
                output_kind="json",
                parser="tasks.small_molecule.core.ldm_tilted_case2.schemas.parse_seed_plan",
                description="LLM emits seed SMILES and per-seed analogue budgets.",
                schema={
                    "type": "object",
                    "required": ["seeds"],
                    "properties": {
                        "seeds": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["smiles", "budget"],
                                "properties": {
                                    "smiles": {"type": "string"},
                                    "budget": {"type": "integer", "minimum": 0},
                                    "intent": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            ),
        ),
        acquisition=acquisition,
        reservoir=ReservoirSpec(
            name="molecular_candidate_reservoir",
            expansions=(
                ReservoirExpansionSpec(
                    name="direct_smiles_generation",
                    action_kind="emit_candidate",
                    response_space="direct_smiles",
                    produces_candidates=True,
                    description="Emit valid molecular candidates as SMILES.",
                ),
                ReservoirExpansionSpec(
                    name="seeded_analogue_generation",
                    action_kind="configure_generator",
                    response_space="seed_plan",
                    produces_candidates=True,
                    description="Choose molecular seeds and budgets for analogue generation.",
                ),
            ),
            candidate_validator="SMILES parse, canonicalization, and molecular constraint checks",
            deduplication_key="canonical SMILES",
            max_size=int(args.max_candidates_per_round),
            metadata={"method": args.method},
        ),
        surrogate=surrogate,
        proposal_search=ProposalSearchSpec(
            name="single_turn",
            evaluation_policy="outer_loop_acquisition_selection",
            parameters={"proposals_per_round": "task_configured_candidate_reservoir"},
        ),
        metadata={
            "method": args.method,
            "init_strategy": args.init_strategy,
            "budget": int(args.budget),
            "init_size": int(args.init_size),
        },
    )


def resolve_q0_smoothing(args: argparse.Namespace) -> float:
    if args.m1_q0_smoothing is not None:
        return float(args.m1_q0_smoothing)
    if args.method in {
        "m1_stratified_direct_llm_sir",
        "m1_stratified_direct_llm_oversample_sir",
        "m1_llm_seed_analog_oversample_sir",
    }:
        return 0.5
    return 0.0


def resolve_output_dir(args: argparse.Namespace) -> Path:
    resume_dir = resolve_resume_dir(args)
    if resume_dir is not None:
        return resume_dir
    raw = args.trajectory_dir or args.output_dir
    if raw:
        path = Path(raw)
    else:
        suffix = "mock" if args.mock else "real"
        path = Path("runs") / "tilted_case2" / f"{args.method}_{suffix}_seed={args.seed}"
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_resume_dir(args: argparse.Namespace) -> Path | None:
    if not args.resume_from:
        return None
    path = Path(args.resume_from)
    path = path if path.is_absolute() else REPO_ROOT / path
    if not path.exists():
        raise SystemExit(f"--resume-from path does not exist: {path}")
    if path.is_file():
        path = path.parent
    explicit = args.trajectory_dir or args.output_dir
    if explicit:
        explicit_path = Path(explicit)
        explicit_path = explicit_path if explicit_path.is_absolute() else REPO_ROOT / explicit_path
        if explicit_path.resolve() != path.resolve():
            raise SystemExit(
                "--resume-from cannot be combined with a different --trajectory-dir or --output-dir"
            )
    args.resume = True
    args.trajectory_dir = str(path)
    args.output_dir = str(path)
    return path


def build_llm(args: argparse.Namespace):
    if args.mock:
        return ExpandingMockCase2LLM()
    from tasks.small_molecule.core.llm_advisor.client import OpenAIChatClient
    from tasks.small_molecule.core.llm_advisor.config import LLMClientConfig

    return OpenAIChatClient(
        LLMClientConfig(
            api_key=args.api_key,
            base_url=args.llm_url.rstrip("/"),
            model=args.llm_model_name,
        ),
        temperature=args.llm_temperature,
        timeout=args.llm_timeout,
        max_tokens=args.llm_max_tokens,
        top_p=resolve_llm_top_p(args),
        presence_penalty=resolve_llm_presence_penalty(args),
        extra_body=build_llm_extra_body(args),
    )


def is_qwen35_reasoning_model(model_name: str) -> bool:
    text = str(model_name).lower().replace("_", "-").replace("/", "-")
    return "qwen3.5" in text and "coder" not in text


def use_qwen35_sampling_defaults(args: argparse.Namespace) -> bool:
    return bool(args.qwen35_sampling_defaults) and is_qwen35_reasoning_model(args.llm_model_name)


def resolve_llm_top_p(args: argparse.Namespace) -> float | None:
    if args.llm_top_p is not None:
        return float(args.llm_top_p)
    if use_qwen35_sampling_defaults(args):
        return float(QWEN35_DEFAULT_SAMPLING["top_p"])
    return None


def resolve_llm_top_k(args: argparse.Namespace) -> int | None:
    if args.llm_top_k is not None:
        return int(args.llm_top_k)
    if use_qwen35_sampling_defaults(args):
        return int(QWEN35_DEFAULT_SAMPLING["top_k"])
    return None


def resolve_llm_min_p(args: argparse.Namespace) -> float | None:
    if args.llm_min_p is not None:
        return float(args.llm_min_p)
    if use_qwen35_sampling_defaults(args):
        return float(QWEN35_DEFAULT_SAMPLING["min_p"])
    return None


def resolve_llm_presence_penalty(args: argparse.Namespace) -> float | None:
    if args.llm_presence_penalty is not None:
        return float(args.llm_presence_penalty)
    if use_qwen35_sampling_defaults(args):
        return float(QWEN35_DEFAULT_SAMPLING["presence_penalty"])
    return None


def resolve_llm_repetition_penalty(args: argparse.Namespace) -> float | None:
    if args.llm_repetition_penalty is not None:
        return float(args.llm_repetition_penalty)
    if use_qwen35_sampling_defaults(args):
        return float(QWEN35_DEFAULT_SAMPLING["repetition_penalty"])
    return None


def build_llm_extra_body(args: argparse.Namespace) -> dict[str, Any] | None:
    extra_body = parse_extra_body_json(args.llm_extra_body_json)
    top_k = resolve_llm_top_k(args)
    min_p = resolve_llm_min_p(args)
    repetition_penalty = resolve_llm_repetition_penalty(args)
    if top_k is not None:
        extra_body["top_k"] = top_k
    if min_p is not None:
        extra_body["min_p"] = min_p
    if repetition_penalty is not None:
        extra_body["repetition_penalty"] = repetition_penalty
    if args.disable_thinking:
        chat_template_kwargs = extra_body.get("chat_template_kwargs")
        if not isinstance(chat_template_kwargs, dict):
            chat_template_kwargs = {}
        chat_template_kwargs["enable_thinking"] = False
        extra_body["chat_template_kwargs"] = chat_template_kwargs
    return extra_body or None


def parse_extra_body_json(raw: str) -> dict[str, Any]:
    if not raw or not str(raw).strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--llm-extra-body-json is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("--llm-extra-body-json must decode to a JSON object")
    return dict(payload)


def build_mock_scorers():
    def vina(smiles_list: Sequence[str]) -> list[float]:
        return [-0.1 * float(len(str(smiles))) for smiles in smiles_list]

    def activity(smiles_list: Sequence[str]) -> list[float]:
        return [
            5.0 + 0.45 * str(smiles).count("N") + 0.10 * str(smiles).count("C")
            for smiles in smiles_list
        ]

    return vina, activity


def build_mock_analog_fn():
    def analog_fn(seed_smiles: Sequence[str]) -> list[str]:
        out: list[str] = []
        for seed in seed_smiles:
            text = str(seed)
            out.extend([text + "C", text + "N", text + "O"])
        return out

    return analog_fn


def build_real_scorers(args: argparse.Namespace, output_dir: Path):
    from tasks.small_molecule.core.objective_nn import NNScorer, NNScorerConfig
    from tasks.small_molecule.core.objective_vina import VinaScorer, VinaScorerConfig

    vina_cache_dir = resolve_optional_path(args.vina_cache_dir) or (output_dir / "vina_cache")
    vina_cfg = VinaScorerConfig(
        pdb_id=args.vina_pdb_id,
        chain_id=args.vina_chain_id or None,
        ligand_resname=args.vina_ligand_resname or None,
        cache_dir=vina_cache_dir,
        allow_zero_charge_fallback=bool(args.vina_allow_zero_charge_fallback),
        allow_debug_receptor=bool(args.vina_allow_debug_receptor),
        vina_bin=args.vina_bin or None,
        exhaustiveness=args.vina_exhaustiveness,
        n_poses=args.vina_n_poses,
        seed=args.vina_seed,
        max_workers=args.vina_max_workers,
        use_cache=not bool(args.vina_no_cache),
    )
    nn_cfg = NNScorerConfig(
        model_path=args.nn_model_path,
        on_error="all_nan",
    )
    return VinaScorer(vina_cfg), NNScorer(nn_cfg)


def resolve_optional_path(raw: str | None) -> Path | None:
    if raw is None or not str(raw).strip():
        return None
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    normalized = str(path).replace("\\", "/")
    if any(
        normalized == prefix or normalized.startswith(prefix + "/")
        for prefix in REPOSITORY_RELATIVE_PREFIXES
    ):
        return WORKSPACE_ROOT / path
    return REPO_ROOT / path


def build_real_analog_fn(args: argparse.Namespace, output_dir: Path):
    from tasks.small_molecule.core.analog import ReasynConfig, generate_analogs

    model_path = args.reasyn_model_path or (
        "data/trained_model/nv-reasyn-ar-166m-v2.ckpt,"
        "data/trained_model/nv-reasyn-eb-174m-v2.ckpt"
    )
    devices = [
        int(part)
        for part in str(args.reasyn_devices).split(",")
        if part.strip()
    ]
    config = ReasynConfig(
        model_path=model_path,
        reasyn_repo=args.reasyn_repo or None,
        python_bin=args.reasyn_python or None,
        devices=devices or [0],
        time_limit=args.reasyn_time_limit,
        temp_dir=output_dir / "reasyn_tmp",
    )

    def analog_fn(seed_smiles: Sequence[str]) -> list[str]:
        df = generate_analogs(list(seed_smiles), config)
        if df is None or len(df) == 0:
            return []
        return [str(smiles) for smiles in df["smiles"].tolist()]

    def generate_with_targets(seed_smiles: Sequence[str]):
        return generate_analogs(list(seed_smiles), config)

    analog_fn.generate_with_targets = generate_with_targets  # type: ignore[attr-defined]
    return analog_fn


def parse_seed_smiles(raw: str) -> list[str]:
    seeds = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not seeds:
        raise SystemExit("--seed-smiles produced an empty seed set")
    return seeds


def best_observed(history, minimize: Sequence[bool]) -> dict[str, object] | None:
    finite = [
        (smiles, scores)
        for smiles, scores in history
        if len(scores) == 2 and scores[0] is not None and scores[1] is not None
    ]
    if not finite:
        return None
    best_vina = min(finite, key=lambda item: float(item[1][0]))
    best_activity = max(finite, key=lambda item: float(item[1][1]))
    balanced = min(finite, key=lambda item: float(item[1][0]) - float(item[1][1]))
    return {
        "best_vina": {"smiles": best_vina[0], "scores": list(best_vina[1])},
        "best_activity": {"smiles": best_activity[0], "scores": list(best_activity[1])},
        "balanced_proxy": {"smiles": balanced[0], "scores": list(balanced[1])},
        "minimize": list(minimize),
    }


def config_to_json(cfg: TiltedLDMCase2Config) -> dict[str, object]:
    payload = dict(cfg.__dict__)
    payload["gp_config"] = dict(cfg.gp_config.__dict__)
    return payload


def resolve_acquisition_weights(raw: str) -> tuple[float, float]:
    values = tuple(float(value.strip()) for value in str(raw).split(",") if value.strip())
    if len(values) != 2:
        raise ValueError("--acq-weights must contain exactly two comma-separated numbers")
    if any(value < 0 for value in values) or sum(values) <= 0:
        raise ValueError("--acq-weights must be non-negative with a positive sum")
    return values


def planned_config_json(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    return {
        "method": args.method,
        "init_size": args.init_size,
        "init_strategy": args.init_strategy,
        "budget": args.budget,
        "batch_size": args.batch_size,
        "smiles_max_len": args.smiles_max_len,
        "max_candidates_per_round": args.max_candidates_per_round,
        "max_empty_reservoir_rounds": args.max_empty_reservoir_rounds,
        "allow_early_stop": bool(args.allow_early_stop),
        "minimize": [True, False],
        "ref_point": [0.0, 5.0],
        "gp_config": {
            "impl": "smiles-strkernel" if args.kernel == "sk" else "fingerprint+tanimoto",
            "device": args.gp_device,
            "fit_n_itersteps": args.gp_fit_itersteps,
            "fp_n_bits": args.gp_fp_n_bits,
            "smiles_maxlen": args.smiles_max_len,
        },
        "acquisition": args.acq,
        "acquisition_weights": list(resolve_acquisition_weights(args.acq_weights)),
        "ehvi_n_samples": args.ehvi_n_samples,
        "alpha_base_measure": args.alpha,
        "eta_ehvi_tilt": args.eta,
        "m1_k_direct_llm": args.m1_k_direct_llm,
        "m1_q0_smoothing": resolve_q0_smoothing(args),
        "m1_analog_n_llm_seeds": args.m1_analog_n_llm_seeds,
        "m1_analog_k_total": args.m1_analog_k_total,
        "llm_model_name": args.llm_model_name,
        "llm_temperature": args.llm_temperature,
        "llm_max_tokens": args.llm_max_tokens,
        "llm_top_p": resolve_llm_top_p(args),
        "llm_presence_penalty": resolve_llm_presence_penalty(args),
        "llm_extra_body": build_llm_extra_body(args),
        "llm_disable_thinking": bool(args.disable_thinking),
        "llm_qwen35_reasoning_model": is_qwen35_reasoning_model(args.llm_model_name),
        "llm_qwen35_sampling_defaults_applied": use_qwen35_sampling_defaults(args),
        "llm_max_retries": args.llm_max_retries,
        "llm_retry_wait_seconds": args.llm_retry_wait_seconds,
        "trajectory_dir": args.trajectory_dir or str(output_dir),
        "resume_from_trajectory": bool(args.resume),
        "seed": args.seed,
        "verbose": bool(args.verbose),
        "vina_cache_dir": args.vina_cache_dir or None,
        "vina_pdb_id": args.vina_pdb_id,
        "vina_chain_id": args.vina_chain_id,
        "vina_ligand_resname": args.vina_ligand_resname or None,
        "vina_exhaustiveness": args.vina_exhaustiveness,
        "vina_n_poses": args.vina_n_poses,
        "vina_seed": args.vina_seed,
        "vina_max_workers": args.vina_max_workers,
        "vina_use_cache": not bool(args.vina_no_cache),
        "vina_allow_zero_charge_fallback": bool(args.vina_allow_zero_charge_fallback),
        "vina_allow_debug_receptor": bool(args.vina_allow_debug_receptor),
    }


if __name__ == "__main__":
    raise SystemExit(main())
