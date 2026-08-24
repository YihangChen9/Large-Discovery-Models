"""Shared-campaign assembly for mutation-effect predictor search."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ldm_tts.campaign import CampaignBudget, CampaignRecipe, CampaignRequest, run_campaign
from ldm_tts.contracts import (
    AcquisitionSpec,
    CandidateDomainSpec,
    LDMTaskSpec,
    ObjectiveSpec,
    ProposalSearchSpec,
    ReservoirExpansionSpec,
    ReservoirSpec,
    ResponseSpaceSpec,
)
from ldm_tts.data import DataCollectionSink
from ldm_tts.engine.run_store import CampaignRuntime, atomic_json_write, unique_run_dir
from ldm_tts.optimization.gp import RBFGPUCBSelector
from ldm_tts.registration.experiment import (
    load_active_experiment_contract,
    load_experiment_contract,
    snapshot_experiment_contract,
)
from ldm_tts.transport.openai import EndpointRequestError, OpenAICompatibleProposalClient

from tasks.ai4bio_mutation_effect_prediction.core.candidate import (
    EMBED_DIM,
    PARAMETER_LIMIT,
    MutationPredictorCandidateDomain,
)
from tasks.ai4bio_mutation_effect_prediction.core.evaluator import (
    MLSBenchMutationEvaluator,
    OFFICIAL_ASSAYS,
    OFFICIAL_COMMIT,
    MockMutationEvaluator,
)
from tasks.ai4bio_mutation_effect_prediction.core.proposals import (
    DeterministicPredictorExpander,
    EndpointPredictorExpander,
    predictor_spec_schema,
    proposal_response_format,
)
from tasks.ai4bio_mutation_effect_prediction.core.surrogate import (
    FEATURE_VERSION,
    PredictorSpecEncoder,
)


TASK_ID = "ai4bio_mutation_effect_prediction"
TASK_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search bounded mutation-effect predictor architectures."
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--reservoir-size", type=int, default=4)
    parser.add_argument("--evaluations-per-round", type=int, default=1)
    parser.add_argument(
        "--proposal-mode", choices=("deterministic", "openai"), default="deterministic"
    )
    parser.add_argument("--llm-url", default=os.environ.get("LDM_LLM_URL", ""))
    parser.add_argument("--llm-model-name", default=os.environ.get("LDM_LLM_MODEL", ""))
    parser.add_argument("--api-key", default=os.environ.get("LDM_LLM_API_KEY", ""))
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument("--llm-max-tokens", type=int, default=4096)
    parser.add_argument("--acquisition-beta", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--cv-dir", type=Path)
    parser.add_argument("--evaluation-timeout", type=float, default=3540.0)
    parser.add_argument("--evaluator-python", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def describe_ldm_task(args: argparse.Namespace) -> LDMTaskSpec:
    objective = "selection_score" if args.mock else "official_score"
    encoder = PredictorSpecEncoder()
    return LDMTaskSpec(
        task=TASK_ID,
        candidate_domain=CandidateDomainSpec(
            name="MutationPredictor architecture specification",
            kind="bounded_neural_architecture",
            dimension=None,
            representation=(
                "A strict JSON specification materialized into the upstream PyTorch "
                "MutationPredictor(embed_dim=1280) interface."
            ),
            constraints={
                "embed_dim": EMBED_DIM,
                "max_hidden_layers": 3,
                "max_parameters": PARAMETER_LIMIT,
                "fixed_pipeline": True,
            },
        ),
        objectives=(
            ObjectiveSpec(
                name=objective,
                direction="maximize",
                description=(
                    "Official baseline-normalized score from the pinned MLS-Bench scorer."
                    if objective == "official_score"
                    else "Synthetic qualification signal; not benchmark-comparable."
                ),
            ),
        ),
        response_spaces=(
            ResponseSpaceSpec(
                name="predictor_spec_json",
                output_kind="json",
                schema=predictor_spec_schema(args.reservoir_size),
                parser=(
                    "tasks.ai4bio_mutation_effect_prediction.core.proposals:"
                    "parse_predictor_specs"
                ),
                description=(
                    "A finite list of bounded architecture and optimizer specifications; "
                    "arbitrary generated source is not admitted."
                ),
            ),
        ),
        acquisition=AcquisitionSpec(
            name="gp_ucb",
            objective_names=(objective,),
            score_direction="maximize",
            selection_rule="Highest shared exact-RBF GP upper confidence bound.",
            parameters={"beta": args.acquisition_beta},
        ),
        reservoir=ReservoirSpec(
            name="mutation_predictor_reservoir",
            expansions=(
                ReservoirExpansionSpec(
                    name="direct_architecture_proposal",
                    action_kind="emit_candidate",
                    response_space="predictor_spec_json",
                    produces_candidates=True,
                    description=(
                        "Materialize accepted architecture specs into benchmark-compatible "
                        "MutationPredictor classes."
                    ),
                ),
            ),
            candidate_validator=(
                "Strict schema, bounds, finite optimizer values, and exact analytical "
                "parameter-budget validation before evaluation."
            ),
            deduplication_key="SHA-256 of canonical predictor specification JSON",
            max_size=args.reservoir_size,
        ),
        surrogate=encoder.describe(),
        proposal_search=ProposalSearchSpec(
            name="single_turn_best_of_n",
            breadth=args.reservoir_size,
            evaluation_policy="acquisition_selected",
        ),
        metadata={
            "benchmark_commit": OFFICIAL_COMMIT,
            "assays": list(OFFICIAL_ASSAYS),
            "mode": "mock" if args.mock else "real",
            "official_suite": not args.mock,
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    spec = describe_ldm_task(args)
    contract, profile_name = load_active_experiment_contract()
    if contract is None:
        contract = load_experiment_contract(TASK_ROOT / "experiment.json")
    payload: dict[str, Any] = {
        "task": TASK_ID,
        "mode": "mock" if args.mock else "real",
        "proposal_mode": args.proposal_mode,
        "iterations": args.iterations,
        "objective": spec.objectives[0].name,
        "contract_profile": profile_name,
        "contract_sha256": "" if contract is None else contract.digest,
        "ldm_task_spec": spec.to_dict(),
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    run_dir = (
        args.resume_from.resolve()
        if args.resume_from is not None
        else unique_run_dir(args.out_dir.resolve() / (args.run_name or "mock"))
    )
    if contract is not None and args.resume_from is None:
        snapshot_experiment_contract(contract, run_dir, profile=profile_name)

    budget = CampaignBudget(
        rounds=args.iterations,
        reservoir_size=args.reservoir_size,
        batch_size=args.evaluations_per_round,
        extra_limits={
            "llm_requests": args.iterations if args.proposal_mode == "openai" else 0,
            "proposal_attempts": args.iterations if args.proposal_mode == "openai" else 0,
            "benchmark_jobs": args.iterations
            * args.evaluations_per_round
            * (1 if args.mock else len(OFFICIAL_ASSAYS)),
        },
    )

    client = None
    if args.proposal_mode == "openai":
        if not args.llm_url or not args.llm_model_name:
            _pause_campaign(
                run_dir, spec, args,
                "paused_endpoint_unavailable", "endpoint_preflight",
                "OpenAI proposal mode requires an endpoint URL and model name.",
                payload,
            )
            return 2
        client = OpenAICompatibleProposalClient(
            url=args.llm_url,
            model=args.llm_model_name,
            api_key=args.api_key,
            timeout_seconds=args.llm_timeout,
            max_tokens=args.llm_max_tokens,
            temperature=0.3,
            max_retries=0,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": proposal_response_format(args.reservoir_size),
            },
        )
        try:
            preflight = client.preflight()
        except Exception as exc:
            _pause_campaign(
                run_dir, spec, args,
                "paused_endpoint_unavailable", "endpoint_preflight",
                str(exc),
                payload,
                details={"model": args.llm_model_name},
            )
            return 2
        payload["endpoint_preflight"] = preflight
        _preflight_record = preflight
    else:
        _preflight_record = None

    sink = DataCollectionSink.from_env(default_root=run_dir / "ldm_data")
    domain = MutationPredictorCandidateDomain(sink)
    expander = (
        EndpointPredictorExpander(client)
        if client is not None
        else DeterministicPredictorExpander(collectable=bool(args.mock))
    )
    encoder = PredictorSpecEncoder()
    selector = RBFGPUCBSelector(
        objective_name=spec.objectives[0].name,
        beta=args.acquisition_beta,
        feature_version=FEATURE_VERSION,
    )
    if args.mock:
        evaluator = MockMutationEvaluator()
    else:
        missing = [
            name
            for name, value in (
                ("--upstream-root", args.upstream_root),
                ("--data-dir", args.data_dir),
                ("--cv-dir", args.cv_dir),
            )
            if value is None
        ]
        if missing:
            raise SystemExit("Real evaluation requires " + ", ".join(missing))
        evaluator = MLSBenchMutationEvaluator(
            upstream_root=args.upstream_root,
            data_dir=args.data_dir,
            cv_dir=args.cv_dir,
            run_dir=run_dir,
            timeout_seconds=args.evaluation_timeout,
            evaluator_python=args.evaluator_python or os.sys.executable,
        )
    try:
        result = run_campaign(
            CampaignRequest(
                run_dir=run_dir,
                budget=budget,
                config=_jsonable_args(args),
                resume=args.resume_from is not None,
                contract_sha256="" if contract is None else contract.digest,
                contract_profile=profile_name,
                context={"assays": list(OFFICIAL_ASSAYS), "profile": profile_name},
                artifact_projector=_materialize_search_artifacts,
                runtime_hook=(
                    lambda runtime: setattr(
                        expander,
                        "before_request",
                        lambda: runtime.consume("llm_requests"),
                    )
                    if client is not None
                    else None
                ),
            ),
            CampaignRecipe(
                task_spec=spec,
                expander=expander,
                candidate_domain=domain,
                evaluator=evaluator,
                surrogate_encoder=encoder,
                selector=selector,
            ),
        )
    except EndpointRequestError as exc:
        _pause_campaign(
            run_dir, spec, args,
            "paused_endpoint_unavailable", "reservoir_expansion",
            str(exc),
            payload,
            details={"model": args.llm_model_name},
        )
        return 2
    if _preflight_record is not None:
        result.runtime.record("endpoint_preflight_succeeded", _preflight_record)
    payload["engine_summary"] = result.engine.summary
    payload["run_dir"] = str(run_dir.resolve())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.engine.summary["successful_evaluation_count"] else 1


def _pause_campaign(
    run_dir: Path,
    spec: LDMTaskSpec,
    args: argparse.Namespace,
    status: str,
    phase: str,
    message: str,
    payload: dict[str, Any],
    *,
    details: dict[str, Any] | None = None,
) -> None:
    """Persist a resumable pause outside the shared campaign lifecycle."""
    runtime = CampaignRuntime.open(
        run_dir,
        task=TASK_ID,
        config=_jsonable_args(args),
        task_spec=spec,
        budget_limits=None,
        resume=(run_dir / "campaign.json").exists(),
    )
    runtime.pause(status, phase=phase, message=message, details=details)
    payload.update(run_dir=str(run_dir.resolve()), status=status)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _validate_args(args: argparse.Namespace) -> None:
    if args.iterations < 0:
        raise SystemExit("--iterations must be non-negative")
    if args.reservoir_size < 1:
        raise SystemExit("--reservoir-size must be positive")
    if args.evaluations_per_round < 1:
        raise SystemExit("--evaluations-per-round must be positive")
    if args.evaluations_per_round > args.reservoir_size:
        raise SystemExit("--evaluations-per-round cannot exceed --reservoir-size")
    if args.acquisition_beta < 0:
        raise SystemExit("--acquisition-beta must be non-negative")
    if not 0 < args.evaluation_timeout <= 3540:
        raise SystemExit("--evaluation-timeout must be in (0, 3540]")


def _materialize_search_artifacts(runtime: CampaignRuntime, engine_result=None) -> None:
    del engine_result  # reserved for projector signature compatibility
    events = runtime.events()
    atomic_json_write(
        runtime.run_dir / "search_manifest.json",
        {
            "schema_version": 1,
            "task": TASK_ID,
            "run_id": runtime.run_id,
            "rounds": [
                event for event in events if event.get("event_type") == "reservoir_built"
            ],
        },
    )
    atomic_json_write(
        runtime.run_dir / "selection_record.json",
        {
            "schema_version": 1,
            "task": TASK_ID,
            "run_id": runtime.run_id,
            "selections": [
                event
                for event in events
                if event.get("event_type") == "candidates_selected"
            ],
        },
    )


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "api_key"
    }
