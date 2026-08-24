"""Shared-campaign assembly for adaptive KV-cache quantization."""

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
    snapshot_experiment_contract,
)
from ldm_tts.transport.openai import EndpointRequestError, OpenAICompatibleProposalClient

from tasks.llm_kv_adaptive_quantization.core.candidate import QuantizerCandidateDomain
from tasks.llm_kv_adaptive_quantization.core.evaluator import (
    OFFICIAL_COMMIT,
    OFFICIAL_WORKLOADS,
    ContractThenMLSBenchEvaluator,
    MLSBenchEvaluator,
    MockQuantizerEvaluator,
    TensorContractEvaluator,
)
from tasks.llm_kv_adaptive_quantization.core.proposals import (
    DeterministicQuantizerExpander,
    EndpointQuantizerExpander,
    proposal_response_format,
    quantizer_spec_schema,
)
from tasks.llm_kv_adaptive_quantization.core.surrogate import (
    FEATURE_VERSION,
    QuantizerSourceEncoder,
)


TASK_ID = "llm_kv_adaptive_quantization"
TASK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = TASK_ROOT / "resources" / "seed_quantizer.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search adaptive KV-cache quantizers with pinned MLS-Bench."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--reservoir-size", type=int, default=4)
    parser.add_argument("--evaluations-per-round", type=int, default=1)
    parser.add_argument(
        "--proposal-mode", choices=("deterministic", "openai"), default="deterministic"
    )
    parser.add_argument("--seed-file", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--llm-url", default=os.environ.get("LDM_LLM_URL", ""))
    parser.add_argument("--llm-model-name", default=os.environ.get("LDM_LLM_MODEL", ""))
    parser.add_argument("--api-key", default=os.environ.get("LDM_LLM_API_KEY", ""))
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument("--llm-max-tokens", type=int, default=8192)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument("--workloads", default=",".join(OFFICIAL_WORKLOADS))
    parser.add_argument("--devices", default="0,1,2,3,4")
    parser.add_argument("--evaluator-python", default="")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--evaluation-timeout", type=float, default=34800.0)
    parser.add_argument("--contract-timeout", type=float, default=60.0)
    parser.add_argument("--contract-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--acquisition-beta", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    # Compatibility aliases for the original draft registration.
    parser.add_argument("--harness-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--mlsbench-task-dir", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def describe_ldm_task(args: argparse.Namespace) -> LDMTaskSpec:
    workloads = _workloads(args)
    objective = _objective_name(args, workloads)
    encoder = QuantizerSourceEncoder()
    return LDMTaskSpec(
        task=TASK_ID,
        candidate_domain=CandidateDomainSpec(
            name="AdaptiveKVQuantizer implementation",
            kind="python_class",
            dimension=None,
            representation=(
                "A complete Python class replacing only AdaptiveKVQuantizer in the "
                "pinned tensor-level replay harness."
            ),
            constraints={
                "max_source_bytes": 64000,
                "fixed_harness": True,
                "required_methods": 7,
                "editable_lines": [41, 172],
            },
        ),
        objectives=(
            ObjectiveSpec(
                name=objective,
                direction="maximize",
                description=(
                    "Official MLS-Bench five-workload aggregate score."
                    if objective == "official_score"
                    else "Non-official qualification signal for mock, contract, or tiny runs."
                ),
            ),
        ),
        response_spaces=(
            ResponseSpaceSpec(
                name="quantizer_parameter_json",
                output_kind="json",
                schema=quantizer_spec_schema(args.reservoir_size),
                parser=(
                    "tasks.llm_kv_adaptive_quantization.core.proposals:"
                    "parse_quantizer_specs"
                ),
                description=(
                    "A finite list of model-proposed quantizer parameter sets that are "
                    "materialized into the pinned contract-valid seed class."
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
            name="quantizer_policy_reservoir",
            expansions=(
                ReservoirExpansionSpec(
                    name="direct_quantizer_proposal",
                    action_kind="emit_candidate",
                    response_space="quantizer_parameter_json",
                    produces_candidates=True,
                    description=(
                        "Materialize model-proposed quantizer parameters into complete "
                        "classes using the pinned seed implementation."
                    ),
                ),
            ),
            candidate_validator=(
                "Exact AST/signature validation followed by isolated tensor preflight "
                "before official evaluation."
            ),
            deduplication_key="SHA-256 of the normalized candidate AST",
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
            "mode": _mode(args),
            "workloads": list(workloads),
            "official_suite": objective == "official_score",
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    workloads = _workloads(args)
    devices = _csv_values(args.devices)
    spec = describe_ldm_task(args)
    contract, profile_name = load_active_experiment_contract()
    payload: dict[str, Any] = {
        "task": TASK_ID,
        "mode": _mode(args),
        "proposal_mode": args.proposal_mode,
        "iterations": args.iterations,
        "workloads": list(workloads),
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
        else unique_run_dir(args.out_dir.resolve() / (args.run_name or _default_run_name(args, workloads)))
    )
    if contract is not None and args.resume_from is None:
        snapshot_experiment_contract(contract, run_dir, profile=profile_name)
    profile = contract.profile(profile_name) if contract is not None and profile_name else None
    budget = CampaignBudget(
        rounds=args.iterations,
        reservoir_size=args.reservoir_size,
        batch_size=args.evaluations_per_round,
        extra_limits=(
            dict(profile.budget)
            if profile is not None
            else _derived_budget(args, workloads)
        ),
    )

    seed_source = args.seed_file.read_text(encoding="utf-8")
    client = None
    if args.proposal_mode == "openai":
        if not args.llm_url or not args.llm_model_name:
            _pause_campaign(
                run_dir, spec, args, budget,
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
                run_dir, spec, args, budget,
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
    domain = QuantizerCandidateDomain(sink)
    expander = (
        EndpointQuantizerExpander(client, seed_source)
        if client is not None
        else DeterministicQuantizerExpander(
            seed_source,
            collectable=bool(args.mock),
        )
    )
    evaluator_python = args.evaluator_python or os.environ.get("PYTHON", "") or os.sys.executable
    tensor_contract = TensorContractEvaluator(
        timeout_seconds=args.contract_timeout,
        device=args.contract_device,
        python_executable=evaluator_python,
    )
    if args.mock:
        evaluator = MockQuantizerEvaluator()
    elif args.preflight:
        evaluator = tensor_contract
    else:
        upstream_root, package_dir = _real_paths(args)
        if upstream_root is None or package_dir is None:
            raise SystemExit("Real evaluation requires --upstream-root and --package-dir")
        evaluator = ContractThenMLSBenchEvaluator(
            tensor_contract,
            MLSBenchEvaluator(
                package_dir=package_dir,
                upstream_root=upstream_root,
                run_dir=run_dir,
                workloads=workloads,
                devices=devices,
                model_id=args.model_id,
                max_examples=args.max_examples,
                timeout_seconds=args.evaluation_timeout,
                cpu=args.cpu,
                evaluator_python=evaluator_python,
            ),
        )

    encoder = QuantizerSourceEncoder()
    selector = RBFGPUCBSelector(
        objective_name=spec.objectives[0].name,
        beta=args.acquisition_beta,
        feature_version=FEATURE_VERSION,
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
                context={"workloads": list(workloads), "profile": profile_name},
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
            run_dir, spec, args, budget,
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
    budget: CampaignBudget,
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
        budget_limits=budget.runtime_limits(),
        resume=(run_dir / "campaign.json").exists(),
    )
    runtime.pause(status, phase=phase, message=message, details=details)
    payload["run_dir"] = str(run_dir.resolve())
    payload["status"] = status
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
    if args.max_examples < 0:
        raise SystemExit("--max-examples must be non-negative")


def _workloads(args: argparse.Namespace) -> tuple[str, ...]:
    workloads = _csv_values(args.workloads)
    if not workloads:
        raise SystemExit("--workloads must select at least one workload")
    unknown = sorted(set(workloads) - set(OFFICIAL_WORKLOADS))
    if unknown:
        raise SystemExit("Unknown workload(s): " + ", ".join(unknown))
    if len(workloads) != len(set(workloads)):
        raise SystemExit("--workloads may not contain duplicates")
    return workloads


def _csv_values(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(raw).split(",") if item.strip())


def _objective_name(args: argparse.Namespace, workloads: tuple[str, ...]) -> str:
    return (
        "official_score"
        if not args.mock
        and not args.preflight
        and set(workloads) == set(OFFICIAL_WORKLOADS)
        and len(workloads) == len(OFFICIAL_WORKLOADS)
        else "selection_score"
    )


def _mode(args: argparse.Namespace) -> str:
    if args.mock:
        return "mock"
    if args.preflight:
        return "preflight"
    return "real"


def _default_run_name(args: argparse.Namespace, workloads: tuple[str, ...]) -> str:
    suffix = "official_suite" if set(workloads) == set(OFFICIAL_WORKLOADS) else workloads[0]
    return f"{_mode(args)}_{suffix}"


def _derived_budget(
    args: argparse.Namespace, workloads: tuple[str, ...]
) -> dict[str, int]:
    selected = args.iterations * args.evaluations_per_round
    jobs_per_evaluation = 1 if args.mock or args.preflight else len(workloads)
    return {
        "outer_iterations": args.iterations,
        "llm_requests": args.iterations if args.proposal_mode == "openai" else 0,
        "proposal_attempts": args.iterations if args.proposal_mode == "openai" else 0,
        "valid_search_candidates": args.iterations * args.reservoir_size,
        "selected_candidates": selected,
        "external_evaluations": selected,
        "expensive_evaluation_attempts": selected,
        "successful_evaluations": selected,
        "benchmark_jobs": selected * jobs_per_evaluation,
    }


def _real_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    upstream_root = args.upstream_root
    package_dir = args.package_dir
    if package_dir is None and args.harness_file is not None:
        package_dir = args.harness_file.parent
    if upstream_root is None and args.mlsbench_task_dir is not None:
        task_dir = args.mlsbench_task_dir.resolve()
        if len(task_dir.parents) >= 3:
            upstream_root = task_dir.parents[2]
    return upstream_root, package_dir


def _materialize_search_artifacts(runtime: CampaignRuntime, engine_result=None) -> None:
    del engine_result  # reserved for projector signature compatibility
    events = runtime.events()
    reservoir_events = [item for item in events if item.get("event_type") == "reservoir_built"]
    selection_events = [item for item in events if item.get("event_type") == "candidates_selected"]
    atomic_json_write(
        runtime.run_dir / "search_manifest.json",
        {
            "schema_version": 1,
            "task": TASK_ID,
            "run_id": runtime.run_id,
            "rounds": reservoir_events,
        },
    )
    atomic_json_write(
        runtime.run_dir / "selection_record.json",
        {
            "schema_version": 1,
            "task": TASK_ID,
            "run_id": runtime.run_id,
            "selections": selection_events,
        },
    )


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "api_key"
    }
