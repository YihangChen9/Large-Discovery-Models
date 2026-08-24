"""Shared-campaign assembly for discrete causal discovery."""

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
from ldm_tts.engine.reporting import build_campaign_result, build_trajectory_rows, load_successful_observations, write_trajectory_csv
from ldm_tts.engine.run_store import CampaignRuntime, atomic_json_write, unique_run_dir
from ldm_tts.optimization.gp import RBFGPUCBSelector
from ldm_tts.registration.experiment import load_active_experiment_contract, load_experiment_contract, snapshot_experiment_contract

from tasks.causal_discovery_discrete.core.candidate import CausalAlgorithmCandidateDomain
from tasks.causal_discovery_discrete.core.evaluator import MLSBenchCausalEvaluator, MockCausalEvaluator, OFFICIAL_CASES, OFFICIAL_COMMIT
from tasks.causal_discovery_discrete.core.proposals import DeterministicCausalExpander, proposal_schema
from tasks.causal_discovery_discrete.core.surrogate import CausalSpecEncoder, FEATURE_VERSION


TASK_ID = "causal_discovery_discrete"
TASK_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search bounded discrete causal-discovery algorithms.")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--reservoir-size", type=int, default=4)
    parser.add_argument("--evaluations-per-round", type=int, default=1)
    parser.add_argument("--proposal-mode", choices=("deterministic",), default="deterministic")
    parser.add_argument("--acquisition-beta", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--evaluation-timeout", type=float, default=3540.0)
    parser.add_argument("--evaluator-python", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def describe_ldm_task(args: argparse.Namespace) -> LDMTaskSpec:
    objective = "selection_score" if args.mock else "official_score"
    encoder = CausalSpecEncoder()
    return LDMTaskSpec(
        task=TASK_ID,
        candidate_domain=CandidateDomainSpec(
            name="Discrete causal skeleton algorithm specification",
            kind="bounded_algorithm_configuration",
            dimension=2,
            representation="A normalized-mutual-information threshold and hard maximum degree.",
            constraints={"min_association": [0.0, 1.0], "max_degree": [1, 20], "arbitrary_code": False},
        ),
        objectives=(ObjectiveSpec(
            name=objective,
            direction="maximize",
            description=(
                "Pinned MLS-Bench bounded-power geometric aggregate across five bnlearn networks."
                if not args.mock else "Synthetic qualification signal; not benchmark-comparable."
            ),
        ),),
        response_spaces=(ResponseSpaceSpec(
            name="causal_algorithm_spec_json",
            output_kind="json",
            schema=proposal_schema(args.reservoir_size),
            parser="tasks.causal_discovery_discrete.core.candidate:normalize_algorithm_spec",
            description="A finite list of bounded association-threshold and degree-cap specifications.",
        ),),
        acquisition=AcquisitionSpec(
            name="gp_ucb",
            objective_names=(objective,),
            score_direction="maximize",
            selection_rule="Highest shared exact-RBF GP upper confidence bound.",
            parameters={"beta": args.acquisition_beta},
        ),
        reservoir=ReservoirSpec(
            name="causal_algorithm_reservoir",
            expansions=(ReservoirExpansionSpec(
                name="direct_algorithm_proposal",
                action_kind="emit_candidate",
                response_space="causal_algorithm_spec_json",
                produces_candidates=True,
                description="Emit bounded algorithm configurations from the versioned finite catalog.",
            ),),
            candidate_validator="Strict field set, finite threshold bounds, and integer degree limits.",
            deduplication_key="SHA-256 of canonical algorithm specification JSON",
            max_size=args.reservoir_size,
        ),
        surrogate=encoder.describe(),
        proposal_search=ProposalSearchSpec(name="single_turn_best_of_n", breadth=args.reservoir_size, evaluation_policy="acquisition_selected"),
        metadata={
            "benchmark_commit": OFFICIAL_COMMIT,
            "cases": list(OFFICIAL_CASES),
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
        "iterations": args.iterations,
        "contract_profile": profile_name,
        "contract_sha256": contract.digest,
        "ldm_task_spec": spec.to_dict(),
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    run_dir = args.resume_from.resolve() if args.resume_from else unique_run_dir(args.out_dir / (args.run_name or "mock"))
    if args.resume_from is None:
        snapshot_experiment_contract(contract, run_dir, profile=profile_name)
    sink = DataCollectionSink.from_env(default_root=run_dir / "ldm_data")
    domain = CausalAlgorithmCandidateDomain(sink)
    expander = DeterministicCausalExpander(collectable=bool(args.mock))
    encoder = CausalSpecEncoder()
    selector = RBFGPUCBSelector(objective_name=spec.objectives[0].name, beta=args.acquisition_beta, feature_version=FEATURE_VERSION)
    if args.mock:
        evaluator = MockCausalEvaluator()
    else:
        if args.upstream_root is None:
            raise SystemExit("Real evaluation requires --upstream-root")
        evaluator = MLSBenchCausalEvaluator(
            upstream_root=args.upstream_root,
            run_dir=run_dir,
            timeout_seconds=args.evaluation_timeout,
            evaluator_python=args.evaluator_python or os.sys.executable,
        )
    result = run_campaign(
        CampaignRequest(
            run_dir=run_dir,
            budget=CampaignBudget(
                rounds=args.iterations,
                reservoir_size=args.reservoir_size,
                batch_size=args.evaluations_per_round,
                extra_limits={
                    "llm_requests": 0,
                    "proposal_attempts": 0,
                    "benchmark_jobs": args.iterations
                    * args.evaluations_per_round
                    * (1 if args.mock else len(OFFICIAL_CASES)),
                },
            ),
            config=_jsonable_args(args),
            resume=args.resume_from is not None,
            context={"cases": list(OFFICIAL_CASES), "profile": profile_name},
            artifact_projector=lambda runtime, engine_result: _materialize_artifacts(
                runtime, objective=spec.objectives[0].name
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
    payload["engine_summary"] = result.engine.summary
    payload["run_dir"] = str(run_dir.resolve())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.engine.summary["successful_evaluation_count"] else 1


def _validate_args(args: argparse.Namespace) -> None:
    if args.iterations < 0:
        raise SystemExit("--iterations must be non-negative")
    if args.reservoir_size < 1:
        raise SystemExit("--reservoir-size must be positive")
    if not 1 <= args.evaluations_per_round <= args.reservoir_size:
        raise SystemExit("--evaluations-per-round must be positive and no larger than the reservoir")
    if args.acquisition_beta < 0:
        raise SystemExit("--acquisition-beta must be non-negative")
    if not 0 < args.evaluation_timeout <= 3540:
        raise SystemExit("--evaluation-timeout must be in (0, 3540]")


def _materialize_artifacts(runtime: CampaignRuntime, *, objective: str) -> None:
    events = runtime.events()
    atomic_json_write(runtime.run_dir / "search_manifest.json", {"schema_version": 1, "task": TASK_ID, "run_id": runtime.run_id, "rounds": [event for event in events if event.get("event_type") == "reservoir_built"]})
    atomic_json_write(runtime.run_dir / "selection_record.json", {"schema_version": 1, "task": TASK_ID, "run_id": runtime.run_id, "selections": [event for event in events if event.get("event_type") == "candidates_selected"]})
    observations = load_successful_observations(runtime.run_dir / "checkpoint.json")
    rows = build_trajectory_rows(observations, objective_name=objective, direction="maximize")
    write_trajectory_csv(runtime.run_dir / "trajectory.csv", rows, fieldnames=("evaluation", "round", "candidate_id", objective, f"best_{objective}"))
    atomic_json_write(runtime.run_dir / "result.json", build_campaign_result(runtime.run_dir, objective_name=objective, direction="maximize"))


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
