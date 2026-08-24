#!/usr/bin/env python3
"""Antibody task workflow for the shared LDM-TTS config runner.

This module wires the native AntBO LDM acquisition loop in
``tasks.antibody.core.ldm_light.ldm_acq``:

    LLM warmup / strategy proposal -> GP acquisition candidate search -> Absolut scoring

The real run uses the same config/oracle path documented in the root READMEs.
The mock mode swaps in a random evaluator and deterministic fake LLM so the
control flow can be smoke-tested without Absolut or an LLM endpoint.

Example dry run:

    python -m tasks.antibody.ldm_task.procedure \
        --antigen 1ADQ_A \
        --budget 40 \
        --dry-run

Use a GP confidence-bound acquisition after warmup:

    python -m tasks.antibody.ldm_task.procedure \
        --antigen 1ADQ_A \
        --budget 200 \
        --acq lcb \
        --acq-beta 2.0

Example mock smoke run in the DGM environment:

    python -m tasks.antibody.ldm_task.procedure \
        --mock \
        --antigen SMOKE_ANTIGEN \
        --budget 4 \
        --n-init 3 \
        --parallel-budget 8 \
        --out-dir runs/antbo_tts_mock

Example real run:

    python -m tasks.antibody.ldm_task.procedure \
        --config resources/default_config.yaml \
        --antigens-file resources/antigens.txt \
        --seed 42 \
        --budget 200 \
        --n-init 20 \
        --parallel-budget 600 \
        --llm-url http://127.0.0.1:52313/v1 \
        --llm-model-name Qwen3-Coder-30B-A3B-Instruct \
        --llm-temperature 0.7 \
        --out-dir runs/experiments/formal_5ag5seed200/antbo_tts
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def find_repo_root(start: Path) -> Path:
    for path in [start.parent, *start.parents]:
        if (path / "core").is_dir() and (path / "core" / "__init__.py").exists():
            return path
    raise RuntimeError(f"Could not find AntBO repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
WORKSPACE_ROOT = REPO_ROOT.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

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
from ldm_tts.optimization.acquisition import SINGLE_OBJECTIVE_ACQUISITIONS
from tasks.antibody.core.ldm_light.methods import (
    METHOD_CHOICES,
    METHOD_SPECS,
    normalize_method,
)


class DeterministicAntBOTTSLLM:
    """Fake LLM for local TTS control-flow checks."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.sequence_counter = 0

    def _direct_sequence(self) -> str:
        alphabet = "GPQST"
        value = self.sequence_counter
        self.sequence_counter += 1
        sequence = []
        for _ in range(11):
            sequence.append(alphabet[value % len(alphabet)])
            value //= len(alphabet)
        return "".join(sequence)

    def call(self, prompt: str, temperature: float = 0.25, timeout_s: int = 30) -> str:
        self.calls.append({
            "temperature": temperature,
            "timeout_s": timeout_s,
            "prompt_prefix": prompt[:200],
        })
        if '"task": "direct_cdrh3_generation"' in prompt:
            payload = json.loads(prompt)
            count = int(payload["constraints"]["num_sequences"])
            return json.dumps([self._direct_sequence() for _ in range(count)])
        if '"candidate_pool"' in prompt:
            return json.dumps({"selected": [{"id": 0, "sequence": "", "score": 1.0}]})
        if '"task": "sample_one_antibody_search_policy"' in prompt:
            return json.dumps({
                "rationale": "deterministic independent mock policy",
                "trust_region": "LatinHyperCubeSampling(num=8)",
                "update_bias": None,
            })
        return json.dumps({
            "rationale": "deterministic mock TTS search",
            "update_trust_region": "LatinHyperCubeSampling(num=8)",
        })

    def call_many(
        self,
        prompt: str,
        temperature: float = 0.25,
        timeout_s: int = 30,
        n: int = 1,
    ) -> list[str]:
        return [self.call(prompt, temperature, timeout_s) for _ in range(int(n))]

    def close(self) -> None:
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AntBO CDRH3 test-time search through the native LDM acquisition loop."
    )
    parser.add_argument("--config", default="resources/default_config.yaml")
    parser.add_argument("--antigens-file", default="")
    parser.add_argument("--antigen", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--budget", "--n-evals", dest="n_evals", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--parallel-budget", type=int, default=600)
    parser.add_argument(
        "--method",
        type=normalize_method,
        choices=METHOD_CHOICES,
        default="policy_max",
        help="Antibody proposal reservoir and acquisition reduction rule.",
    )
    parser.add_argument("--gen-m", type=int, default=5)
    parser.add_argument("--n-strategies", type=int, default=5)
    parser.add_argument(
        "--planner-mode",
        choices=("choices", "independent"),
        default="choices",
        help="Sample direct candidates or policies through one multi-choice request or independent requests.",
    )
    parser.add_argument("--softmax-eta", type=float, default=1.0)
    parser.add_argument(
        "--per-strategy-budget",
        type=int,
        default=0,
        help="Per-policy candidate budget; zero divides --parallel-budget across policies.",
    )
    parser.add_argument("--pool-score", choices=("acq", "combined"), default="acq")
    parser.add_argument("--selection-score", choices=("acq", "combined"), default="acq")
    parser.add_argument("--bias-weight", type=float, default=0.05)
    parser.add_argument("--sample-timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="",
        help="Override the AntBO GP/acquisition device from the task config.",
    )
    parser.add_argument(
        "--acq",
        "--acquisition",
        dest="acq",
        choices=sorted(SINGLE_OBJECTIVE_ACQUISITIONS),
        default="ei",
        help=(
            "Shared GP posterior acquisition used after warmup. All modes are "
            "internally scored so larger is better."
        ),
    )
    parser.add_argument(
        "--acq-beta",
        type=float,
        default=1.0,
        help="Exploration coefficient for LCB/UCB acquisitions.",
    )
    parser.add_argument(
        "--acq-xi",
        type=float,
        default=0.001,
        help="Expected-improvement margin used by EI.",
    )
    parser.add_argument("--out-dir", "--trajectory-dir", dest="out_dir", default="")
    parser.add_argument(
        "--absolut-path",
        default=os.environ.get("ABSOLUT_PATH", ""),
        help="Absolut installation root (default: ABSOLUT_PATH).",
    )
    parser.add_argument("--temperature", "--llm-temperature", dest="temperature", type=float, default=0.25)
    parser.add_argument("--llm-url", default=os.environ.get("LLM_BASE_URL", ""))
    parser.add_argument(
        "--llm-model-name",
        default=os.environ.get("LLM_MODEL_NAME") or os.environ.get("LLM_MODEL", ""),
    )
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""))
    parser.add_argument("--llm-max-tokens", type=int, default=None)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--history-top-k", type=int, default=10)
    parser.add_argument(
        "--disable-thinking",
        "--disable-reasoning",
        dest="disable_thinking",
        action="store_true",
        default=None,
        help="For Qwen-style reasoning models, request hidden thinking/reasoning to be disabled.",
    )
    parser.add_argument(
        "--enable-thinking",
        "--enable-reasoning",
        dest="disable_thinking",
        action="store_false",
        help="Force hidden thinking/reasoning to remain enabled, overriding LLM_DISABLE_THINKING.",
    )
    parser.add_argument("--include-antigen-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fallback-random", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def load_config(path: str) -> dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def sanitize_run_tag_part(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._=-]+", "-", text)
    text = text.strip("-._")
    return text or "unset"


def default_out_dir_base(args: argparse.Namespace) -> Path:
    if args.out_dir:
        return resolve_path(args.out_dir)
    suffix = "mock" if args.mock else "real"
    return REPO_ROOT / "runs" / "antbo_tts" / f"{suffix}_seed={args.seed}"


def make_run_tag(args: argparse.Namespace) -> str:
    mode = "mock" if args.mock else "real"
    if args.antigen:
        antigen_part = f"antigen-{sanitize_run_tag_part(args.antigen)}"
    elif args.antigens_file:
        antigen_part = f"antigens-{sanitize_run_tag_part(Path(args.antigens_file).stem)}"
    else:
        antigen_part = "antigens-unset"
    model_name = args.llm_model_name or os.getenv("LLM_MODEL", "env")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [
        mode,
        sanitize_run_tag_part(args.acq),
        f"budget{args.n_evals}",
        f"method-{args.method}",
        f"init{args.n_init}",
        f"parallel{args.parallel_budget}",
        f"batch{args.batch_size}",
        f"seed{args.seed}",
        f"trials{args.n_trials}",
        antigen_part,
        f"model-{sanitize_run_tag_part(model_name)}",
        timestamp,
    ]
    return "_".join(parts)


def resolve_out_dir(args: argparse.Namespace) -> Path:
    resolved = getattr(args, "_resolved_out_dir", None)
    if resolved:
        return Path(resolved)
    base = default_out_dir_base(args)
    out_dir = base / make_run_tag(args)
    args._resolved_out_dir = str(out_dir)
    return out_dir


def resolve_antigens(args: argparse.Namespace) -> list[str]:
    if args.antigen:
        return [args.antigen]
    if not args.antigens_file:
        raise SystemExit("Provide either --antigen or --antigens-file.")
    path = resolve_path(args.antigens_file)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def planned_config_json(args: argparse.Namespace, config: dict[str, Any], antigens: list[str]) -> dict[str, Any]:
    bbox = dict(config.get("bbox", {}))
    if args.mock:
        bbox["tool"] = "random"
        bbox["path"] = "/tmp"
        bbox["process"] = 1
        bbox["startTask"] = 0
    return {
        "repo_root": str(REPO_ROOT),
        "config": str(resolve_path(args.config)),
        "antigens": antigens,
        "seed": args.seed,
        "n_trials": args.n_trials,
        "budget": args.n_evals,
        "batch_size": args.batch_size,
        "n_init": args.n_init,
        "parallel_budget": args.parallel_budget,
        "method": args.method,
        "method_spec": METHOD_SPECS[args.method],
        "gen_m": args.gen_m,
        "n_strategies": args.n_strategies,
        "planner_mode": args.planner_mode,
        "softmax_eta": args.softmax_eta,
        "per_strategy_budget": args.per_strategy_budget,
        "pool_score": args.pool_score,
        "selection_score": args.selection_score,
        "bias_weight": args.bias_weight,
        "sample_timeout_s": args.sample_timeout_s,
        "device": str(config.get("device") or "cuda"),
        "acq": args.acq,
        "acq_beta": args.acq_beta,
        "acq_xi": args.acq_xi,
        "out_dir_base": str(default_out_dir_base(args)),
        "out_dir": str(resolve_out_dir(args)),
        "mock": bool(args.mock),
        "bbox": bbox,
        "llm_env": {
            "LLM_API_KEY": bool(args.api_key or os.getenv("LLM_API_KEY") or args.llm_url),
            "LLM_BASE_URL": args.llm_url or os.getenv("LLM_BASE_URL", ""),
            "LLM_MODEL": args.llm_model_name or os.getenv("LLM_MODEL", ""),
            "temperature": args.temperature,
            "max_tokens": args.llm_max_tokens,
            "disable_thinking": args.disable_thinking,
        },
        "ldm_task_spec": describe_ldm_task(args, config, antigens).to_dict(),
    }


def describe_ldm_task(
    args: argparse.Namespace,
    config: dict[str, Any],
    antigens: list[str] | None = None,
) -> LDMTaskSpec:
    seq_len = int(config.get("seq_len", 11))
    acq_name = str(args.acq).lower()
    method_spec = METHOD_SPECS[args.method]
    return LDMTaskSpec(
        task="antibody",
        candidate_domain=CandidateDomainSpec(
            name="cdrh3_sequence",
            kind="categorical_sequence",
            dimension=seq_len,
            representation="fixed-length amino-acid sequence encoded as categorical indices",
            constraints={
                "alphabet": "ACDEFGHIKLMNPQRSTVWY",
                "alphabet_size": 20,
                "max_cysteine": 1,
                "max_hydrophobic_run": 4,
                "max_aromatic_FWY": 2,
                "net_charge_range": [-1.0, 2.0],
                "forbid_n_glycosylation_NXS_or_NXT": True,
            },
            metadata={
                "antigens": list(antigens or []),
                "n_init": int(args.n_init),
                "parallel_budget": int(args.parallel_budget),
                "method": args.method,
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
                schema={
                    "type": "object",
                    "required": ["selected"],
                    "properties": {
                        "selected": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["id", "sequence", "score"],
                                "properties": {
                                    "id": {"type": "integer"},
                                    "sequence": {"type": "string"},
                                    "score": {"type": "number"},
                                },
                            },
                        }
                    },
                },
            ),
            ResponseSpaceSpec(
                name="direct_sequence_generation",
                output_kind="json",
                parser="tasks.antibody.core.ldm_light.direct.parse_direct_sequences",
                description="Direct variants emit a JSON list of CDRH3 sequences.",
                schema={
                    "type": "array",
                    "items": {"type": "string", "minLength": seq_len, "maxLength": seq_len},
                },
            ),
            ResponseSpaceSpec(
                name="dsl_update",
                output_kind="json",
                parser="tasks.antibody.core.ldm.llm.response_parser.parse_response",
                description="Post-warmup LLM updates search-space and optional bias DSL atoms.",
                schema={
                    "type": "object",
                    "properties": {
                        "rationale": {"type": "string"},
                        "update_trust_region": {"type": "string"},
                        "update_bias": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
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
                "beta": float(args.acq_beta),
                "xi": float(args.acq_xi),
                "n_init": int(args.n_init),
                "parallel_budget": int(args.parallel_budget),
                "batch_size": int(args.batch_size),
                "softmax_eta": float(args.softmax_eta),
                "gen_m": int(args.gen_m),
                "n_strategies": int(args.n_strategies),
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
                "tasks.antibody.core.engine_adapters.AntibodySurrogateEncoder"
                if method_spec["uses_acquisition"]
                else ""
            ),
            version="antbo_categorical_sequence_v1" if method_spec["uses_acquisition"] else "",
        ),
        proposal_search=ProposalSearchSpec(
            name="single_turn",
            evaluation_policy="outer_loop_acquisition_selection",
            parameters={
                "method": args.method,
                "proposal_mode": method_spec["base_measure"],
                "reduction": method_spec["reduction"],
                "planner_mode": args.planner_mode,
            },
        ),
        metadata={
            "seed": int(args.seed),
            "n_trials": int(args.n_trials),
            "include_antigen_context": bool(args.include_antigen_context),
        },
    )


def configure_llm_environment(args: argparse.Namespace) -> None:
    if args.mock:
        return
    if args.llm_url:
        os.environ["LLM_BASE_URL"] = args.llm_url.rstrip("/")
    if args.llm_model_name:
        os.environ["LLM_MODEL"] = args.llm_model_name
    if args.api_key:
        os.environ["LLM_API_KEY"] = args.api_key
    elif args.llm_url and not os.environ.get("LLM_API_KEY"):
        os.environ["LLM_API_KEY"] = "EMPTY"
    if args.llm_max_tokens is not None:
        os.environ["LLM_MAX_TOKENS"] = str(args.llm_max_tokens)
    if args.disable_thinking is True:
        os.environ["LLM_DISABLE_THINKING"] = "1"
    elif args.disable_thinking is False:
        os.environ["LLM_DISABLE_THINKING"] = "0"


def make_runner_args(args: argparse.Namespace, antigens_file: Path, out_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        config=str(resolve_path(args.config)),
        antigens_file=str(antigens_file),
        seed=args.seed,
        n_trials=args.n_trials,
        n_evals=args.n_evals,
        batch_size=args.batch_size,
        out_root=str(out_dir),
        temperature=args.temperature,
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
        history_top_k=args.history_top_k,
        parallel_budget=args.parallel_budget,
        method=args.method,
        gen_m=args.gen_m,
        n_strategies=args.n_strategies,
        planner_mode=args.planner_mode,
        softmax_eta=args.softmax_eta,
        per_strategy_budget=args.per_strategy_budget,
        pool_score=args.pool_score,
        selection_score=args.selection_score,
        bias_weight=args.bias_weight,
        sample_timeout_s=args.sample_timeout_s,
        device=getattr(args, "_runtime_device", args.device or "cpu"),
        n_init=args.n_init,
        acq=args.acq,
        acq_beta=args.acq_beta,
        acq_xi=args.acq_xi,
        include_antigen_context=args.include_antigen_context,
        fallback_random=args.fallback_random,
        disable_thinking=args.disable_thinking,
    )


def run(args: argparse.Namespace) -> list[str]:
    config = load_config(args.config)
    if args.device or args.absolut_path:
        config = dict(config)
        if args.device:
            config["device"] = args.device
        if args.absolut_path:
            bbox = dict(config.get("bbox") or {})
            bbox["path"] = args.absolut_path
            config["bbox"] = bbox
    antigens = resolve_antigens(args)
    out_dir = resolve_out_dir(args)
    configure_llm_environment(args)

    if args.dry_run:
        print(json.dumps(planned_config_json(args, config, antigens), indent=2, sort_keys=True))
        return []

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import tasks.antibody.core.ldm_light.ldm_acq as antbo_tts

    if args.mock:
        config = dict(config)
        config["bbox"] = {
            "antigen": "PLACEHOLDER",
            "tool": "random",
            "path": "/tmp",
            "process": 1,
            "startTask": 0,
        }
        config["tabular_search_csv"] = None
        config["device"] = "cpu"
        config["seq_len"] = int(config.get("seq_len", 11))
        antbo_tts.make_llm_client = lambda: DeterministicAntBOTTSLLM()

    args._runtime_device = str(config.get("device") or "cpu")

    out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs: list[str] = []
    with tempfile.TemporaryDirectory(prefix="antbo_tts_") as td:
        antigens_path = Path(td) / "antigens.txt"
        antigens_path.write_text("\n".join(antigens) + "\n", encoding="utf-8")
        runner_args = make_runner_args(args, antigens_path, out_dir)
        for antigen in antigens:
            for seed in range(args.seed, args.seed + args.n_trials):
                run_dir = antbo_tts.run_one(config, antigen, seed, runner_args)
                run_dirs.append(str(Path(run_dir).resolve()))

    print(json.dumps({"run_dirs": run_dirs}, indent=2, sort_keys=True))
    return run_dirs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.acq_beta < 0:
        raise ValueError("--acq-beta must be non-negative")
    if args.acq_xi < 0:
        raise ValueError("--acq-xi must be non-negative")
    if args.llm_max_tokens is not None and args.llm_max_tokens <= 0:
        raise ValueError("--llm-max-tokens must be positive")
    if args.gen_m <= 0:
        raise ValueError("--gen-m must be positive")
    if args.n_strategies <= 0:
        raise ValueError("--n-strategies must be positive")
    if args.parallel_budget <= 0:
        raise ValueError("--parallel-budget must be positive")
    if args.per_strategy_budget < 0:
        raise ValueError("--per-strategy-budget must be non-negative")
    if args.softmax_eta < 0 or args.softmax_eta != args.softmax_eta:
        raise ValueError("--softmax-eta must be non-negative or positive infinity")
    if args.sample_timeout_s <= 0:
        raise ValueError("--sample-timeout-s must be positive")
    if METHOD_SPECS[args.method]["uses_acquisition"] and args.n_init <= 0:
        raise ValueError("Acquisition-guided antibody methods require --n-init to be positive")
    if args.method.startswith("direct_") and args.batch_size > args.gen_m:
        raise ValueError("Direct methods require --batch-size <= --gen-m")
    if args.method.startswith("policy_") and args.batch_size > args.n_strategies:
        raise ValueError("Policy methods require --batch-size <= --n-strategies")
    if (
        args.method.startswith("policy_")
        and args.per_strategy_budget == 0
        and args.parallel_budget < args.n_strategies
    ):
        raise ValueError(
            "Policy methods require --parallel-budget >= --n-strategies "
            "when --per-strategy-budget is zero"
        )
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
