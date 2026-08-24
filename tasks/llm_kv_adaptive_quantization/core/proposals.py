"""Reservoir expansion adapters for quantizer source proposals."""

from __future__ import annotations

from collections.abc import Callable
from itertools import product
import json
import re
from typing import Any

from ldm_tts.contracts import RawProposal
from ldm_tts.engine.expansion import ExpansionRequest, ExpansionResult
from ldm_tts.transport import ProposalClient, ProposalRequest

from tasks.llm_kv_adaptive_quantization.core.candidate import canonical_source_key


BIT_CAPS = (2, 3, 4)
GROUP_SIZES = (16, 32, 64, 128)
RESIDUAL_LENGTHS = (16, 32, 64, 128, 256)
SPEC_KEYS = ("bit_cap", "key_group_size", "value_group_size", "residual_length")
SPEC_SPACE = tuple(
    dict(zip(SPEC_KEYS, values, strict=True))
    for values in product(BIT_CAPS, GROUP_SIZES, GROUP_SIZES, RESIDUAL_LENGTHS)
)


def quantizer_spec_schema(candidate_count: int) -> dict[str, Any]:
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    return {
        "type": "object",
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": candidate_count,
                "maxItems": candidate_count,
                "items": {
                    "type": "object",
                    "required": list(SPEC_KEYS),
                    "properties": {
                        "bit_cap": {"type": "integer", "enum": list(BIT_CAPS)},
                        "key_group_size": {
                            "type": "integer",
                            "enum": list(GROUP_SIZES),
                        },
                        "value_group_size": {
                            "type": "integer",
                            "enum": list(GROUP_SIZES),
                        },
                        "residual_length": {
                            "type": "integer",
                            "enum": list(RESIDUAL_LENGTHS),
                        },
                    },
                    "additionalProperties": False,
                },
            }
        },
        "additionalProperties": False,
    }


def proposal_response_format(candidate_count: int) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "quantizer_candidate_specs",
            "strict": True,
            "schema": quantizer_spec_schema(candidate_count),
        },
    }


# Aliases some LLMs use for the canonical quantizer spec fields.
_SPEC_ALIASES = {
    "bits": "bit_cap",
    "bit_cap": "bit_cap",
    "group_size": None,  # ambiguous: fills whichever of key/value group size is missing
    "key_group_size": "key_group_size",
    "value_group_size": "value_group_size",
    "residual_length": "residual_length",
    "residual": "residual_length",
}
_SPEC_CHOICES = {
    "bit_cap": BIT_CAPS,
    "key_group_size": GROUP_SIZES,
    "value_group_size": GROUP_SIZES,
    "residual_length": RESIDUAL_LENGTHS,
}


def _snap_to_choice(name: str, value: Any) -> int:
    """Snap a numeric value to the nearest allowed enum entry for a spec field."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"candidate has non-integer {name}: {value!r}")
    choices = _SPEC_CHOICES[name]
    return min(choices, key=lambda choice: abs(choice - value))


def _normalize_quantizer_spec(item: Any, *, index: int) -> dict[str, int]:
    """Normalize an LLM-produced candidate object into a canonical spec.

    Tolerates the aliases and off-enum values that open-ended chat completions
    commonly return when the serving endpoint ignores ``response_format``.
    """
    if not isinstance(item, dict):
        raise ValueError(f"candidate {index + 1} is not an object")
    mapped: dict[str, int] = {}
    for key, value in item.items():
        target = _SPEC_ALIASES.get(str(key))
        if target is None:
            continue
        snapped = _snap_to_choice(target, value)
        if target in ("key_group_size", "value_group_size"):
            mapped.setdefault(target, snapped)
        else:
            mapped[target] = snapped
    if "group_size" in item:
        shared = _snap_to_choice("key_group_size", item["group_size"])
        mapped.setdefault("key_group_size", shared)
        mapped.setdefault("value_group_size", shared)
    missing = [name for name in SPEC_KEYS if name not in mapped]
    if missing:
        raise ValueError(
            f"candidate {index + 1} is missing field(s): {', '.join(missing)}"
        )
    return {name: mapped[name] for name in SPEC_KEYS}


def parse_quantizer_specs(text: str, *, expected_count: int) -> list[dict[str, int]]:
    raw = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("proposal response is not valid JSON") from exc
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
        candidates = payload["candidates"]
    else:
        candidates = None
    if not isinstance(candidates, list) or len(candidates) != expected_count:
        raise ValueError(f"proposal response must contain exactly {expected_count} candidates")
    return [
        _normalize_quantizer_spec(item, index=index + 1)
        for index, item in enumerate(candidates)
    ]


def materialize_quantizer_source(seed_source: str, spec: dict[str, int]) -> str:
    residual = int(spec["residual_length"])
    replacements = (
        ("self.bits = 4", f"self.bits = {spec['bit_cap']}"),
        (
            'self.bits = min(4, int(budget_state.get("budget_bits", 4)))',
            f'self.bits = min({spec["bit_cap"]}, int(budget_state.get("budget_bits", 4)))',
        ),
        ("self.key_group_size = 32", f"self.key_group_size = {spec['key_group_size']}"),
        (
            "self.value_group_size = 32",
            f"self.value_group_size = {spec['value_group_size']}",
        ),
        ("self.key_residual_length = 128", f"self.key_residual_length = {residual}"),
        (
            "self.value_residual_length = 128",
            f"self.value_residual_length = {residual}",
        ),
        (
            'residual = 128 if workload.startswith("longbench_") else 32',
            f'residual = {residual} if workload.startswith("longbench_") else {max(16, residual // 4)}',
        ),
    )
    source = seed_source
    for old, new in replacements:
        if old not in source:
            raise ValueError(f"seed source is missing materialization anchor: {old}")
        source = source.replace(old, new, 1)
    return source


def _materialize_distinct_specs(
    seed_source: str,
    requested_specs: list[dict[str, int]],
    *,
    evaluated_keys: set[str],
) -> list[tuple[str, dict[str, int], bool]]:
    occupied = set(evaluated_keys)
    materialized: list[tuple[str, dict[str, int], bool]] = []
    space_keys = [tuple(spec[name] for name in SPEC_KEYS) for spec in SPEC_SPACE]
    for requested in requested_specs:
        requested_key = tuple(requested[name] for name in SPEC_KEYS)
        start = space_keys.index(requested_key)
        for offset in range(len(SPEC_SPACE)):
            spec = dict(SPEC_SPACE[(start + offset) % len(SPEC_SPACE)])
            source = materialize_quantizer_source(seed_source, spec)
            key = canonical_source_key(source)
            if key in occupied:
                continue
            occupied.add(key)
            materialized.append((source, spec, spec != requested))
            break
        else:
            raise ValueError("quantizer specification space is exhausted")
    return materialized


class DeterministicQuantizerExpander:
    def __init__(self, seed_source: str, *, collectable: bool = True) -> None:
        self.seed_source = seed_source
        self.collectable = bool(collectable)

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        proposals = []
        for index in range(request.reservoir_size):
            bits = 4 - ((request.round_idx + index) % 3)
            residual = (128, 64, 32, 16)[index % 4]
            source = self.seed_source.replace(
                "self.bits = min(4, int(budget_state.get(\"budget_bits\", 4)))",
                f"self.bits = min({bits}, int(budget_state.get(\"budget_bits\", 4)))",
                1,
            ).replace(
                'residual = 128 if workload.startswith("longbench_") else 32',
                f'residual = {residual} if workload.startswith("longbench_") else 32',
                1,
            )
            proposals.append(
                RawProposal(
                    {"code": source},
                    "mock_model_quantizer" if self.collectable else "official_seed",
                    {"collectable": self.collectable, "variant": index},
                )
            )
        return ExpansionResult(
            proposals=tuple(proposals),
            metadata={"mode": "deterministic", "round": request.round_idx},
        )


class EndpointQuantizerExpander:
    def __init__(
        self,
        client: ProposalClient,
        seed_source: str,
        *,
        before_request: Callable[[], None] | None = None,
    ) -> None:
        self.client = client
        self.seed_source = seed_source
        self.before_request = before_request

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        history = [
            {
                "candidate_id": item.candidate_id,
                "spec": item.candidate.metadata.get("proposal_spec"),
                "metrics": dict(item.metrics),
            }
            for item in request.observations[-5:]
        ]
        prompt = (
            f"Propose exactly {request.reservoir_size} distinct adaptive KV quantizer parameter "
            "sets as JSON. Each candidate must be an object with exactly these four integer "
            "fields and only these allowed values: bit_cap in [4, 3, 2], key_group_size in "
            "[16, 32, 64, 128], value_group_size in [16, 32, 64, 128], residual_length in "
            "[128, 64, 32, 16]. "
            'Return a JSON object shaped like {"candidates": [{"bit_cap": 4, '
            '"key_group_size": 32, "value_group_size": 64, "residual_length": 128}]}. '
            "Balance quality against compression, vary bit caps, group sizes, and residual "
            "lengths within this reservoir, and avoid repeating observed specifications. Do "
            f"not return Python code, markdown, or prose. Observed history: {history}"
        )
        if self.before_request is not None:
            self.before_request()
        response = self.client.propose(
            ProposalRequest(
                messages=(
                    {
                        "role": "system",
                        "content": "You design safe tensor-level KV-cache quantizers.",
                    },
                    {"role": "user", "content": prompt},
                ),
                metadata={"round_idx": request.round_idx},
            )
        )
        requested_specs = parse_quantizer_specs(
            response.text, expected_count=request.reservoir_size
        )
        evaluated_keys = {
            item.candidate.canonical_key for item in request.observations
        }
        proposals = tuple(
            RawProposal(
                {"code": source},
                "openai_quantizer_spec",
                {
                    "round_idx": request.round_idx,
                    "collectable": True,
                    "requested_spec": requested,
                    "proposal_spec": materialized_spec,
                    "proposal_repaired": repaired,
                },
            )
            for requested, (source, materialized_spec, repaired) in zip(
                requested_specs,
                _materialize_distinct_specs(
                    self.seed_source,
                    requested_specs,
                    evaluated_keys=evaluated_keys,
                ),
                strict=True,
            )
        )
        return ExpansionResult(
            proposals=proposals,
            attempts=(response,),
            metadata={"mode": "openai", "round": request.round_idx},
        )
