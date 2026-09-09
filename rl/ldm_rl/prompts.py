"""Policy-facing text rendering for the LDM environment.

The environment is text-in / text-out for the policy: the reset observation
explains the task, the objectives, the candidate domain and the exact response
schema; each step observation reports admission rejections, evaluation results,
the incumbent and the remaining budget. Structured data stays in
``EnvStep.info`` for tooling; the rendered text is what the policy sees and is
appended to the trajectory with ``loss_mask = 0``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

from ldm_tts.contracts import (
    CandidateDomainSpec,
    LDMTaskSpec,
    ObjectiveSpec,
    Observation,
    ResponseSpaceSpec,
)


def _render_objectives(specs: Sequence[ObjectiveSpec]) -> str:
    lines = []
    for spec in specs:
        direction = "MAXIMIZE" if spec.direction == "maximize" else "MINIMIZE"
        lines.append(f"- {spec.name!r} ({direction}): {spec.description or 'no description'}")
    return "\n".join(lines)


def _render_domain(domain: CandidateDomainSpec) -> str:
    lines = [f"- Domain: {domain.name} ({domain.kind})"]
    if domain.representation:
        lines.append(f"- Representation: {domain.representation}")
    if domain.constraints:
        lines.append(f"- Constraints: {json.dumps(domain.constraints, sort_keys=True, default=str)}")
    return "\n".join(lines)


def _schema_skeleton(schema: Any, *, n_items: int = 2, _depth: int = 0) -> Any:
    """Build a structurally valid *instance* of ``schema`` (not the schema itself).

    The prompt used to show only the JSON Schema, which every model we tried
    echoed back in schema *shape* rather than filling in::

        {"properties": {"direct_smiles": {"items": [...]}, "type": "object"}
        {"direct_smiles": {"items": [...]}, "required": null}

    Both parse-fail against ``bridge.py``'s ``require_list(data, "direct_smiles")``,
    so every round returned zero candidates, every episode returned reward 0.0,
    and GRPO then divided by a zero-std reward group -> NaN in the backward pass.
    A schema tells the model what is *legal*; it does not show what to *emit*.
    """

    if not isinstance(schema, dict) or _depth > 6:
        return "..."
    kind = schema.get("type")
    if kind == "object":
        props = schema.get("properties") or {}
        return {
            name: _schema_skeleton(sub, n_items=n_items, _depth=_depth + 1)
            for name, sub in props.items()
        }
    if kind == "array":
        item = schema.get("items") or {}
        return [
            _schema_skeleton(item, n_items=n_items, _depth=_depth + 1)
            for _ in range(max(1, n_items))
        ]
    if kind in ("number", "integer"):
        return 0
    if kind == "boolean":
        return False
    return "..."


def _render_response_space(space: ResponseSpaceSpec, *, n_items: int = 2) -> str:
    lines = [f"- Action format: {space.output_kind}"]
    if space.description:
        lines.append(f"- {space.description}")
    if space.schema:
        lines.append(
            "- Response schema:\n" + json.dumps(space.schema, indent=2, sort_keys=True)
        )
        # Showing the schema alone is not enough -- see _schema_skeleton.
        # Set LDM_RL_NO_SCHEMA_EXAMPLE=1 to reproduce the schema-only prompt.
        if not os.environ.get("LDM_RL_NO_SCHEMA_EXAMPLE"):
            try:
                skeleton = _schema_skeleton(space.schema, n_items=n_items)
            except Exception:  # noqa: BLE001 - never break prompt rendering
                skeleton = None
            if skeleton is not None:
                lines.append(
                    "- Emit an INSTANCE of that schema, not the schema. Exact shape:\n"
                    + json.dumps(skeleton, sort_keys=True)
                    + '\n  (replace every "..." with your own value; keep the keys and nesting)'
                )
    return "\n".join(lines)


def render_reset_observation(
    spec: LDMTaskSpec,
    *,
    reservoir_size: int,
    context: dict[str, Any] | None = None,
    extra_instructions: str = "",
) -> str:
    """Render the initial observation (episode prompt) for one campaign."""

    producing = [
        expansion
        for expansion in spec.reservoir.expansions
        if expansion.produces_candidates
    ]
    space_names = sorted({expansion.response_space for expansion in producing})
    spaces = [space for space in spec.response_spaces if space.name in space_names]
    if not spaces:
        spaces = list(spec.response_spaces[:1])
    if not spaces:
        raise ValueError(f"task {spec.task!r} declares no usable response space")

    parts = [
        f"You are solving LDM task {spec.task!r}.",
        "",
        "Objectives:",
        _render_objectives(spec.objectives),
        "",
        "Candidate domain:",
        _render_domain(spec.candidate_domain),
        "",
        "Each turn, propose exactly "
        f"{reservoir_size} distinct candidate payload(s) as raw JSON.",
        "",
        _render_response_space(spaces[0]),
    ]
    if context:
        parts += [
            "",
            "Episode context:",
            json.dumps(context, indent=2, sort_keys=True, default=str),
        ]
    if extra_instructions:
        parts += ["", extra_instructions]
    parts += [
        "",
        "Return only the JSON. Do not add prose, markdown fences, or code.",
    ]
    return "\n".join(parts)


def _brief_payload(payload: Any, *, limit: int = 400) -> str:
    try:
        text = json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError):
        text = str(payload)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render_step_observation(
    *,
    round_idx: int,
    remaining_rounds: int,
    parse_error: str | None,
    proposals_count: int,
    rejections: Sequence[Any],
    evaluations: Sequence[Any],
    incumbent: Observation | None,
    succeeded_total: int | None = None,
) -> str:
    """Render the post-step feedback appended to the policy transcript.

    ``succeeded_total`` is the number of successful evaluations so far in this
    episode. It exists because ``incumbent`` being ``None`` means two very
    different things, and the old text conflated them (see below).
    """

    lines = [f"<round {round_idx}>"]
    if parse_error:
        lines += [
            "Your response could not be parsed as a proposal:",
            f"  {parse_error}",
        ]
        lines += [f"({proposals_count} raw candidate(s) admitted before the error)"]
    else:
        lines += [f"Received {proposals_count} candidate proposal(s)."]
    for rejection in rejections:
        message = getattr(rejection, "message", "") or str(rejection)
        reason = getattr(rejection, "reason", "rejected")
        lines.append(f"- Rejected ({reason}): {message}")
    for item in evaluations:
        evaluation = item.evaluation
        status = evaluation.status
        if status == "succeeded":
            metrics = json.dumps(evaluation.metrics, sort_keys=True)
            lines.append(
                f"- {item.candidate.candidate_id}: SUCCEEDED metrics={metrics}"
            )
        else:
            error = evaluation.error or "unknown error"
            lines.append(f"- {item.candidate.candidate_id}: {status.upper()} ({error})")
    if incumbent is not None:
        lines += [
            "Best so far:",
            f"  candidate: {_brief_payload(incumbent.candidate.payload)}",
            f"  metrics: {json.dumps(dict(incumbent.metrics), sort_keys=True)}",
        ]
    elif succeeded_total:
        # env.py:440 computes an incumbent only for single-objective tasks --
        # with two objectives there is no single best point, just a Pareto set.
        # That is correct, but rendering it as "no successful evaluation yet"
        # states something else entirely, and it is false: on 2026-09-02 a run
        # showed "mol-3ee8e371814d: SUCCEEDED" and "no successful evaluation
        # yet" in the same observation, after which the policy stopped emitting
        # anything at all for the rest of the episode.
        lines.append(
            f"Best so far: {succeeded_total} successful evaluation(s); "
            "no single best with multiple objectives (a Pareto set, not a point)."
        )
    else:
        lines.append("Best so far: none (no successful evaluation yet).")
    if remaining_rounds > 0:
        lines.append(f"Rounds remaining in this episode: {remaining_rounds}")
    lines.append("Propose the next candidates.")
    return "\n".join(lines)
