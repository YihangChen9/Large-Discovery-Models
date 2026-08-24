#!/usr/bin/env python3

"""nanoGPT task workflow for the shared LDM-TTS config runner."""

from __future__ import annotations

import argparse
import asyncio
import ast
import difflib
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from ldm_tts.contracts.evaluation import as_float as shared_as_float
from ldm_tts.contracts.evaluation import is_finite_number
from ldm_tts.campaign import (
    CampaignBudget,
    CampaignRecipe,
    CampaignRequest,
    async_run_campaign,
)
from ldm_tts.engine import LDMEngineState
from ldm_tts.optimization.acquisition import make_acquisition
from ldm_tts.data import DataCollectionSink, make_parameter_edit_ir
from ldm_tts.optimization.search import (
    SEARCH_METHOD_ALIASES as SHARED_SEARCH_METHOD_ALIASES,
    canonical_search_method,
    estimate_generated,
    get_traversal_method,
)
from tasks.nanogpt.core.expansion_schema import (
    OperationParameter,
    OperationSchema,
    ValidatedOperation,
    canonical_name,
    choice_values_equal,
    initial_active_operation_schema,
    load_operation_schema,
    normalize_operation_numeric,
    normalize_operation_parameter,
    operation_feature_dim,
    operation_feature_version,
    operation_parameter_from_payload,
    operation_parameter_to_json,
    operation_schema_to_json,
    replace_operation_schema,
    validate_operation_payload,
    validate_operation_value,
)
from ldm_tts.contracts import (
    AcquisitionSpec,
    CandidateDomainSpec,
    LDMTaskSpec,
    ObjectiveSpec,
    ProposalSearchSpec,
    ReservoirExpansionSpec,
    ReservoirSpec,
    ResponseSpaceSpec,
    SurrogateSpaceSpec,
)

from tasks.nanogpt.core.single_search import make_unique_run_dir, safe_path_tag
from tasks.nanogpt.core.search_core import (
    DEFAULT_TASK_CONTEXT,
    ProgressBar,
    SearchConfig,
    SearchEngine,
    SearchState,
    add_token_usage,
    extract_generation_logprobs,
    normalize_token_usage,
    summarize_generation_logprobs,
)


GENERATORS = {
    "api",
    "closed_loop",
    "harness",
    "mock",
    "operation_mock",
    "operation_tool",
    "operation_tool_plain_text",
    "tool_call",
}
OPERATION_GENERATORS = {"operation_mock", "operation_tool", "operation_tool_plain_text"}
SURROGATE_SEARCH_METHODS = {"single_turn", "best_of_n", "beam_search", "tree_search"}
SEARCH_METHOD_ALIASES = {
    "auto": "auto",
    **{
        alias: canonical
        for alias, canonical in SHARED_SEARCH_METHOD_ALIASES.items()
        if canonical in SURROGATE_SEARCH_METHODS
    },
}

FEATURE_VERSION = "code_numeric_hash_v2"
DEFAULT_GP_REJECT_STATUSES = {
    "crash",
    "evaluation_error",
    "generation_error",
    "score_missing",
    "timeout",
}
COMMON_NUMERIC_NAMES = [
    "ADAM_BETA1",
    "ADAM_BETA2",
    "ASPECT_RATIO",
    "BATCH_SIZE",
    "BLOCK_SIZE",
    "DEPTH",
    "DEVICE_BATCH_SIZE",
    "EMBEDDING_LR",
    "FINAL_LR_FRAC",
    "GRAD_ACCUM",
    "HEAD_DIM",
    "LEARNING_RATE",
    "MATRIX_LR",
    "MAX_SEQ_LEN",
    "MLP_TAU",
    "MODEL_DIM",
    "MUON_MOMENTUM",
    "N_EMBD",
    "N_HEAD",
    "N_LAYER",
    "N_KV_HEAD",
    "NGRAM_MULT",
    "NGRAM_SCALE",
    "NS_STEPS",
    "ROTARY_BASE",
    "SCALAR_LR",
    "TIME_BUDGET",
    "TOTAL_BATCH_SIZE",
    "UNEMBEDDING_LR",
    "VALUE_GATE_CHANNELS",
    "WARMDOWN_RATIO",
    "WARMUP_RATIO",
    "WEIGHT_DECAY",
    "WIDTH",
    "WINDOW_SIZE",
    "X0_LAMBDA_INIT",
]


def require_numpy() -> Any:
    global np
    try:
        return np
    except NameError:
        import numpy as np_module

        np = np_module
        return np_module


@dataclass
class Features:
    vector: list[float]
    params: dict[str, Any]
    source_hash: str


@dataclass
class BufferEntry:
    score: float
    feature_vector: list[float]
    feature_version: str
    source_hash: str
    params: dict[str, float]
    metrics: dict[str, Any]
    state_id: str
    iteration: int
    train_path: str
    run_name: str


@dataclass
class Prediction:
    mean: float
    std: float
    ei: float
    lcb: float
    selection_score: float


@dataclass
class GeneratorAction:
    kind: str
    operations: list[ValidatedOperation] | None = None
    feature: OperationParameter | None = None
    rationale: str = ""
    source: str = ""


@dataclass
class OperationApplyResult:
    text: str
    patch: str
    records: list[dict[str, Any]]


@dataclass
class RunLogger:
    path: Path

    def write(self, message: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.path.open("a", encoding="utf-8") as file:
            file.write(f"[{stamp}] {message}\n")


class ModelBasedProgress:
    def __init__(
        self,
        *,
        enabled: bool,
        total: int,
        width: int,
        score_key: str,
        minimize: bool,
        feature_status: Callable[[], str] | None = None,
    ):
        self.enabled = enabled and total > 0
        self.score_key = score_key
        self.minimize = minimize
        self.feature_status = feature_status
        self.count = 0
        self.best_score: float | None = None
        self.bar = ProgressBar(total=total, label="model_based", width=width) if self.enabled else None
        if self.bar is not None:
            self.bar.update(0, status=self._with_feature_status("starting"))

    def _with_feature_status(self, message: str) -> str:
        if self.feature_status is None:
            return message
        feature_text = self.feature_status().strip()
        if not feature_text:
            return message
        return f"{message} {feature_text}" if message else feature_text

    def status(self, message: str) -> None:
        if self.bar is not None:
            self.bar.update(self.count, best_score=self.best_score, status=self._with_feature_status(message))

    def generated(self, state: SearchState) -> None:
        score = as_float(state.metrics.get("surrogate_score"))
        suffix = "" if score is None else f" sg={score:.4g}"
        self.step(f"scored {state.state_id}{suffix}")

    def evaluated(self, state: SearchState) -> None:
        if state.score is not None and finite_score(state.score):
            score = float(state.score)
            if self.best_score is None or is_better(score, self.best_score, minimize=self.minimize):
                self.best_score = score
            self.step(f"evaluated {state.state_id} {self.score_key}={score:.6g}")
        else:
            self.step(f"evaluated {state.state_id} {state.status}")

    def step(self, message: str) -> None:
        if self.bar is None:
            return
        self.count += 1
        self.bar.update(self.count, best_score=self.best_score, status=self._with_feature_status(message))

    def finish(self, message: str = "done") -> None:
        if self.bar is not None:
            self.bar.finish(self.count, best_score=self.best_score, status=self._with_feature_status(message))


class FeedbackMemory:
    columns = [
        "row",
        "kind",
        "iteration",
        "state_id",
        "parent_id",
        "root_state_id",
        "status",
        "score_key",
        "score",
        "score_valid",
        "previous_best_score",
        "best_score_after",
        "improved_previous_best",
        "surrogate_score",
        "surrogate_pred",
        "surrogate_std",
        "surrogate_ei",
        "action",
        "path",
        "error",
    ]

    def __init__(self, path: Path, *, score_key: str, minimize: bool, max_rows: int):
        self.path = path
        self.score_key = score_key
        self.minimize = minimize
        self.max_rows = max(0, int(max_rows))
        self.rows: list[dict[str, Any]] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("\t".join(self.columns) + "\n", encoding="utf-8")

    def record(
        self,
        *,
        kind: str,
        iteration: int,
        state: SearchState,
        root: SearchState,
        selected_surrogate_metrics: dict[str, Any],
        previous_best_score: float | None,
        best_score_after: float | None,
    ) -> None:
        score = as_float(state.metrics.get(self.score_key, state.score))
        score_valid = score is not None and finite_score(score)
        improved = (
            None
            if score is None or previous_best_score is None
            else is_better(score, previous_best_score, minimize=self.minimize)
        )
        row = {
            "row": len(self.rows) + 1,
            "kind": kind,
            "iteration": iteration,
            "state_id": state.state_id,
            "parent_id": state.parent_id or "",
            "root_state_id": root.state_id,
            "status": state.status,
            "score_key": self.score_key,
            "score": "" if score is None else f"{score:.10g}",
            "score_valid": int(bool(score_valid)),
            "previous_best_score": "" if previous_best_score is None else f"{float(previous_best_score):.10g}",
            "best_score_after": "" if best_score_after is None else f"{float(best_score_after):.10g}",
            "improved_previous_best": "" if improved is None else int(bool(improved)),
            "surrogate_score": format_tsv_float(selected_surrogate_metrics.get("surrogate_score")),
            "surrogate_pred": format_tsv_float(selected_surrogate_metrics.get("surrogate_pred")),
            "surrogate_std": format_tsv_float(selected_surrogate_metrics.get("surrogate_std")),
            "surrogate_ei": format_tsv_float(selected_surrogate_metrics.get("surrogate_ei")),
            "action": feedback_action_summary(state),
            "path": feedback_path_summary(state),
            "error": clean_tsv_text(state.error or ""),
        }
        self.rows.append(row)
        with self.path.open("a", encoding="utf-8") as file:
            file.write("\t".join(clean_tsv_text(row.get(column, "")) for column in self.columns) + "\n")

    def prompt_context(self) -> str:
        if self.max_rows <= 0 or not self.rows:
            return ""
        rows = self.selected_rows()
        best = self.best_row()
        lines = []
        if best is not None:
            lines.append(
                "Current best real evaluation: "
                f"{best['state_id']} {self.score_key}={best['score']} "
                f"status={best['status']} action={best['action']}"
            )
            if best.get("path"):
                lines.append(f"Best path: {best['path']}")
        lines.append("Recent evaluated trials:")
        for row in rows:
            score_text = row["score"] if row["score"] != "" else "missing"
            validity = "valid" if str(row["score_valid"]) == "1" else "invalid"
            improvement = ""
            if row.get("improved_previous_best") != "":
                improvement = " improved" if str(row["improved_previous_best"]) == "1" else " not_improved"
            surrogate = ""
            if row.get("surrogate_pred"):
                surrogate = (
                    f" surrogate_pred={row['surrogate_pred']}"
                    f" std={row['surrogate_std']}"
                    f" score={row['surrogate_score']}"
                )
            lines.append(
                f"- iter={row['iteration']} {row['kind']} {row['state_id']} "
                f"status={row['status']} {self.score_key}={score_text} ({validity}{improvement});"
                f"{surrogate} action={row['action']}"
            )
            if row.get("error"):
                lines.append(f"  error={row['error']}")
        return "\n".join(lines)

    def selected_rows(self) -> list[dict[str, Any]]:
        if len(self.rows) <= self.max_rows:
            return list(self.rows)
        best = self.best_row()
        tail = self.rows[-self.max_rows :]
        if best is not None and best not in tail and self.max_rows > 1:
            return [best] + tail[-(self.max_rows - 1) :]
        return tail

    def best_row(self) -> dict[str, Any] | None:
        valid_rows = []
        for row in self.rows:
            score = as_float(row.get("score"))
            if score is not None:
                valid_rows.append((score, row))
        if not valid_rows:
            return None
        return sorted(valid_rows, key=lambda item: item[0], reverse=not self.minimize)[0][1]


def operation_schema_ir_parameters(
    schema: OperationSchema,
    current_values: dict[str, Any],
) -> list[dict[str, Any]]:
    """Map one operation schema into the canonical ldm-2.0 parameter shape."""

    parameters: list[dict[str, Any]] = []
    for parameter in schema.parameters.values():
        domain = (
            list(parameter.choices)
            if parameter.kind == "choice"
            else [parameter.min_value, parameter.max_value]
        )
        parameters.append(
            {
                "name": parameter.name,
                "type": parameter.kind,
                "domain": domain,
                "scale": parameter.scale if parameter.kind != "choice" else None,
                "edit_op": (
                    "set_choice" if parameter.kind == "choice" else "set_numeric"
                ),
                "current_value": current_values.get(parameter.name),
            }
        )
    return parameters


def nanogpt_ir_observations(engine: SearchEngine) -> list[dict[str, Any]]:
    """Return the evaluated history that was visible to the proposal model."""

    observations: list[dict[str, Any]] = []
    for index, candidate in enumerate(engine.ranked_states()[:8]):
        if candidate.score is None or not finite_score(candidate.score):
            continue
        roles = ["best_so_far", "best_path"] if index == 0 else ["evaluated"]
        observations.append(
            {
                "design": {"state_id": candidate.state_id},
                "results": {engine.config.score_key: float(candidate.score)},
                "round": candidate.depth,
                "roles": roles,
                "description": candidate.description or None,
            }
        )
    return observations


class OperationSearchEngine(SearchEngine):
    def __init__(self, config: SearchConfig, operation_schema: OperationSchema, args: argparse.Namespace):
        super().__init__(config)
        self.args = args
        self.full_operation_schema = operation_schema
        self.operation_schema = initial_active_operation_schema(operation_schema, args)
        self.operation_retries = max(0, int(args.operation_retries))
        self.max_operations_per_step = max(1, int(args.max_operations_per_step))
        self.allow_feature_expansion = bool(getattr(args, "allow_feature_expansion", True))
        self.allow_new_feature_specs = bool(getattr(args, "allow_new_feature_specs", False))
        requested_max_features = int(getattr(args, "max_active_operation_features", 0) or 0)
        if requested_max_features <= 0:
            self.max_active_operation_features = len(self.full_operation_schema.parameters)
        else:
            self.max_active_operation_features = max(1, requested_max_features)
        if len(self.operation_schema.parameters) > self.max_active_operation_features:
            raise ValueError(
                f"Initial expansion parameter count {len(self.operation_schema.parameters)} exceeds "
                f"--max-expansion-parameters={self.max_active_operation_features}."
            )
        self.expansion_history: list[dict[str, Any]] = []
        self.current_iteration: int | None = None
        self._mock_counter = 0
        self.args.operation_schema_object = self.operation_schema
        self._data_sink = DataCollectionSink.from_env(
            default_root=self.config.out_dir / "ldm_data"
        )

    def inactive_operation_schema(self) -> OperationSchema:
        active = set(self.operation_schema.parameters)
        inactive = {
            name: parameter
            for name, parameter in self.full_operation_schema.parameters.items()
            if name not in active
        }
        return replace_operation_schema(
            self.full_operation_schema,
            inactive,
            version_suffix="inactive",
            description_prefix="Inactive reservoir expansion parameters.",
        )

    def feature_expansion_available(self) -> bool:
        if not self.allow_feature_expansion:
            return False
        if len(self.operation_schema.parameters) >= self.max_active_operation_features:
            return False
        return bool(self.inactive_operation_schema().parameters) or self.allow_new_feature_specs

    def add_operation_feature(
        self,
        parameter: OperationParameter,
        *,
        source: str,
        rationale: str = "",
        iteration: int | None = None,
        state_id: str | None = None,
    ) -> bool:
        name = canonical_name(parameter.name)
        if name in self.operation_schema.parameters:
            return False
        if len(self.operation_schema.parameters) >= self.max_active_operation_features:
            return False
        parameter = normalize_operation_parameter(parameter)
        parameters = dict(self.operation_schema.parameters)
        parameters[name] = parameter
        self.operation_schema = replace_operation_schema(
            self.full_operation_schema,
            parameters,
            version_suffix="active",
            description_prefix="Active reservoir expansion schema.",
        )
        if name not in self.full_operation_schema.parameters:
            full_parameters = dict(self.full_operation_schema.parameters)
            full_parameters[name] = parameter
            self.full_operation_schema = replace_operation_schema(
                self.full_operation_schema,
                full_parameters,
                version_suffix="full",
                description_prefix="Full reservoir expansion schema including proposed parameters.",
            )
        record = {
            "name": name,
            "kind": parameter.kind,
            "source": source,
            "rationale": rationale,
            "iteration": iteration,
            "state_id": state_id,
            "active_feature_count": len(self.operation_schema.parameters),
            "active_feature_names": list(self.operation_schema.parameters),
            "feature_version": operation_feature_version(self.operation_schema),
        }
        self.expansion_history.append(record)
        self.args.operation_schema_object = self.operation_schema
        active_schema_path = self.config.out_dir / "active_operation_schema.json"
        active_schema_path.write_text(
            json.dumps(operation_schema_to_json(self.operation_schema), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        full_schema_path = self.config.out_dir / "operation_schema.json"
        full_schema_path.write_text(
            json.dumps(operation_schema_to_json(self.full_operation_schema), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expansion_path = self.config.out_dir / "operation_feature_expansions.json"
        expansion_path.write_text(
            json.dumps(self.expansion_history, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return True

    async def _generate_one(
        self,
        parent: SearchState,
        *,
        child_index: int,
        sibling_count: int,
        search_note: str,
    ) -> SearchState | None:
        state = self._new_state(parent=parent, depth=parent.depth + 1)
        current_text = parent.train_path.read_text(encoding="utf-8")
        num_edits = max(1, self.config.num_edits_per_step)
        prior_operations: list[dict[str, Any]] = []
        try:
            async with self._get_generation_semaphore():
                for edit_index in range(1, num_edits + 1):
                    self._progress_status(f"generating {state.state_id} operation {edit_index}/{num_edits}")
                    prompt = build_operation_prompt(
                        self,
                        parent,
                        current_text,
                        child_index=child_index,
                        sibling_count=sibling_count,
                        search_note=search_note,
                        edit_index=edit_index,
                        total_edits=num_edits,
                        prior_operations=prior_operations,
                    )
                    state.prompt_path = self._edit_artifact_path(state, "prompt", edit_index, num_edits, "md")
                    state.prompt_path.write_text(prompt, encoding="utf-8")

                    active_schema_before = self.operation_schema
                    inactive_schema_before = self.inactive_operation_schema()
                    expansion_history_before = list(self.expansion_history)

                    action, response, token_usage, validation_log = await self._call_operation_generator(
                        prompt,
                        current_text,
                    )
                    completion_tokens, token_usage_detail = normalize_token_usage(token_usage)
                    logprobs_payload = extract_generation_logprobs(token_usage)
                    logprob_summary = summarize_generation_logprobs(logprobs_payload)
                    logprobs_path = None
                    if logprobs_payload is not None:
                        logprobs_path = self._edit_artifact_path(state, "logprobs", edit_index, num_edits, "json")
                        logprobs_path.write_text(
                            json.dumps(logprobs_payload, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                    state.token_usage += completion_tokens
                    state.token_usage_detail = add_token_usage(state.token_usage_detail, token_usage_detail)

                    state.response_path = self._edit_artifact_path(state, "response", edit_index, num_edits, "md")
                    state.response_path.write_text(response, encoding="utf-8")
                    state.llm_response, state.llm_response_truncated = self._inline_llm_response(response)

                    operations_path = self._edit_artifact_path(state, "operations", edit_index, num_edits, "json")
                    if action.kind == "feature":
                        if action.feature is None:
                            raise ValueError("feature action did not include a feature.")
                        added = self.add_operation_feature(
                            action.feature,
                            source=action.source or self.config.generator,
                            rationale=action.rationale,
                            iteration=self.current_iteration,
                            state_id=state.state_id,
                        )
                        if not added:
                            raise ValueError(f"Operation feature {action.feature.name!r} could not be activated.")
                        apply_result = OperationApplyResult(
                            text=current_text,
                            patch="",
                            records=[
                                {
                                    "name": action.feature.name,
                                    "op": "add_feature",
                                    "old_value": None,
                                    "new_value": operation_parameter_to_json(action.feature),
                                    "rationale": action.rationale,
                                    "line": None,
                                }
                            ],
                        )
                        operations_payload = {
                            "summary": f"add operation feature {action.feature.name}",
                            "schema_version": self.operation_schema.version,
                            "active_feature_names": list(self.operation_schema.parameters),
                            "full_schema_version": self.full_operation_schema.version,
                            "max_operations_per_step": self.max_operations_per_step,
                            "operations": [],
                            "feature_action": operation_parameter_to_json(action.feature),
                            "applied": apply_result.records,
                            "validation_log": validation_log,
                        }
                        patch_text = (
                            f"# operation feature activated: {action.feature.name}\n"
                            f"# active_features: {', '.join(self.operation_schema.parameters)}\n"
                        )
                    else:
                        proposal = action.operations or []
                        apply_result = apply_operations_to_train_text(current_text, proposal, self.operation_schema)
                        operations_payload = {
                            "summary": operation_summary(proposal, apply_result.records),
                            "schema_version": self.operation_schema.version,
                            "active_feature_names": list(self.operation_schema.parameters),
                            "full_schema_version": self.full_operation_schema.version,
                            "max_operations_per_step": self.max_operations_per_step,
                            "operations": [
                                {
                                    "name": op.name,
                                    "op": op.op,
                                    "value": op.value,
                                    "rationale": op.rationale,
                                }
                                for op in proposal
                            ],
                            "applied": apply_result.records,
                            "validation_log": validation_log,
                        }
                        patch_text = apply_result.patch
                    operations_path.write_text(
                        json.dumps(operations_payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    state.patch_path = self._edit_artifact_path(state, "patch", edit_index, num_edits, "diff")
                    state.patch_path.write_text(patch_text, encoding="utf-8")
                    edit_record = {
                        "edit_index": edit_index,
                        "total_edits": num_edits,
                        "description": operations_payload["summary"],
                        "prompt_path": str(state.prompt_path),
                        "response_path": str(state.response_path),
                        "patch_path": str(state.patch_path),
                        "operations_path": str(operations_path),
                        "llm_response": state.llm_response,
                        "llm_response_truncated": state.llm_response_truncated,
                        "token_usage": completion_tokens,
                        "token_usage_detail": token_usage_detail,
                        "logprobs_path": None if logprobs_path is None else str(logprobs_path),
                        "logprob_summary": logprob_summary,
                        "edit_source": self.config.generator,
                        "detected_edit_count": len(action.operations or []),
                        "action_kind": action.kind,
                        "operations": operations_payload["operations"],
                        "feature_action": operations_payload.get("feature_action"),
                        "applied_operations": apply_result.records,
                        "status": "applied",
                    }
                    state.edits.append(edit_record)
                    self._collect_operation_action(
                        action=action,
                        active_schema=active_schema_before,
                        inactive_schema=inactive_schema_before,
                        expansion_history=expansion_history_before,
                        current_text=current_text,
                        parent=parent,
                        state=state,
                        search_note=search_note,
                        prior_operations=prior_operations,
                        operations_payload=operations_payload,
                        operations_path=operations_path,
                        edit_index=edit_index,
                    )
                    prior_operations.extend(apply_result.records)
                    current_text = apply_result.text

                if num_edits > 1:
                    self._write_latest_artifact_aliases(state)
                    latest_operations = state.edits[-1].get("operations_path")
                    if latest_operations:
                        shutil.copy2(latest_operations, state.workdir / "operations.json")
                state.train_path.write_text(current_text, encoding="utf-8")
                state.description = " | ".join(
                    edit["description"] for edit in state.edits if edit.get("description")
                )
                state.description = state.description[:300]
                state.metrics["operation_schema_version"] = self.operation_schema.version
                state.metrics["operation_schema_feature_names"] = list(self.operation_schema.parameters)
                state.metrics["operation_schema_feature_count"] = len(self.operation_schema.parameters)
                state.metrics["operation_feature_expansions"] = list(self.expansion_history)
                state.metrics["feature_only_action"] = bool(state.edits) and all(
                    edit.get("action_kind") == "feature" for edit in state.edits
                )
                state.metrics["operations"] = [
                    record
                    for edit in state.edits
                    for record in edit.get("applied_operations", [])
                    if isinstance(record, dict)
                ]
                state.status = "generated"
        except Exception as exc:
            state.status = "generation_error"
            state.score = self.config.failure_score
            state.error = repr(exc)
            state.description = " | ".join(
                edit["description"] for edit in state.edits if edit.get("description")
            )
            state.description = state.description[:300]
            if not state.train_path.exists():
                state.train_path.write_text(current_text, encoding="utf-8")

        self._write_state_meta(state)
        self._record_manifest(state)
        self._progress_status(f"generated {state.state_id}")
        return state

    def _collect_operation_action(
        self,
        *,
        action: GeneratorAction,
        active_schema: OperationSchema,
        inactive_schema: OperationSchema,
        expansion_history: list[dict[str, Any]],
        current_text: str,
        parent: SearchState,
        state: SearchState,
        search_note: str,
        prior_operations: list[dict[str, Any]],
        operations_payload: dict[str, Any],
        operations_path: Path,
        edit_index: int,
    ) -> None:
        if not self._data_sink.enabled:
            return

        current_values = extract_top_level_assignment_values(current_text)
        active_parameters = operation_schema_ir_parameters(
            active_schema,
            current_values,
        )
        inactive_parameters = operation_schema_ir_parameters(
            inactive_schema,
            current_values,
        )
        allowed_actions = ["propose"]
        if inactive_parameters:
            allowed_actions.append("expand_design_space")
        if self.allow_new_feature_specs:
            allowed_actions.append("add_new_parameter")

        if action.kind == "feature":
            if action.feature is None:
                raise ValueError("feature action did not include a feature")
            if action.feature.name in inactive_schema.parameters:
                action_type = "expand_design_space"
                payload = {
                    "activate": action.feature.name,
                    "initial_value": current_values.get(action.feature.name),
                }
                summary = f"Activate {action.feature.name}"
            else:
                action_type = "add_new_parameter"
                payload = {
                    "parameter": operation_parameter_to_json(action.feature),
                    "code_sketch": None,
                    "why_new_axis": action.rationale or None,
                }
                summary = f"Add operation feature {action.feature.name}"
            ir_action = {
                "type": action_type,
                "reasoning": action.rationale or None,
                "payload": payload,
                "summary": summary,
            }
        else:
            operations = action.operations or []
            ir_action = {
                "type": "propose",
                "reasoning": " ".join(
                    operation.rationale
                    for operation in operations
                    if operation.rationale
                )
                or None,
                "payload": {
                    "candidates": [
                        {
                            "parent": parent.state_id,
                            "edits": [
                                {
                                    "parameter": operation.name,
                                    "edit_op": operation.op,
                                    "value": operation.value,
                                    "rationale": operation.rationale or None,
                                }
                                for operation in operations
                            ],
                        }
                    ]
                },
                "summary": operations_payload.get("summary"),
            }

        observations = nanogpt_ir_observations(self)
        best_so_far = observations[0] if observations else None
        acquisition_context = str(
            getattr(self.config, "acquisition_context", "") or ""
        ).strip()
        raw_context: dict[str, Any] = {"parent_train_py": current_text}
        if search_note:
            raw_context["search_state_note"] = search_note
        if self.config.feedback_context.strip():
            raw_context["feedback"] = self.config.feedback_context.strip()

        ir = make_parameter_edit_ir(
            task_id="nanogpt",
            domain="training_program",
            task_description=self.config.task_context,
            objectives=[
                {
                    "name": self.config.score_key,
                    "direction": "minimize" if self.config.minimize else "maximize",
                    "description": (
                        "Objective measured after executing the candidate train.py."
                    ),
                }
            ],
            active_parameters=active_parameters,
            inactive_parameters=inactive_parameters,
            action=ir_action,
            request_description=(
                "Choose one valid action: propose edits to active parameters or "
                "update the reservoir expansion schema."
            ),
            design_space_description=active_schema.description,
            allowed_actions=allowed_actions,
            num_candidates=1,
            max_edits_per_candidate=self.max_operations_per_step,
            round_idx=self.current_iteration,
            num_evaluated=sum(
                state.score is not None and finite_score(state.score)
                for state in self.states
            ),
            observations=observations,
            best_so_far=best_so_far,
            surrogate_feedback=(
                {"description": acquisition_context} if acquisition_context else None
            ),
            expansion_history=[
                {
                    "round": item.get("iteration"),
                    "activated": item.get("name"),
                    "reason": item.get("rationale") or item.get("source"),
                }
                for item in expansion_history
            ],
            applied_this_transition=prior_operations,
            allows_new_parameters=self.allow_new_feature_specs,
            reasoning_available=True,
            raw_context=raw_context,
        )
        self._data_sink.append(
            ir,
            provenance={
                "task": "nanogpt",
                "run_dir": str(self.config.out_dir),
                "state_id": state.state_id,
                "parent_state_id": parent.state_id,
                "iteration": self.current_iteration,
                "edit_index": edit_index,
                "generator": self.config.generator,
                "action_source": action.source or self.config.generator,
                "prompt_path": str(state.prompt_path),
                "operations_path": str(operations_path),
            },
        )

    async def _call_operation_generator(
        self,
        prompt: str,
        current_text: str,
    ) -> tuple[GeneratorAction, str, Any, list[dict[str, Any]]]:
        if self.config.generator == "operation_mock":
            action = make_mock_generator_action(current_text, self, self._mock_counter)
            self._mock_counter += 1
            if action.kind == "feature" and action.feature is not None:
                response_payload = {
                    "summary": "mock dynamic feature proposal",
                    "feature": operation_parameter_to_json(action.feature),
                }
            else:
                response_payload = {
                    "summary": "mock dynamic operation proposal",
                    "operations": [
                        {
                            "name": op.name,
                            "op": op.op,
                            "value": op.value,
                            "rationale": op.rationale,
                        }
                        for op in (action.operations or [])
                    ],
                }
            response = json.dumps(response_payload, indent=2, sort_keys=True)
            validation_log = [{"attempt": 1, "status": "accepted", "source": "operation_mock"}]
            return action, response, 0, validation_log

        if self.config.generator not in {"operation_tool", "operation_tool_plain_text"}:
            raise ValueError(f"OperationSearchEngine does not support generator {self.config.generator!r}.")

        from tasks.nanogpt.core.api_generate import (
            add_usage,
            get_tool_call_arguments,
            get_tool_call_name,
            openai_compatible_chat_turn,
            openai_compatible_generate,
        )

        plain_text_mode = self.config.generator == "operation_tool_plain_text"
        plain_text_instruction = (
            "\nIn this plain-text compatibility mode, native API tool calls may be unavailable. "
            "If you cannot return a real tool call, output only one of these plain-text forms: "
            "a raw JSON object with an operations array, or an <operation_tool_transcript> JSON "
            "transcript containing one propose_train_operations/propose_operation_feature tool call."
            if plain_text_mode
            else ""
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You propose train.py search actions using a dynamically updated "
                    "reservoir expansion schema. Call exactly one provided tool: either "
                    "propose_train_operations to edit active expansion parameters, or "
                    "propose_operation_feature to activate a new expansion parameter."
                    + plain_text_instruction
                ),
            },
            {"role": "user", "content": prompt},
        ]
        total_usage: dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        transcript: list[dict[str, Any]] = []
        validation_log: list[dict[str, Any]] = []
        last_error = "operation_tool did not return a valid proposal."
        attempts = 1 + self.operation_retries
        for attempt in range(1, attempts + 1):
            if plain_text_mode:
                content, usage = await openai_compatible_generate(
                    messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    stop=self.config.stop,
                    llm_url=self.config.llm_url,
                    llm_model_name=self.config.llm_model_name,
                    disable_thinking=self.config.disable_thinking,
                    api_key=self.config.api_key,
                    tools=None,
                    tool_choice=None,
                    logprobs=self.config.request_logprobs,
                    top_logprobs=self.config.top_logprobs,
                )
                assistant_message = {"role": "assistant", "content": "" if content is None else str(content)}
            else:
                assistant_message, usage = await openai_compatible_chat_turn(
                    messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    stop=self.config.stop,
                    llm_url=self.config.llm_url,
                    llm_model_name=self.config.llm_model_name,
                    disable_thinking=self.config.disable_thinking,
                    api_key=self.config.api_key,
                    tools=make_dynamic_operation_tools(self),
                    tool_choice="auto",
                    logprobs=self.config.request_logprobs,
                    top_logprobs=self.config.top_logprobs,
                )
            total_usage = add_usage(total_usage, usage)
            messages.append(assistant_message)
            transcript.append(compact_operation_message_for_log(assistant_message))
            tool_calls = assistant_message.get("tool_calls") if isinstance(assistant_message, dict) else None
            proposal_name = ""
            proposal_args: dict[str, Any] | None = None
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    tool_name = get_tool_call_name(tool_call)
                    if tool_name in {"propose_train_operations", "propose_operation_feature"}:
                        proposal_name = tool_name
                        proposal_args = get_tool_call_arguments(tool_call)
                        break
            if proposal_args is None:
                content = assistant_message.get("content") if isinstance(assistant_message, dict) else None
                proposal_name, proposal_args = extract_dynamic_operation_json_from_text(content if isinstance(content, str) else "")
            try:
                action = validate_generator_action(proposal_name, proposal_args, self, current_text=current_text)
                validation_log.append({"attempt": attempt, "status": "accepted"})
                response = (
                    "<operation_tool_transcript>\n"
                    + json.dumps(transcript, indent=2, sort_keys=True)
                    + "\n</operation_tool_transcript>\n"
                )
                return action, response, total_usage, validation_log
            except ValueError as exc:
                last_error = str(exc)
                validation_log.append({"attempt": attempt, "status": "rejected", "error": last_error})
                if attempt < attempts:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The operation proposal was rejected: "
                                f"{last_error}\n"
                                "Call exactly one valid tool. Use propose_train_operations for active "
                                "features or propose_operation_feature for one valid inactive feature. "
                                "In plain-text mode, output only a raw JSON payload or a valid "
                                "<operation_tool_transcript> block."
                            ),
                        }
                    )
        response = (
            "<operation_tool_transcript>\n"
            + json.dumps(transcript, indent=2, sort_keys=True)
            + "\n</operation_tool_transcript>\n"
        )
        raise ValueError(f"{last_error}\n{response[:2000]}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        description=(
            "Run iterative dynamically expanded model-based train.py search: start from "
            "a subset of operation features, let the LLM activate extra features during "
            "search, score unevaluated states with a GP surrogate, evaluate selected "
            "candidates, then add results to a reusable training buffer."
        ),
    )
    parser.add_argument(
        "--method",
        choices=sorted(SEARCH_METHOD_ALIASES),
        default="auto",
        help=(
            "Inner surrogate-guided search method. auto preserves the old behavior: "
            "tree_search when --beam-width is 0, beam_search when --beam-width is positive."
        ),
    )
    parser.add_argument("--breadth", type=int, default=2, help="Children generated per expanded state.")
    parser.add_argument("--depth", type=int, default=2, help="Depth of each model-based search tree.")
    parser.add_argument("--iterations", type=int, default=3, help="Outer model/evaluate/update iterations.")
    parser.add_argument(
        "--beam-width",
        type=int,
        default=0,
        help=(
            "Beam states kept per depth for beam_search/auto. "
            "0 means auto uses tree_search; explicit beam_search with 0 keeps breadth states."
        ),
    )
    parser.add_argument(
        "--select-from",
        choices=["leaves", "all"],
        default="leaves",
        help="Choose the real-evaluation candidate from final leaves or all surrogate-scored states.",
    )
    parser.add_argument(
        "--seed-policy",
        choices=["original", "latest", "best"],
        default="best",
        help=(
            "Root train.py for each outer iteration. original always uses --train-file; "
            "latest uses the most recently evaluated candidate; best uses the best evaluated candidate."
        ),
    )
    parser.add_argument(
        "--max-generated-per-iteration",
        type=int,
        default=1024,
        help="Safety cap on generated states per outer iteration. 0 disables the cap.",
    )
    parser.add_argument("--evaluate-root", action="store_true", help="Evaluate the root state in iteration 1.")
    parser.add_argument("--skip-eval", action="store_true", help="Do not execute the selected final candidate.")
    parser.add_argument(
        "--max-real-evaluations",
        type=int,
        default=0,
        help="Maximum real train.py executions across all iterations. 0 means no cap.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help=(
            "Number of real evaluated warm-up candidates to collect before model-based iterations. "
            "These points are appended to the same GP buffer."
        ),
    )
    parser.add_argument(
        "--warmup-include-root",
        action="store_true",
        help="Evaluate the seed train.py as the first warm-up buffer point.",
    )
    parser.add_argument(
        "--warmup-strategy",
        choices=["auto", "random_operation", "agent"],
        default="auto",
        help=(
            "Warm-up proposal strategy. auto uses random_operation for operation generators with a schema, "
            "otherwise agent. agent asks the configured generator to propose warm-up edits."
        ),
    )
    parser.add_argument(
        "--warmup-seed",
        type=int,
        default=0,
        help="Random seed for random_operation warm-up. 0 uses a time-derived seed.",
    )
    parser.add_argument(
        "--warmup-updates-seed",
        action="store_true",
        help=(
            "Allow evaluated warm-up candidates to update latest/best seed selection before iteration 1. "
            "By default warm-up only trains the GP buffer and does not change the first search root."
        ),
    )

    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--train-file", type=Path, default=Path("train.py"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Parent directory for model-based runs. Default: runs/model_based.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "Existing model-based run directory, model_based_summary.json, or summary.json to append to. "
            "--iterations is interpreted as additional iterations when resuming."
        ),
    )
    parser.add_argument("--run-name", default="", help="Optional explicit child run folder name.")
    parser.add_argument("--export-best", type=Path, default=None, help="Optional path to copy the best evaluated train.py.")

    parser.add_argument("--eval-command", default="uv run python {train_path}")
    parser.add_argument("--eval-shell", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--score-key", default="val_bpb")
    parser.add_argument("--maximize", action="store_true")
    parser.add_argument("--failure-score", type=float, default=1.0e9)
    parser.add_argument(
        "--gp-reject-score-at-or-above",
        type=float,
        default=None,
        help=(
            "Do not load or append buffer rows with scores >= this threshold. "
            "Default: --failure-score for minimize runs."
        ),
    )
    parser.add_argument(
        "--gp-reject-score-at-or-below",
        type=float,
        default=None,
        help=(
            "Do not load or append buffer rows with scores <= this threshold. "
            "Default: --failure-score for maximize runs."
        ),
    )
    parser.add_argument(
        "--gp-allow-failure-status",
        action="store_true",
        help="Allow failed evaluation statuses such as crash/timeout/score_missing into the GP buffer.",
    )

    parser.add_argument("--generator", choices=sorted(GENERATORS), default="api")
    parser.add_argument(
        "--llm-url",
        default=(
            os.environ.get("TTS_LLM_URL")
            or os.environ.get("LLM_BASE_URL")
            or ""
        ),
    )
    parser.add_argument(
        "--llm-model-name",
        default=(
            os.environ.get("TTS_LLM_MODEL")
            or os.environ.get("LLM_MODEL_NAME")
            or os.environ.get("LLM_MODEL")
            or ""
        ),
    )
    parser.add_argument(
        "--api-key",
        default=(
            os.environ.get("TTS_LLM_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--logprobs", action="store_true")
    parser.add_argument("--top-logprobs", type=int, default=None)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--num-edits-per-step", type=int, default=1)
    parser.add_argument("--prompt-max-chars", type=int, default=100_000)
    parser.add_argument("--context-file", type=Path, default=None)
    parser.add_argument("--context", default="")
    parser.add_argument("--instruction", default="")
    parser.add_argument(
        "--feedback-max-rows",
        type=int,
        default=12,
        help="Maximum recent/best feedback rows included in later LLM prompts. 0 disables feedback injection.",
    )
    parser.add_argument(
        "--feedback-tsv",
        type=Path,
        default=None,
        help="Optional path for iteration feedback TSV. Default: run_dir/iteration_feedback.tsv.",
    )
    parser.add_argument(
        "--acquisition-feedback",
        choices=["none", "brief", "verbose"],
        default="none",
        help=(
            "Add qualitative GP acquisition-decomposition guidance to operation_tool prompts. "
            "This steers the LLM with predicted quality/uncertainty directions without exposing raw GP values."
        ),
    )
    parser.add_argument(
        "--acquisition-feedback-probes",
        type=int,
        default=96,
        help="Maximum local schema-operation probes used to summarize GP guidance for each parent.",
    )
    parser.add_argument(
        "--acquisition-feedback-top-k",
        type=int,
        default=3,
        help="Number of top exploitation/exploration/risk hints to include in acquisition feedback.",
    )
    parser.add_argument(
        "--acquisition-feedback-max-chars",
        type=int,
        default=1800,
        help="Maximum characters of acquisition feedback inserted into each operation prompt.",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--progress-width", type=int, default=28)
    parser.add_argument("--response-log-chars", type=int, default=20_000)
    parser.add_argument(
        "--operation-schema",
        type=Path,
        default=None,
        help=(
            "JSON schema containing all reservoir expansion parameters. Required for "
            "--generator operation_tool, operation_tool_plain_text, or operation_mock."
        ),
    )
    parser.add_argument(
        "--operation-retries",
        type=int,
        default=2,
        help="LLM repair retries when operation_tool emits invalid operations.",
    )
    parser.add_argument(
        "--max-operations-per-step",
        type=int,
        default=2,
        help="Maximum structured operations allowed in one generated child.",
    )
    parser.add_argument(
        "--operation-surrogate",
        "--operation-features",
        dest="operation_features",
        action="store_true",
        help="Represent candidates for the surrogate through the active expansion schema.",
    )
    parser.add_argument(
        "--initial-expansion-parameters",
        "--initial-operation-features",
        dest="initial_operation_features",
        default="5",
        help=(
            "Initial active expansion-schema parameters. Use an integer count such as 5, "
            "or comma-separated schema parameter names. Default: first 5 schema parameters."
        ),
    )
    parser.add_argument(
        "--max-expansion-parameters",
        "--max-active-operation-features",
        dest="max_active_operation_features",
        type=int,
        default=0,
        help="Maximum active expansion-schema parameters. 0 means the full declared schema.",
    )
    parser.add_argument(
        "--disable-expansion-schema-updates",
        "--disable-feature-expansion",
        dest="allow_feature_expansion",
        action="store_false",
        help="Keep the initial reservoir expansion schema fixed.",
    )
    parser.set_defaults(allow_feature_expansion=True)
    parser.add_argument(
        "--allow-new-expansion-parameters",
        "--allow-new-feature-specs",
        dest="allow_new_feature_specs",
        action="store_true",
        help=(
            "Allow the LLM to propose a new top-level assignment parameter. By default "
            "it may only activate inactive parameters already present in --operation-schema."
        ),
    )
    parser.add_argument(
        "--mock-expand-every",
        type=int,
        default=0,
        help="For operation_mock tests, activate one inactive expansion parameter every N mock generations. 0 disables.",
    )

    parser.add_argument(
        "--buffer",
        type=Path,
        default=None,
        help=(
            "Optional reusable JSONL buffer of real evaluated states used to train the GP surrogate. "
            "Default: write and read the run-local out-dir/model_based_buffer.jsonl."
        ),
    )
    parser.add_argument(
        "--surrogate-mode",
        choices=["lcb", "ucb", "mean", "ei"],
        default="lcb",
        help=(
            "Shared posterior acquisition used to choose surrogate candidates. "
            "The adapter converts its larger-is-better score to the legacy "
            "lower-is-better selection_score."
        ),
    )
    parser.add_argument("--gp-beta", type=float, default=1.0, help="LCB/UCB exploration coefficient.")
    parser.add_argument("--gp-xi", type=float, default=0.001, help="Expected-improvement margin.")
    parser.add_argument("--gp-lengthscale", type=float, default=1.5)
    parser.add_argument("--gp-noise", type=float, default=1.0e-4)
    parser.add_argument("--prior-score", type=float, default=1.0)
    parser.add_argument("--prior-std", type=float, default=0.15)
    parser.add_argument("--hash-dims", type=int, default=48)
    args = parser.parse_args(argv)
    args._explicit_options = explicit_options_from_argv(argv)
    return args


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    resume_info = resolve_model_based_resume_info(args.resume_from, project_root) if args.resume_from is not None else None
    if resume_info is not None:
        apply_model_based_resume_defaults(args, resume_info, project_root)
    train_file = args.train_file if args.train_file.is_absolute() else project_root / args.train_file
    operation_schema = resolve_operation_schema(args, project_root)
    if operation_schema is not None:
        args.operation_features = True
    args.operation_schema_object = operation_schema
    if (
        operation_schema is not None
        and bool(getattr(args, "allow_feature_expansion", True))
        and int(getattr(args, "concurrency", 1)) != 1
    ):
        print("warning: reservoir schema updates use global active state; forcing --concurrency 1.", file=sys.stderr)
        args.concurrency = 1
    effective_method = resolve_search_method(args)
    args.effective_method = effective_method
    if resume_info is None:
        out_parent_dir = project_root / "runs" / "model_based" if args.out_dir is None else args.out_dir
        if not out_parent_dir.is_absolute():
            out_parent_dir = project_root / out_parent_dir
        run_name = safe_path_tag(args.run_name) if args.run_name.strip() else default_run_name(args, train_file)
        out_dir = make_unique_run_dir(out_parent_dir, run_name)
        starting_iteration = 1
        previous_iteration_records: list[dict[str, Any]] = []
        previous_warmup_record: dict[str, Any] | None = None
        previous_real_evaluations = 0
    else:
        out_dir = resume_info["run_dir"]
        out_parent_dir = out_dir.parent
        run_name = safe_path_tag(args.run_name) if args.run_name.strip() else out_dir.name
        starting_iteration = int(resume_info["next_iteration"])
        previous_iteration_records = list(resume_info.get("iterations", []))
        previous_warmup_record = resume_info.get("warmup") if isinstance(resume_info.get("warmup"), dict) else None
        previous_real_evaluations = int(resume_info.get("real_evaluations") or 0)
    run_buffer_path = out_dir / "model_based_buffer.jsonl"
    buffer_path = resolve_buffer_path(args.buffer, project_root, run_buffer_path)
    buffer_snapshot_path = run_buffer_path
    log_path = out_dir / "model_based.log"
    logger = RunLogger(log_path)
    logger.write(
        ("resume " if resume_info is not None else "start ")
        +
        f"run={run_name} method={args.method} effective_method={effective_method} "
        f"train_file={train_file} buffer={buffer_path} run_buffer={buffer_snapshot_path} "
        f"start_iteration={starting_iteration} additional_iterations={max(0, int(args.iterations))}"
    )
    if operation_schema is not None:
        schema_out_path = out_dir / "operation_schema.json"
        schema_out_path.write_text(
            json.dumps(operation_schema_to_json(operation_schema), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        logger.write(
            f"operation_full_schema version={operation_schema.version} "
            f"path={operation_schema.path} feature_dim={operation_feature_dim(operation_schema)}"
        )
    if resume_info is None or not buffer_snapshot_path.exists():
        sync_run_buffer(buffer_path, buffer_snapshot_path)

    if "{train_path}" not in args.eval_command:
        warning = "--eval-command does not contain {train_path}; evaluation will not run the generated child state."
        print(f"warning: {warning}", file=sys.stderr)
        logger.write(f"warning {warning}")
    if args.generator in OPERATION_GENERATORS and operation_schema is None:
        raise SystemExit(
            "--generator operation_tool/operation_tool_plain_text/operation_mock requires --operation-schema, "
            "or a default resources/schemas/real_operations.json / resources/schemas/mock_operations.json."
        )

    generated_estimate = estimate_generated(args.breadth, args.depth, args.beam_width, effective_method)
    total_progress = estimate_progress_total(args, generated_estimate)
    if args.max_generated_per_iteration > 0 and generated_estimate > args.max_generated_per_iteration:
        raise SystemExit(
            f"Refusing to generate about {generated_estimate} states per iteration. "
            f"Raise --max-generated-per-iteration or use --beam-width."
        )

    task_context = build_task_context(project_root, args)
    config = SearchConfig(
        project_root=project_root,
        seed_train_path=train_file,
        out_dir=out_dir,
        eval_command=args.eval_command,
        eval_shell=args.eval_shell,
        score_key=args.score_key,
        minimize=not args.maximize,
        failure_score=args.failure_score,
        timeout_seconds=args.timeout_seconds,
        run_evaluation=not args.skip_eval,
        generator=args.generator,
        llm_url=args.llm_url,
        llm_model_name=args.llm_model_name,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        request_logprobs=args.logprobs or args.top_logprobs is not None,
        top_logprobs=args.top_logprobs,
        disable_thinking=args.disable_thinking,
        concurrency=args.concurrency,
        num_edits_per_step=args.num_edits_per_step,
        prompt_max_chars=args.prompt_max_chars,
        task_context=task_context,
        extra_instruction=args.instruction,
        feedback_context="",
        show_progress=False,
        progress_width=args.progress_width,
        response_log_chars=args.response_log_chars,
    )
    if operation_schema is not None:
        engine: SearchEngine = OperationSearchEngine(config, operation_schema, args)
    else:
        engine = SearchEngine(config)
    active_operation_schema = (
        engine.operation_schema if isinstance(engine, OperationSearchEngine) else operation_schema
    )
    if isinstance(engine, OperationSearchEngine):
        active_schema_out_path = out_dir / "active_operation_schema.json"
        active_schema_out_path.write_text(
            json.dumps(operation_schema_to_json(engine.operation_schema), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        logger.write(
            f"operation_active_schema version={engine.operation_schema.version} "
            f"features={list(engine.operation_schema.parameters)} "
            f"feature_dim={operation_feature_dim(engine.operation_schema)}"
        )
    ldm_task_spec = describe_ldm_task(
        args,
        operation_schema,
        active_operation_schema,
        effective_method=effective_method,
    )
    ldm_task_spec_dict = ldm_task_spec.to_dict()
    (out_dir / "ldm_task_spec.json").write_text(
        json.dumps(ldm_task_spec_dict, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.write(
        "ldm_task_spec "
        f"candidate_domain={ldm_task_spec.candidate_domain.name} "
        f"dimension={ldm_task_spec.candidate_domain.dimension} "
        f"acquisition={ldm_task_spec.acquisition.name}"
    )
    feedback_path = resolve_feedback_path(args.feedback_tsv, project_root, out_dir)
    feedback_memory = FeedbackMemory(
        feedback_path,
        score_key=args.score_key,
        minimize=not args.maximize,
        max_rows=max(0, int(args.feedback_max_rows)),
    )
    if resume_info is not None:
        load_feedback_memory_rows(feedback_memory, feedback_path)
    engine.config.feedback_context = feedback_memory.prompt_context()
    logger.write(f"feedback path={feedback_path} max_rows={feedback_memory.max_rows}")
    progress_feature_status: Callable[[], str] | None = None
    if isinstance(engine, OperationSearchEngine):
        def progress_feature_status() -> str:
            schema = engine.operation_schema
            return f"feature_dim={len(schema.parameters)} gp_dim={operation_feature_dim(schema)}"

    progress = ModelBasedProgress(
        enabled=not args.no_progress,
        total=total_progress,
        width=args.progress_width,
        score_key=args.score_key,
        minimize=not args.maximize,
        feature_status=progress_feature_status,
    )

    buffer_entries = load_projected_buffer(buffer_path, args, active_operation_schema)
    if resume_info is not None:
        load_existing_search_states(engine, out_dir)
        logger.write(f"loaded resume states={len(engine.states)} next_state_counter={engine._counter}")
    iteration_records: list[dict[str, Any]] = []
    warmup_record: dict[str, Any] = {
        "requested": max(0, int(args.warmup)),
        "strategy": args.warmup_strategy,
        "state_ids": [],
        "scores": [],
        "real_evaluations": 0,
        "buffer_size_before": len(buffer_entries),
        "buffer_size_after": len(buffer_entries),
    }
    if resume_info is not None and previous_warmup_record is not None:
        warmup_record = dict(previous_warmup_record)
        warmup_record.setdefault("resumed", True)
    best_actual: SearchState | None = (
        state_from_id(engine, resume_info.get("best_state_id")) if resume_info is not None else None
    )
    latest_actual: SearchState | None = latest_evaluated_child_state(engine) if resume_info is not None else None
    if best_actual is not None and best_actual.score is not None and finite_score(best_actual.score):
        progress.best_score = float(best_actual.score)
    real_evaluations = previous_real_evaluations
    engine.evaluation_count = previous_real_evaluations

    from tasks.nanogpt.core import engine_adapters

    tracker = engine_adapters.NanogptCampaignTracker(
        best_state=best_actual, latest_state=latest_actual
    )
    if best_actual is not None and best_actual.score is not None:
        tracker.previous_best_score = float(best_actual.score)
    engine_state = LDMEngineState()
    if resume_info is not None:
        checkpoint_path = out_dir / "checkpoint.json"
        if checkpoint_path.exists():
            checkpoint_payload = json.loads(
                checkpoint_path.read_text(encoding="utf-8")
            )
            engine_state = LDMEngineState.from_checkpoint(
                checkpoint_payload.get("state", checkpoint_payload)
            )
    warmup_target = (
        max(0, int(args.warmup))
        if resume_info is None and not args.skip_eval
        else 0
    )
    engine_iterations = (
        engine_state.next_round
        + warmup_target
        + max(0, int(args.iterations))
    )
    evaluation_limit = engine_iterations * 2
    if args.max_real_evaluations > 0:
        evaluation_limit = max(0, int(args.max_real_evaluations))
    search_expander = engine_adapters.NanogptIterationExpander(
        engine=engine,
        args=args,
        logger=logger,
        progress=progress,
        buffer_entries=buffer_entries,
        feedback_memory=feedback_memory,
        buffer_path=buffer_path,
        run_buffer_path=buffer_snapshot_path,
        run_name=run_name,
        train_file=train_file,
        tracker=tracker,
    )
    warmup_expander = engine_adapters.NanogptWarmupExpander(
        engine=engine,
        args=args,
        logger=logger,
        progress=progress,
        requested=warmup_target,
    )
    expander = engine_adapters.NanogptCampaignExpander(
        warmup_expander,
        search_expander,
        warmup_target=warmup_target,
    )
    evaluator = engine_adapters.NanogptEvaluator(
        engine=engine,
        args=args,
        logger=logger,
        progress=progress,
        buffer_entries=buffer_entries,
        feedback_memory=feedback_memory,
        buffer_path=buffer_path,
        run_buffer_path=buffer_snapshot_path,
        run_name=run_name,
        tracker=tracker,
    )
    selector = engine_adapters.NanogptPrecomputedSelector(
        search_expander,
        objective_name=engine.config.score_key,
    )
    recipe = CampaignRecipe(
        task_spec=ldm_task_spec,
        expander=expander,
        candidate_domain=engine_adapters.NanogptCandidateDomain(),
        evaluator=evaluator,
        selector=selector,
    )
    campaign = await async_run_campaign(
        CampaignRequest(
            run_dir=out_dir,
            config=_jsonable_args(args),
            resume=resume_info is not None and (out_dir / "campaign.json").exists(),
            state=engine_state,
            budget=CampaignBudget(
                rounds=engine_iterations,
                reservoir_size=max(
                    2,
                    int(generated_estimate) + (1 if args.evaluate_root else 0),
                ),
                batch_size=2,
                max_evaluation_attempts=evaluation_limit,
                max_empty_reservoir_rounds=max(engine_iterations, 1),
            ),
        ),
        recipe,
    )
    engine_result = campaign.engine
    if resume_info is None:
        warmup_record = _warmup_record_from_engine(
            engine_result,
            engine,
            args,
            warmup_expander,
            buffer_size_before=int(warmup_record["buffer_size_before"]),
        )
    best_actual = tracker.best_state or best_actual
    latest_actual = tracker.latest_state or latest_actual
    iteration_records = _iteration_records_from_expander(
        search_expander.records,
        engine_result,
        engine,
        args,
        buffer_entries,
        logger,
    )
    for record in iteration_records:
        actual_states = record.get("actual_states")
        if not isinstance(actual_states, list):
            actual_states = []
        for actual_state in actual_states:
            if (
                not isinstance(actual_state, SearchState)
                or actual_state.score is None
                or not finite_score(actual_state.score)
            ):
                continue
            latest_actual = actual_state
            if best_actual is None or is_better(
                actual_state.score,
                best_actual.score,
                minimize=engine.config.minimize,
            ):
                best_actual = actual_state
    if resume_info is None:
        real_evaluations += int(warmup_record.get("real_evaluations") or 0)
    real_evaluations += sum(
        int(record.get("real_evaluations") or 0) for record in iteration_records
    )
    progress.finish("done")


    all_iteration_records = previous_iteration_records + iteration_records
    summary_args = dict(vars(args))
    summary_args.pop("operation_schema_object", None)
    summary_args.pop("_explicit_options", None)
    summary_args["resume_from"] = None if resume_info is None else str(resume_info["run_dir"])
    summary_args["continued_from_iteration"] = None if resume_info is None else starting_iteration - 1
    summary_args["additional_iterations_requested"] = max(0, int(args.iterations))
    if isinstance(engine, OperationSearchEngine):
        summary_args["operation_schema_version"] = engine.operation_schema.version
        summary_args["operation_schema_path"] = None if engine.full_operation_schema.path is None else str(engine.full_operation_schema.path)
        summary_args["operation_feature_version"] = operation_feature_version(engine.operation_schema)
        summary_args["operation_full_schema_version"] = engine.full_operation_schema.version
        summary_args["operation_schema_feature_names"] = list(engine.operation_schema.parameters)
        summary_args["operation_schema_feature_count"] = len(engine.operation_schema.parameters)
        summary_args["operation_schema_feature_dim"] = operation_feature_dim(engine.operation_schema)
        summary_args["buffer_projection_mode"] = "reproject_jsonl_rows_to_active_operation_schema"
        summary_args["operation_feature_expansions"] = list(engine.expansion_history)
    summary_args["run_name"] = run_name
    summary_args["run_parent_dir"] = str(out_parent_dir)
    summary_args["run_dir"] = str(out_dir)
    summary_args["buffer"] = str(buffer_path)
    summary_args["run_buffer"] = str(buffer_snapshot_path)
    summary_args["log"] = str(log_path)
    summary_args["feedback_tsv"] = str(feedback_path)
    summary_args["ldm_task_spec"] = ldm_task_spec_dict
    summary_path = engine.write_summary(method=f"model_based_{effective_method}", args=summary_args, best=best_actual)
    sync_run_buffer(buffer_path, buffer_snapshot_path)
    write_model_based_summary(
        out_dir,
        summary_path,
        buffer_path,
        buffer_snapshot_path,
        log_path,
        feedback_path,
        warmup_record,
        all_iteration_records,
        best_actual,
        ldm_task_spec=ldm_task_spec_dict,
    )
    from tasks.nanogpt.core import engine_adapters as _nanogpt_adapters

    _nanogpt_adapters.merge_engine_summary(summary_path, engine_result.summary)
    logger.write(
        "finish "
        f"new_iterations={len(iteration_records)} total_iterations={len(all_iteration_records)} "
        f"real_evaluations={real_evaluations} "
        f"best_state={None if best_actual is None else best_actual.state_id} "
        f"best_score={None if best_actual is None else best_actual.score}"
    )

    if args.export_best is not None and best_actual is not None:
        export_path = args.export_best if args.export_best.is_absolute() else project_root / args.export_best
        export_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_actual.train_path, export_path)

    payload = {
        "out_dir": str(out_dir.resolve()),
        "summary": str(summary_path.resolve()),
        "model_based_summary": str((out_dir / "model_based_summary.json").resolve()),
        "buffer": str(buffer_path.resolve()),
        "run_buffer": str(buffer_snapshot_path.resolve()),
        "log": str(log_path.resolve()),
        "feedback_tsv": str(feedback_path.resolve()),
        "best_state_id": None if best_actual is None else best_actual.state_id,
        "best_score": None if best_actual is None else best_actual.score,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return json.loads(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key not in {"api_key", "operation_schema_object"}
            },
            default=str,
        )
    )


def _warmup_record_from_engine(
    engine_result,
    engine: SearchEngine,
    args: argparse.Namespace,
    expander,
    *,
    buffer_size_before: int,
) -> dict[str, Any]:
    """Project engine-owned warm-up observations into the legacy summary shape."""

    observations = [
        observation
        for observation in engine_result.state.observations
        if str(observation.candidate.payload.get("kind", "")).startswith("warmup")
    ]
    actual_states: list[SearchState] = []
    for observation in observations:
        if not observation.evaluation.succeeded:
            continue
        state = state_from_id(engine, observation.candidate.payload.get("state_id"))
        if state is not None:
            actual_states.append(state)
    root_state_id = None
    if observations:
        root_state_id = observations[0].candidate.payload.get("root_state_id")
    elif expander.root is not None:
        root_state_id = expander.root.state_id
    return {
        "requested": int(expander.requested),
        "strategy": expander.strategy,
        "rng_seed": int(expander.rng_seed),
        "include_root": bool(args.warmup_include_root),
        "updates_seed": bool(args.warmup_updates_seed),
        "root_state_id": root_state_id,
        "state_ids": [state.state_id for state in actual_states],
        "generated_state_ids": [
            state.state_id
            for state in actual_states
            if not state.metrics.get("warmup_root")
        ],
        "scores": [state.score for state in actual_states],
        "score_key": engine.config.score_key,
        "real_evaluations": len(observations),
        "buffer_size_before": int(buffer_size_before),
        "buffer_size_after": int(buffer_size_before) + len(actual_states),
        "gp_after": {},
        "actual_states": actual_states,
    }


def _iteration_records_from_expander(
    expander_records: list[dict[str, Any]],
    engine_result,
    engine: SearchEngine,
    args: argparse.Namespace,
    buffer_entries: list[Any],
    logger: RunLogger,
) -> list[dict[str, Any]]:
    """Rebuild model-based iteration records from expander rounds + engine observations."""
    observations_by_round: dict[int, list[Any]] = {}
    for observation in engine_result.state.observations:
        if observation.round_idx is None:
            continue
        observations_by_round.setdefault(int(observation.round_idx), []).append(observation)
    records: list[dict[str, Any]] = []
    running_best: float | None = None
    buffer_at: int | None = None
    for expander_record in expander_records:
        round_observations = observations_by_round.get(
            int(expander_record.get("round_idx", 0)), []
        )
        actual_states: list[SearchState] = []
        for observation in round_observations:
            if not observation.evaluation.succeeded:
                continue
            state = state_from_id(engine, observation.candidate.payload.get("state_id"))
            if state is not None:
                actual_states.append(state)
        selected_state = expander_record.get("_selected_state")
        selected_real_score = None
        if selected_state is not None:
            for observation in round_observations:
                if (
                    observation.evaluation.succeeded
                    and observation.candidate.payload.get("state_id") == selected_state.state_id
                ):
                    selected_real_score = observation.evaluation.metrics.get(
                        engine.config.score_key
                    )
                    break
        iteration_best = best_score_from_states(
            actual_states, minimize=engine.config.minimize
        )
        best_after = updated_best_score(
            running_best, iteration_best, minimize=engine.config.minimize
        )
        running_best = best_after
        previous_best = expander_record.get("previous_best_score")
        score_delta = None
        if (
            selected_real_score is not None
            and finite_score(selected_real_score)
            and previous_best is not None
        ):
            score_delta = float(selected_real_score) - float(previous_best)
        if buffer_at is None:
            buffer_at = int(expander_record.get("buffer_size_before_iteration", 0))
        buffer_at += len(actual_states)
        gp_after: dict[str, Any] = {}
        try:
            gp_after = GPSurrogate(
                buffer_entries[:buffer_at],
                lengthscale=args.gp_lengthscale,
                noise=args.gp_noise,
                prior_score=args.prior_score,
                prior_std=args.prior_std,
                minimize=engine.config.minimize,
            ).summary()
        except Exception as exc:  # pragma: no cover - defensive GP replay
            logger.write(f"gp_after replay failed iteration={expander_record.get('iteration')} {exc}")
        records.append({
            "iteration": expander_record["iteration"],
            "method": expander_record["method"],
            "root_state_id": expander_record["root_state_id"],
            "selected_state_id": expander_record["selected_state_id"],
            "selected_surrogate_score": expander_record.get("selected_surrogate_score"),
            "selected_pred": expander_record.get("selected_pred"),
            "selected_std": expander_record.get("selected_std"),
            "selected_ei": expander_record.get("selected_ei"),
            "selected_real_score": selected_real_score,
            "score_key": expander_record["score_key"],
            "previous_best_score": previous_best,
            "iteration_best_score": iteration_best,
            "best_score_after_iteration": best_after,
            "selected_score_delta_vs_previous_best": score_delta,
            "selected_improved_previous_best": (
                None
                if selected_real_score is None or previous_best is None
                else is_better(
                    selected_real_score,
                    previous_best,
                    minimize=engine.config.minimize,
                )
            ),
            "gp_before": expander_record.get("gp_before", {}),
            "gp_after": gp_after,
            "generated_count": expander_record.get("generated_count", 0),
            "buffer_size_before_iteration": expander_record.get(
                "buffer_size_before_iteration", 0
            ),
            "buffer_size_after_iteration": buffer_at,
            "real_evaluations": len(actual_states),
            "actual_state_ids": [state.state_id for state in actual_states],
            "actual_states": actual_states,
            "selected_state": selected_state,
        })
    return records


async def run_inner_surrogate_search(
    engine: SearchEngine,
    args: argparse.Namespace,
    root: SearchState,
    surrogate: "GPSurrogate",
    *,
    iteration: int,
    progress: ModelBasedProgress,
) -> tuple[list[SearchState], list[SearchState]]:
    method = resolve_search_method(args)
    adapter = SurrogateSearchAdapter(
        engine,
        args,
        root,
        surrogate,
        iteration=iteration,
        progress=progress,
    )
    traversal = get_traversal_method(method)
    result = await traversal(
        adapter,
        root,
        breadth=max(1, args.breadth),
        depth=max(1, args.depth),
        beam_width=max(1, args.beam_width if args.beam_width > 0 else args.breadth),
        max_evaluations=None,
    )
    return list(result.generated), list(result.leaves)


class SurrogateSearchAdapter:
    """Score shared proposal traversals with nanoGPT's inner GP surrogate."""

    def __init__(
        self,
        engine: SearchEngine,
        args: argparse.Namespace,
        root: SearchState,
        surrogate: "GPSurrogate",
        *,
        iteration: int,
        progress: ModelBasedProgress,
    ) -> None:
        self.engine = engine
        self.args = args
        self.root = root
        self.surrogate = surrogate
        self.iteration = iteration
        self.progress = progress
        self.evaluation_count = 0

    @property
    def evaluation_interval(self) -> int:
        return 1

    def create_seed_state(self) -> SearchState:
        return self.root

    async def expand_state(
        self,
        parent: SearchState,
        count: int,
        *,
        search_note: str = "",
    ) -> list[SearchState]:
        self.progress.status(f"generating children from {parent.state_id}")
        previous_context = getattr(self.engine.config, "acquisition_context", "")
        self.engine.config.acquisition_context = build_acquisition_feedback(
            self.engine,
            self.args,
            parent,
            self.surrogate,
            iteration=self.iteration,
        )
        try:
            return await self.engine.expand_state(
                parent,
                max(1, count),
                search_note=f"model_based iteration {self.iteration}: {search_note}",
            )
        finally:
            self.engine.config.acquisition_context = previous_context

    async def evaluate_many(self, states: list[SearchState]) -> None:
        for state in states:
            self._refresh_operation_features()
            score_with_surrogate(
                self.engine,
                state,
                self.surrogate,
                self.args,
                iteration=self.iteration,
            )
            self.progress.generated(state)
            self.evaluation_count += 1

    async def defer_evaluation_many(self, states: list[SearchState], *, reason: str) -> None:
        del states, reason
        raise RuntimeError("Surrogate proposal traversal scores every generated depth.")

    def should_evaluate_depth(self, depth: int, max_depth: int | None = None) -> bool:
        del depth, max_depth
        return True

    def ranked_states(self, states: list[SearchState] | None = None) -> list[SearchState]:
        candidates = self.engine.states if states is None else states
        return sorted(candidates, key=surrogate_sort_key)

    def best_state(self, states: list[SearchState] | None = None) -> SearchState | None:
        ranked = self.ranked_states(states)
        return ranked[0] if ranked else None

    def reward(self, state: SearchState) -> float:
        ranked = self.ranked_states()
        if state not in ranked:
            return 0.0
        return 1.0 - ranked.index(state) / max(1, len(ranked) - 1)

    def start_progress(self, total: int, *, label: str) -> None:
        del total, label

    def finish_progress(self) -> None:
        return None

    def _refresh_operation_features(self) -> None:
        if not isinstance(self.engine, OperationSearchEngine) or not self.surrogate.entries:
            return
        current_dim = operation_feature_dim(self.engine.operation_schema)
        first_dim = len(self.surrogate.entries[0].feature_vector)
        if first_dim != current_dim:
            self.surrogate.entries = project_buffer_entries(
                self.surrogate.entries,
                self.engine.operation_schema,
                self.args,
            )
            self.surrogate._fit()


def build_acquisition_feedback(
    engine: SearchEngine,
    args: argparse.Namespace,
    parent: SearchState,
    surrogate: "GPSurrogate",
    *,
    iteration: int,
) -> str:
    mode = str(getattr(args, "acquisition_feedback", "none") or "none")
    if mode == "none":
        return ""
    schema = getattr(engine, "operation_schema", None) or getattr(args, "operation_schema_object", None)
    if schema is None:
        return ""
    if not parent.train_path.exists():
        return ""
    try:
        parent_features = featurize_operation_schema(parent.train_path, schema)
        parent_pred = surrogate.predict(
            parent_features.vector,
            mode=args.surrogate_mode,
            beta=args.gp_beta,
            xi=args.gp_xi,
        )
        probes = acquisition_probe_candidates(
            parent.train_path.read_text(encoding="utf-8"),
            schema,
            surrogate,
            args,
            max_probes=max(0, int(getattr(args, "acquisition_feedback_probes", 96))),
        )
    except Exception as exc:
        return f"- GP guidance unavailable for this parent because local probing failed: {exc!r}"
    feedback = format_acquisition_feedback(
        parent,
        parent_pred,
        probes,
        args,
        iteration=iteration,
        mode=mode,
    )
    max_chars = max(200, int(getattr(args, "acquisition_feedback_max_chars", 1800)))
    if len(feedback) > max_chars:
        feedback = feedback[: max_chars - 3].rstrip() + "..."
    return feedback


def acquisition_probe_candidates(
    parent_text: str,
    schema: OperationSchema,
    surrogate: "GPSurrogate",
    args: argparse.Namespace,
    *,
    max_probes: int,
) -> list[dict[str, Any]]:
    current_values = extract_top_level_assignment_values(parent_text)
    probes: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    if max_probes <= 0:
        return probes
    for parameter in schema.parameters.values():
        if parameter.name not in current_values:
            continue
        for value in acquisition_probe_values(parameter, current_values.get(parameter.name)):
            key = (parameter.name, json.dumps(value, sort_keys=True, default=str))
            if key in seen:
                continue
            seen.add(key)
            operation = ValidatedOperation(
                name=parameter.name,
                op="set_choice" if parameter.kind == "choice" else "set_numeric",
                value=value,
                rationale="Local GP acquisition probe.",
            )
            try:
                apply_result = apply_operations_to_train_text(parent_text, [operation], schema)
                features = featurize_operation_text(apply_result.text, schema)
                pred = surrogate.predict(
                    features.vector,
                    mode=args.surrogate_mode,
                    beta=args.gp_beta,
                    xi=args.gp_xi,
                )
            except Exception:
                continue
            probes.append(
                {
                    "name": parameter.name,
                    "kind": parameter.kind,
                    "old_value": current_values.get(parameter.name),
                    "new_value": value,
                    "summary": operation_summary([operation], apply_result.records),
                    "prediction": pred,
                    "params": features.params,
                }
            )
            if len(probes) >= max_probes:
                return probes
    return probes


def acquisition_probe_values(parameter: OperationParameter, current_value: Any) -> list[Any]:
    candidates: list[Any] = []
    if parameter.kind == "choice":
        candidates.extend(parameter.choices)
    elif parameter.kind == "int":
        lo = int(math.ceil(float(parameter.min_value)))
        hi = int(math.floor(float(parameter.max_value)))
        current = as_float(current_value)
        midpoint = int(round((lo + hi) / 2.0))
        candidates.extend([lo, hi, midpoint])
        if current is not None:
            current_int = int(round(current))
            candidates.extend(
                [
                    max(lo, current_int - 2),
                    max(lo, current_int - 1),
                    min(hi, current_int + 1),
                    min(hi, current_int + 2),
                ]
            )
    else:
        lo = float(parameter.min_value)
        hi = float(parameter.max_value)
        current = as_float(current_value)
        if parameter.scale == "log" and lo > 0 and hi > 0:
            midpoint = math.exp((math.log(lo) + math.log(hi)) / 2.0)
        else:
            midpoint = (lo + hi) / 2.0
        candidates.extend([lo, hi, midpoint])
        if current is not None:
            if current > 0:
                candidates.extend([max(lo, current * 0.75), min(hi, current * 1.25)])
            span = hi - lo
            candidates.extend([max(lo, current - 0.1 * span), min(hi, current + 0.1 * span)])
            candidates.append(fallback_operation_value(current_value, parameter))
    output: list[Any] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            value = validate_operation_value(candidate, parameter, index=1)
        except ValueError:
            continue
        if current_value is not None and choice_values_equal(current_value, value):
            continue
        key = json.dumps(value, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def featurize_operation_text(text: str, schema: OperationSchema) -> Features:
    source_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    values = extract_top_level_assignment_values(text)
    vector: list[float] = []
    params: dict[str, Any] = {}
    for parameter in schema.parameters.values():
        present = parameter.name in values
        raw_value = values.get(parameter.name)
        if present:
            params[parameter.name] = raw_value
        if parameter.kind == "choice":
            vector.extend(1.0 if present and choice_values_equal(raw_value, choice) else 0.0 for choice in parameter.choices)
            vector.append(1.0 if present else 0.0)
            continue
        value = as_float(raw_value)
        if value is None:
            vector.append(0.0)
            vector.append(0.0)
            continue
        vector.append(normalize_operation_numeric(value, parameter))
        vector.append(1.0)
    return Features(vector=vector, params=params, source_hash=source_hash)


def format_acquisition_feedback(
    parent: SearchState,
    parent_pred: Prediction,
    probes: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    iteration: int,
    mode: str,
) -> str:
    top_k = max(1, int(getattr(args, "acquisition_feedback_top_k", 3)))
    minimize = not bool(getattr(args, "maximize", False))
    best_probe = None
    if probes:
        best_probe = sorted(probes, key=lambda probe: probe["prediction"].selection_score)[0]
    regime = acquisition_regime(parent_pred, probes, minimize=minimize)
    target = mutation_target_sentence(regime)
    exploit = sorted(probes, key=lambda probe: quality_sort_value(probe["prediction"], minimize=minimize))[:top_k]
    explore = sorted(probes, key=lambda probe: float(probe["prediction"].std), reverse=True)[:top_k]
    acquisition = sorted(probes, key=lambda probe: float(probe["prediction"].selection_score))[:top_k]
    risky = sorted(probes, key=lambda probe: risky_sort_value(probe["prediction"], minimize=minimize), reverse=True)[:top_k]

    lines = [
        f"- Parent regime: {regime}. {regime_description(regime)}",
        f"- Mutation target: {target}",
    ]
    if best_probe is not None:
        lines.append(
            "- Overall GP-acquisition hint: "
            f"{probe_direction_text(best_probe, include_values=False)} looks most acquisition-favored among local one-knob probes."
        )
    if acquisition:
        lines.append("- Acquisition-favored local moves: " + join_probe_phrases(acquisition, include_values=False))
    if exploit:
        lines.append("- Exploitation hints (predicted quality): " + join_probe_phrases(exploit, include_values=False))
    if explore:
        lines.append("- Exploration hints (uncertainty): " + join_probe_phrases(explore, include_values=False))
    if risky:
        lines.append("- Lower-priority or risky local moves: " + join_probe_phrases(risky, include_values=False))
    lines.append(
        "- Use this as qualitative guidance only; the GP is a small-data surrogate, not a guarantee. "
        "Return valid schema operations that remain coherent for the current train.py."
    )
    if mode == "verbose":
        lines.append(
            "- Internal summary: "
            f"iteration={iteration}, parent={parent.state_id}, "
            f"parent_readout={qualitative_readout(parent_pred, minimize=minimize, brief=False)}, "
            f"local_probes={len(probes)}."
        )
    return "\n".join(lines)


def acquisition_regime(parent_pred: Prediction, probes: list[dict[str, Any]], *, minimize: bool) -> str:
    quality_values = [float(parent_pred.mean)]
    std_values = [float(parent_pred.std)]
    for probe in probes:
        pred = probe["prediction"]
        quality_values.append(float(pred.mean))
        std_values.append(float(pred.std))
    quality_rank = percentile_rank(float(parent_pred.mean), quality_values, lower_is_better=minimize)
    uncertainty_rank = percentile_rank(float(parent_pred.std), std_values, lower_is_better=False)
    high_quality = quality_rank >= 0.60
    high_uncertainty = uncertainty_rank >= 0.60
    low_quality = quality_rank <= 0.35
    low_uncertainty = uncertainty_rank <= 0.35
    if high_quality and high_uncertainty:
        return "promising unknown"
    if high_quality and low_uncertainty:
        return "exploit"
    if low_quality and high_uncertainty:
        return "explore"
    if low_quality and low_uncertainty:
        return "uninteresting"
    return "mixed"


def percentile_rank(value: float, values: list[float], *, lower_is_better: bool) -> float:
    finite = sorted(float(item) for item in values if finite_score(item))
    if not finite:
        return 0.5
    if len(finite) == 1:
        return 0.5
    if lower_is_better:
        count = sum(1 for item in finite if item >= value)
    else:
        count = sum(1 for item in finite if item <= value)
    return max(0.0, min(1.0, count / len(finite)))


def regime_description(regime: str) -> str:
    descriptions = {
        "exploit": "The GP sees this parent as relatively strong and well understood.",
        "explore": "The GP is uncertain here, so this branch can usefully test a plausible uncertain direction.",
        "promising unknown": "The GP sees promise while still assigning meaningful uncertainty.",
        "uninteresting": "The GP predicts weak quality with little uncertainty, so local tweaks may have low value.",
        "mixed": "The GP readout is not decisive; balance one plausible quality move with diversity.",
    }
    return descriptions.get(regime, descriptions["mixed"])


def mutation_target_sentence(regime: str) -> str:
    targets = {
        "exploit": "preserve the current structure and make a focused local refinement.",
        "explore": "move toward a different but still plausible schema region to reduce uncertainty.",
        "promising unknown": "preserve promising choices while testing one uncertain complementary knob.",
        "uninteresting": "avoid tiny no-op-like tweaks; use this branch for a more distinct valid move.",
        "mixed": "make one coherent operation that either improves predicted quality or explores a plausible uncertain knob.",
    }
    return targets.get(regime, targets["mixed"])


def quality_sort_value(pred: Prediction, *, minimize: bool) -> float:
    return float(pred.mean) if minimize else -float(pred.mean)


def risky_sort_value(pred: Prediction, *, minimize: bool) -> float:
    if minimize:
        return float(pred.mean) + 0.25 * float(pred.std)
    return -float(pred.mean) + 0.25 * float(pred.std)


def probe_direction_text(probe: dict[str, Any], *, include_values: bool) -> str:
    name = str(probe.get("name") or "")
    old_value = probe.get("old_value")
    new_value = probe.get("new_value")
    direction = value_direction(old_value, new_value)
    if include_values and direction:
        return f"{direction} {name} ({old_value!r} -> {new_value!r})"
    if include_values:
        return f"set {name} {old_value!r} -> {new_value!r}"
    if direction:
        return f"{direction} {name}"
    return f"change {name}"


def value_direction(old_value: Any, new_value: Any) -> str:
    old_number = as_float(old_value)
    new_number = as_float(new_value)
    if old_number is not None and new_number is not None:
        if new_number > old_number:
            return "increase"
        if new_number < old_number:
            return "decrease"
        return ""
    if old_value != new_value:
        return "change"
    return ""


def join_probe_phrases(probes: list[dict[str, Any]], *, include_values: bool) -> str:
    parts = []
    seen: set[str] = set()
    for probe in probes:
        direction = probe_direction_text(probe, include_values=include_values)
        if not include_values and direction in seen:
            continue
        seen.add(direction)
        parts.append(direction)
    return "; ".join(parts) if parts else "none"


def qualitative_readout(pred: Prediction, *, minimize: bool, brief: bool) -> str:
    quality = "quality-favored" if brief else ("lower predicted score" if minimize else "higher predicted score")
    uncertainty = "uncertain" if brief else ("higher uncertainty" if float(pred.std) > 0 else "low uncertainty")
    ei = "EI-positive" if brief else ("some expected-improvement signal" if float(pred.ei) > 0 else "little expected-improvement signal")
    return f"{quality}, {uncertainty}, {ei}"


def score_with_surrogate(
    engine: SearchEngine,
    state: SearchState,
    surrogate: "GPSurrogate",
    args: argparse.Namespace,
    *,
    iteration: int,
) -> None:
    if state.status == "generation_error":
        state.metrics["surrogate_score"] = engine.config.failure_score
        write_state_update(engine, state)
        return
    operation_schema = getattr(engine, "operation_schema", None) or getattr(args, "operation_schema_object", None)
    features = featurize_for_surrogate(state.train_path, args, operation_schema)
    feature_version = feature_version_for_args(args, operation_schema)
    pred = surrogate.predict(features.vector, mode=args.surrogate_mode, beta=args.gp_beta, xi=args.gp_xi)
    state.metrics.update(
        {
            "model_based_iteration": iteration,
            "feature_version": feature_version,
            "feature_source_hash": features.source_hash,
            "surrogate_mode": args.surrogate_mode,
            "surrogate_pred": float(pred.mean),
            "surrogate_std": float(pred.std),
            "surrogate_ei": float(pred.ei),
            "surrogate_lcb": float(pred.lcb),
            "surrogate_score": float(pred.selection_score),
            "surrogate_history_size": len(surrogate.entries),
            "extracted_params": features.params,
        }
    )
    if operation_schema is not None:
        state.metrics["operation_schema_version"] = operation_schema.version
        state.metrics["operation_schema_feature_names"] = list(operation_schema.parameters)
        state.metrics["operation_schema_feature_count"] = len(operation_schema.parameters)
    state.status = "surrogate_scored"
    write_state_update(engine, state)


def surrogate_feature_dim(args: argparse.Namespace, schema: OperationSchema | None) -> int:
    if use_operation_features(args, schema):
        assert schema is not None
        return operation_feature_dim(schema)
    return feature_dim(args.hash_dims)


def feature_version_for_args(args: argparse.Namespace, schema: OperationSchema | None) -> str:
    if use_operation_features(args, schema):
        assert schema is not None
        return operation_feature_version(schema)
    return FEATURE_VERSION


def use_operation_features(args: argparse.Namespace, schema: OperationSchema | None) -> bool:
    return bool(schema is not None and (args.operation_features or args.generator in OPERATION_GENERATORS))


def featurize_for_surrogate(path: Path, args: argparse.Namespace, schema: OperationSchema | None) -> Features:
    if use_operation_features(args, schema):
        assert schema is not None
        return featurize_operation_schema(path, schema)
    return featurize_train_py(path, args.hash_dims)


def featurize_operation_schema(path: Path, schema: OperationSchema) -> Features:
    text = path.read_text(encoding="utf-8", errors="replace")
    source_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    values = extract_top_level_assignment_values(text)
    vector: list[float] = []
    params: dict[str, Any] = {}
    for parameter in schema.parameters.values():
        present = parameter.name in values
        raw_value = values.get(parameter.name)
        if present:
            params[parameter.name] = raw_value
        if parameter.kind == "choice":
            vector.extend(1.0 if present and choice_values_equal(raw_value, choice) else 0.0 for choice in parameter.choices)
            vector.append(1.0 if present else 0.0)
            continue
        value = as_float(raw_value)
        if value is None:
            vector.append(0.0)
            vector.append(0.0)
            continue
        vector.append(normalize_operation_numeric(value, parameter))
        vector.append(1.0)
    return Features(vector=vector, params=params, source_hash=source_hash)


def extract_top_level_assignment_values(text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return values
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = literal_assignment_value(node.value)
            if value is not None:
                values[canonical_name(node.targets[0].id)] = value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = literal_assignment_value(node.value)
            if value is not None:
                values[canonical_name(node.target.id)] = value
    return values


def literal_assignment_value(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        number = literal_number(node)
        if number is None:
            return None
        if abs(number - round(number)) <= 1e-9:
            return int(round(number))
        return float(number)


def make_operation_tool_schema(schema: OperationSchema, max_operations: int) -> dict[str, Any]:
    names = list(schema.parameters)
    return {
        "type": "function",
        "function": {
            "name": "propose_train_operations",
            "description": (
                "Propose active-feature edits to train.py. Only use the allowed parameter names "
                "and value ranges from the active schema. Do not propose arbitrary code patches."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Short summary of the proposed knob changes.",
                    },
                    "operations": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": max(1, int(max_operations)),
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "enum": names,
                                    "description": "Schema parameter name to edit.",
                                },
                                "op": {
                                    "type": "string",
                                    "enum": ["set_numeric", "set_choice"],
                                    "description": "Use set_numeric for int/float parameters and set_choice for choice parameters.",
                                },
                                "value": {
                                    "description": "The new value. It must satisfy the schema for the chosen name.",
                                },
                                "rationale": {
                                    "type": "string",
                                    "description": "One short reason for this operation.",
                                },
                            },
                            "required": ["name", "op", "value"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["operations"],
                "additionalProperties": False,
            },
        },
    }


def make_dynamic_operation_tools(engine: OperationSearchEngine) -> list[dict[str, Any]]:
    tools = [make_operation_tool_schema(engine.operation_schema, engine.max_operations_per_step)]
    if not engine.feature_expansion_available():
        return tools
    inactive_schema = engine.inactive_operation_schema()
    inactive_names = list(inactive_schema.parameters)
    properties: dict[str, Any] = {
        "name": {
            "type": "string",
            "description": "Inactive schema parameter name to activate as a new GP/search feature.",
        },
        "rationale": {
            "type": "string",
            "description": "Short reason this additional feature should now be searchable.",
        },
    }
    if inactive_names:
        properties["name"]["enum"] = inactive_names
    if engine.allow_new_feature_specs:
        properties.update(
            {
                "type": {
                    "type": "string",
                    "enum": ["int", "float", "choice"],
                    "description": "Required only for brand-new feature specs not already in the inactive schema.",
                },
                "min": {
                    "type": "number",
                    "description": "Minimum for int/float brand-new features.",
                },
                "max": {
                    "type": "number",
                    "description": "Maximum for int/float brand-new features.",
                },
                "scale": {
                    "type": "string",
                    "enum": ["linear", "log"],
                    "description": "Numeric scaling for brand-new features.",
                },
                "choices": {
                    "type": "array",
                    "description": "Allowed values for choice brand-new features.",
                    "items": {},
                },
            }
        )
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "propose_operation_feature",
                "description": (
                    "Activate one additional operation feature dimension for later GP scoring. "
                    "Prefer inactive schema parameters. This action does not edit train.py by itself."
                ),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": ["name"],
                    "additionalProperties": bool(engine.allow_new_feature_specs),
                },
            },
        }
    )
    return tools


def describe_ldm_task(
    args: argparse.Namespace,
    full_schema: OperationSchema | None,
    active_schema: OperationSchema | None,
    *,
    effective_method: str | None = None,
) -> LDMTaskSpec:
    operation_mode = use_operation_features(args, active_schema)
    expansion_available = False
    inactive_names: list[str] = []
    max_active = 0
    full_dimension: int | None = None
    if operation_mode:
        assert active_schema is not None
        max_active = int(getattr(args, "max_active_operation_features", 0) or 0)
        if full_schema is not None and max_active <= 0:
            max_active = len(full_schema.parameters)
        full_dimension = None if full_schema is None else operation_feature_dim(full_schema)
        inactive_names = (
            []
            if full_schema is None
            else [name for name in full_schema.parameters if name not in active_schema.parameters]
        )
        expansion_available = (
            bool(getattr(args, "allow_feature_expansion", True))
            and (max_active <= 0 or len(active_schema.parameters) < max_active)
            and (bool(inactive_names) or bool(getattr(args, "allow_new_feature_specs", False)))
        )

    candidate_domain = CandidateDomainSpec(
        name="train_program",
        kind="structured_python_program",
        dimension=None,
        representation="complete train.py program reached through validated edits",
        constraints={
            "edit_mode": "structured_operations" if operation_mode else "code_edit",
            "maximum_operations_per_step": int(getattr(args, "max_operations_per_step", 1)),
        },
        metadata={"train_file": str(args.train_file)},
    )

    if operation_mode:
        assert active_schema is not None
        surrogate = SurrogateSpaceSpec(
            kind="vector",
            representation="normalized numeric and one-hot expansion-schema parameters",
            dimension_policy="evolving" if expansion_available else "fixed",
            dimension=operation_feature_dim(active_schema),
            encoder="tasks.nanogpt.core.workflow.featurize_operation_params",
            version=active_schema.version,
            metadata={
                "active_parameter_names": list(active_schema.parameters),
                "active_parameter_count": len(active_schema.parameters),
                "maximum_active_parameters": max_active,
                "full_parameter_count": None if full_schema is None else len(full_schema.parameters),
                "full_representation_dimension": full_dimension,
                "full_schema_version": None if full_schema is None else full_schema.version,
            },
        )
    else:
        surrogate = SurrogateSpaceSpec(
            kind="vector",
            representation="numeric assignments plus hashed train.py tokens",
            dimension_policy="fixed",
            dimension=feature_dim(int(args.hash_dims)),
            encoder="tasks.nanogpt.core.workflow.featurize_code",
            version=FEATURE_VERSION,
            metadata={"hash_dimensions": int(args.hash_dims)},
        )

    response_spaces: list[ResponseSpaceSpec] = []
    reservoir_expansions: list[ReservoirExpansionSpec] = []
    if active_schema is not None and args.generator in OPERATION_GENERATORS:
        response_spaces.append(
            ResponseSpaceSpec(
                name="train_operations",
                output_kind="tool_call_or_json",
                parser="tasks.nanogpt.ldm_task.procedure.validate_generator_action",
                description="LLM edits active expansion-schema parameters with deterministic application.",
                schema=make_operation_tool_schema(
                    active_schema,
                    int(getattr(args, "max_operations_per_step", 1)),
                )["function"]["parameters"],
                metadata={
                    "tool_name": "propose_train_operations",
                    "active_parameter_names": list(active_schema.parameters),
                },
            )
        )
        reservoir_expansions.append(
            ReservoirExpansionSpec(
                name="structured_train_program_edit",
                action_kind="edit_candidate",
                response_space="train_operations",
                produces_candidates=True,
                description="Apply validated parameter operations to produce a train.py candidate.",
            )
        )
        if expansion_available:
            response_spaces.append(
                ResponseSpaceSpec(
                    name="reservoir_schema_update",
                    output_kind="tool_call_or_json",
                    parser="tasks.nanogpt.ldm_task.procedure.validate_feature_payload",
                    description="LLM activates one parameter for subsequent reservoir expansion.",
                    schema={
                        "type": "object",
                        "required": ["name"],
                        "properties": {
                            "name": {"type": "string", "enum": inactive_names} if inactive_names else {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                    },
                    metadata={
                        "tool_name": "propose_operation_feature",
                        "inactive_parameter_names": inactive_names,
                        "allow_new_parameter_specs": bool(getattr(args, "allow_new_feature_specs", False)),
                    },
                )
            )
            reservoir_expansions.append(
                ReservoirExpansionSpec(
                    name="expansion_schema_update",
                    action_kind="update_expansion_schema",
                    response_space="reservoir_schema_update",
                    produces_candidates=False,
                    description="Expand the parameter schema used by later candidate edits.",
                    parameters={
                        "inactive_parameter_names": inactive_names,
                        "allow_new_parameter_specs": bool(
                            getattr(args, "allow_new_feature_specs", False)
                        ),
                    },
                )
            )
    else:
        response_spaces.append(
            ResponseSpaceSpec(
                name="code_edit",
                output_kind="text",
                parser="tasks.nanogpt.core.search_core.SearchEngine",
                description="Task-local generator proposes code edits that are applied to train.py.",
                metadata={"generator": args.generator},
            )
        )
        reservoir_expansions.append(
            ReservoirExpansionSpec(
                name="train_program_code_edit",
                action_kind="edit_candidate",
                response_space="code_edit",
                produces_candidates=True,
                description="Apply one task-validated code edit to produce a train.py candidate.",
            )
        )

    method = effective_method or resolve_search_method(args)
    return LDMTaskSpec(
        task="nanogpt",
        candidate_domain=candidate_domain,
        objectives=(
            ObjectiveSpec(
                name=str(args.score_key),
                direction="maximize" if bool(args.maximize) else "minimize",
                description="Metric extracted from train.py evaluation output.",
            ),
        ),
        response_spaces=tuple(response_spaces),
        acquisition=AcquisitionSpec(
            name=str(args.surrogate_mode),
            objective_names=(str(args.score_key),),
            score_direction="maximize",
            selection_rule=(
                f"inner {method} surrogate-scored search; select from {args.select_from}; "
                "maximize shared acquisition score (negated only for legacy queue ordering)"
            ),
            parameters={
                "gp_beta": float(args.gp_beta),
                "gp_xi": float(args.gp_xi),
                "gp_lengthscale": float(args.gp_lengthscale),
                "gp_noise": float(args.gp_noise),
                "breadth": int(args.breadth),
                "depth": int(args.depth),
                "beam_width": int(args.beam_width),
            },
        ),
        reservoir=ReservoirSpec(
            name="train_program_candidate_reservoir",
            expansions=tuple(reservoir_expansions),
            candidate_validator="train.py syntax, edit, operation-schema, and evaluation-contract checks",
            deduplication_key="normalized train.py source hash",
            metadata={
                "expansion_schema_policy": (
                    "evolving" if expansion_available else "fixed"
                )
            },
        ),
        surrogate=surrogate,
        proposal_search=ProposalSearchSpec(
            name=method,
            breadth=max(1, int(args.breadth)),
            depth=max(1, int(args.depth)),
            beam_width=max(1, int(args.beam_width) if int(args.beam_width) > 0 else int(args.breadth)),
            evaluation_policy="surrogate_each_depth_then_real_selected_candidate",
            parameters={"select_from": str(args.select_from)},
        ),
        metadata={
            "generator": args.generator,
            "method": method,
            "iterations": int(args.iterations),
            "warmup": int(args.warmup),
            "seed_policy": args.seed_policy,
        },
    )


def build_operation_prompt(
    engine: OperationSearchEngine,
    parent: SearchState,
    current_text: str,
    *,
    child_index: int,
    sibling_count: int,
    search_note: str,
    edit_index: int,
    total_edits: int,
    prior_operations: list[dict[str, Any]],
) -> str:
    schema = engine.operation_schema
    train_text = current_text
    if len(train_text) > engine.config.prompt_max_chars:
        keep = engine.config.prompt_max_chars // 2
        train_text = (
            train_text[:keep]
            + "\n\n# ... train.py truncated in the prompt middle ...\n\n"
            + train_text[-keep:]
        )
    current_values = extract_top_level_assignment_values(current_text)
    schema_values = {
        name: current_values.get(name, "<missing>")
        for name in schema.parameters
    }
    schema_lines = []
    for parameter in schema.parameters.values():
        if parameter.kind == "choice":
            schema_lines.append(f"- {parameter.name}: choice in {list(parameter.choices)!r}; operation=set_choice")
        else:
            schema_lines.append(
                f"- {parameter.name}: {parameter.kind} in [{parameter.min_value}, {parameter.max_value}] "
                f"scale={parameter.scale}; operation=set_numeric"
            )
    inactive_schema = engine.inactive_operation_schema()
    inactive_lines = []
    for parameter in inactive_schema.parameters.values():
        current_value = current_values.get(parameter.name, "<missing>")
        if parameter.kind == "choice":
            inactive_lines.append(
                f"- {parameter.name}: choice in {list(parameter.choices)!r}; current={current_value!r}"
            )
        else:
            inactive_lines.append(
                f"- {parameter.name}: {parameter.kind} in [{parameter.min_value}, {parameter.max_value}] "
                f"scale={parameter.scale}; current={current_value!r}"
            )
    if not inactive_lines:
        inactive_lines.append("- No inactive schema features remain.")
    history_lines = []
    for state in engine.ranked_states()[:8]:
        direction = "lower is better" if engine.config.minimize else "higher is better"
        history_lines.append(
            f"- {state.state_id}: {engine.config.score_key}={state.score} ({direction}); "
            f"status={state.status}; note={state.description or 'n/a'}"
        )
    history = "\n".join(history_lines) if history_lines else "- No real evaluated candidates yet."
    prior_text = json.dumps(prior_operations, indent=2, sort_keys=True) if prior_operations else "[]"
    extra_instruction = (
        "\nAdditional instruction from the caller:\n"
        + engine.config.extra_instruction.strip()
        + "\n"
        if engine.config.extra_instruction.strip()
        else ""
    )
    feedback_context = (
        "\nFeedback from previous iterations:\n"
        + engine.config.feedback_context.strip()
        + "\n"
        if engine.config.feedback_context.strip()
        else ""
    )
    acquisition_context = str(getattr(engine.config, "acquisition_context", "") or "").strip()
    acquisition_context = (
        "\nAcquisition-decomposition guidance from the GP:\n"
        + acquisition_context
        + "\n"
        if acquisition_context
        else ""
    )
    task_context = engine.config.task_context.strip()
    task_context_block = "\nProject and benchmark context:\n" + task_context + "\n" if task_context else ""
    expansion_instruction = (
        "- Or update the reservoir expansion schema: call `propose_operation_feature` to activate one inactive expansion parameter. "
        "Use this when the current active schema is too narrow for the next useful search move.\n"
        if engine.feature_expansion_available()
        else "- Reservoir expansion schema updates are unavailable; choose only active-parameter edits.\n"
    )
    return f"""We are doing dynamically expanded model-based search over `train.py`.

Search state:
- Parent state: {parent.state_id}
- Parent depth: {parent.depth}
- Parent {engine.config.score_key}: {"unknown" if parent.score is None else f"{parent.score:.8g}"}
- Candidate among siblings: {child_index}/{sibling_count}
- Operation pass in this transition: {edit_index}/{total_edits}
- Search note: {search_note or "propose one useful fixed-operation child"}
{task_context_block}

Objective:
- Improve `{engine.config.score_key}` after executing the script.
- The metric is lower-is-better unless the run says otherwise.
- You may edit only active top-level assignments whose names appear in the active schema below.
- Values must stay inside the active schema range or choice set.
- The GP surrogate representation currently encodes only active expansion parameters. Activating an inactive parameter changes that representation for later candidates.

Reservoir expansion schema version: {schema.version}
Active expansion parameter count: {len(schema.parameters)}
Active expansion parameters: {", ".join(schema.parameters)}
{schema.description}

Active edit operations:
{chr(10).join(schema_lines)}

Inactive expansion parameters available for schema updates:
{chr(10).join(inactive_lines)}

Current active schema values:
```json
{json.dumps(schema_values, indent=2, sort_keys=True)}
```

Prior operations already applied inside this child state:
```json
{prior_text}
```

Recent real evaluated states:
{history}
{feedback_context}
{acquisition_context}
{extra_instruction}
Return format:
- Choose exactly one action: either use the current reservoir expansion schema or update it.
- To use the current schema and edit train.py, call `propose_train_operations` with 1 to {engine.max_operations_per_step} operations.
{expansion_instruction.rstrip()}
- Use `set_numeric` for int/float schema parameters and `set_choice` for choice parameters.
- Do not output SEARCH/REPLACE blocks or unified diffs; the runner will apply valid operations deterministically.

Current parent `train.py`:
```python
{train_text}
```
"""


def compact_operation_message_for_log(message: dict[str, Any]) -> dict[str, Any]:
    compact = {"role": message.get("role", "assistant")}
    content = message.get("content")
    if content:
        compact["content"] = str(content)[:2000]
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        compact["tool_calls"] = []
        for tool_call in tool_calls:
            function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            arguments = function.get("arguments") if isinstance(function, dict) else None
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw_arguments": arguments[:2000]}
            compact["tool_calls"].append(
                {
                    "name": function.get("name") if isinstance(function, dict) else "",
                    "arguments": arguments,
                }
            )
    return compact


OPERATION_TOOL_NAMES = {"propose_train_operations", "propose_operation_feature"}


def extract_operation_json_from_text(text: str) -> dict[str, Any] | None:
    _tool_name, payload = extract_operation_tool_payload_from_text(text)
    if isinstance(payload, dict):
        return payload
    return None


def extract_dynamic_operation_json_from_text(text: str) -> tuple[str, dict[str, Any] | None]:
    tool_name, payload = extract_operation_tool_payload_from_text(text)
    if not isinstance(payload, dict):
        return "", None
    if tool_name in OPERATION_TOOL_NAMES:
        return tool_name, payload
    if "operations" in payload:
        return "propose_train_operations", payload
    if "feature" in payload or "name" in payload or "parameter" in payload:
        return "propose_operation_feature", payload
    return "", payload


def extract_operation_tool_payload_from_text(text: str, *, _depth: int = 0) -> tuple[str, dict[str, Any] | None]:
    if not isinstance(text, str) or not text.strip() or _depth > 6:
        return "", None
    for loaded in operation_json_candidates(text):
        tool_name, payload = extract_operation_tool_payload_from_json(loaded, _depth=_depth + 1)
        if isinstance(payload, dict):
            return tool_name, payload
    return "", None


def operation_json_candidates(text: str) -> list[Any]:
    candidates: list[str] = []
    candidates.append(text.strip())
    candidates.extend(candidate.strip() for candidate in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE))
    for tag in ("tool_calls", "operation_tool_transcript"):
        greedy = re.search(rf"<{tag}>\s*(.*)\s*</{tag}>", text, flags=re.DOTALL)
        if greedy:
            candidates.append(greedy.group(1).strip())
        candidates.extend(
            candidate.strip()
            for candidate in re.findall(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.DOTALL)
        )

    loaded: list[Any] = []
    seen_candidates: set[str] = set()
    decoder = json.JSONDecoder()
    for candidate in candidates:
        if not candidate or candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)
        try:
            loaded.append(json.loads(candidate))
        except json.JSONDecodeError:
            pass

    # Last resort: scan embedded JSON values. This handles text wrappers such as
    # <operation_tool_transcript>[...]</operation_tool_transcript> and partial prose.
    seen_spans: set[tuple[int, int]] = set()
    for match in re.finditer(r"[\[{]", text):
        start = match.start()
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        span = (start, start + end)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        loaded.append(value)
    return loaded


def extract_operation_tool_payload_from_json(value: Any, *, _depth: int = 0) -> tuple[str, dict[str, Any] | None]:
    if _depth > 6:
        return "", None
    if isinstance(value, list):
        for item in value:
            tool_name, payload = unpack_plain_operation_tool_call(item)
            normalized_payload = normalize_operation_alias_payload(payload)
            if tool_name in OPERATION_TOOL_NAMES and isinstance(normalized_payload, dict):
                return tool_name, normalized_payload
            if tool_name == "edit_train_py" and isinstance(normalized_payload, dict):
                return "propose_train_operations", normalized_payload
        for item in value:
            tool_name, payload = extract_operation_tool_payload_from_json(item, _depth=_depth + 1)
            if isinstance(payload, dict):
                return tool_name, payload
        return "", None
    if not isinstance(value, dict):
        return "", None

    tool_name, payload = unpack_plain_operation_tool_call(value)
    normalized_payload = normalize_operation_alias_payload(payload)
    if tool_name in OPERATION_TOOL_NAMES and isinstance(normalized_payload, dict):
        return tool_name, normalized_payload
    if tool_name == "edit_train_py" and isinstance(normalized_payload, dict):
        return "propose_train_operations", normalized_payload
    normalized_value = normalize_operation_alias_payload(value)
    if isinstance(normalized_value, dict):
        return "propose_train_operations", normalized_value
    if isinstance(value.get("tool_calls"), list):
        tool_name, payload = extract_operation_tool_payload_from_json(value["tool_calls"], _depth=_depth + 1)
        if isinstance(payload, dict):
            return tool_name, payload

    content = value.get("content")
    if isinstance(content, str):
        tool_name, payload = extract_operation_tool_payload_from_text(content, _depth=_depth + 1)
        if isinstance(payload, dict):
            return tool_name, payload
    elif isinstance(content, list):
        tool_name, payload = extract_operation_tool_payload_from_json(content, _depth=_depth + 1)
        if isinstance(payload, dict):
            return tool_name, payload

    for key in ("message", "delta", "choices", "messages"):
        if key in value:
            tool_name, payload = extract_operation_tool_payload_from_json(value[key], _depth=_depth + 1)
            if isinstance(payload, dict):
                return tool_name, payload

    if "feature" in value or "parameter" in value or (
        "name" in value and "op" not in value and "role" not in value and "tool_calls" not in value
    ):
        return "", value
    return "", None


def normalize_operation_alias_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("operations"), list):
        return payload
    edits = payload.get("edits")
    if not isinstance(edits, list) or not edits:
        return None
    for edit in edits:
        if not isinstance(edit, dict) or not {"name", "op", "value"}.issubset(edit):
            return None
    normalized = dict(payload)
    normalized["operations"] = edits
    return normalized


def unpack_plain_operation_tool_call(tool_call: Any) -> tuple[str, dict[str, Any] | None]:
    if not isinstance(tool_call, dict):
        return "", None
    function = tool_call.get("function")
    name = tool_call.get("name")
    # Some local/plain-text models emit `tool_name` instead of the OpenAI
    # `name`/`function.name` key; accept it as an alias so their proposals
    # parse instead of failing with "operation 1 uses unknown parameter ''".
    if not name and isinstance(tool_call.get("tool_name"), str):
        name = tool_call["tool_name"]
    arguments = tool_call.get("arguments")
    if isinstance(function, dict):
        name = function.get("name") or name
        arguments = function.get("arguments") if function.get("arguments") is not None else arguments
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return str(name or ""), None
    if isinstance(arguments, dict):
        return str(name or ""), arguments
    return str(name or ""), None


def validate_generator_action(
    tool_name: str,
    payload: Any,
    engine: OperationSearchEngine,
    *,
    current_text: str,
) -> GeneratorAction:
    if tool_name == "propose_operation_feature":
        parameter, rationale = validate_feature_payload(payload, engine, current_text=current_text)
        return GeneratorAction(
            kind="feature",
            feature=parameter,
            rationale=rationale,
            source="propose_operation_feature",
        )
    if tool_name in {"", "propose_train_operations"}:
        proposal = validate_operation_payload(
            payload,
            engine.operation_schema,
            max_operations=engine.max_operations_per_step,
        )
        return GeneratorAction(kind="operations", operations=proposal, source="propose_train_operations")
    raise ValueError(f"unknown tool {tool_name!r}; expected propose_train_operations or propose_operation_feature.")


def validate_feature_payload(
    payload: Any,
    engine: OperationSearchEngine,
    *,
    current_text: str,
) -> tuple[OperationParameter, str]:
    if not isinstance(payload, dict):
        raise ValueError("expansion-schema update payload must be an object.")
    raw_feature = payload.get("feature") if isinstance(payload.get("feature"), dict) else payload
    if not isinstance(raw_feature, dict):
        raise ValueError("expansion-schema update must include a parameter object or fields.")
    name = canonical_name(str(raw_feature.get("name") or raw_feature.get("parameter") or payload.get("name") or ""))
    if not name:
        raise ValueError("expansion-schema update must include a non-empty parameter name.")
    if name in engine.operation_schema.parameters:
        raise ValueError(f"expansion parameter {name} is already active.")
    if len(engine.operation_schema.parameters) >= engine.max_active_operation_features:
        raise ValueError("active expansion-parameter limit has already been reached.")
    inactive = engine.inactive_operation_schema().parameters
    rationale = str(raw_feature.get("rationale") or payload.get("rationale") or "").strip()
    if name in inactive:
        return inactive[name], rationale
    if not engine.allow_new_feature_specs:
        raise ValueError(f"parameter {name} is not in the inactive expansion schema.")
    parameter = operation_parameter_from_payload(raw_feature)
    values = extract_top_level_assignment_values(current_text)
    if parameter.name not in values:
        raise ValueError(
            f"new expansion parameter {parameter.name} is not a literal top-level assignment in current train.py."
        )
    current_value = values.get(parameter.name)
    if parameter.kind == "choice":
        validate_operation_value(current_value, parameter, index=1)
    else:
        value = as_float(current_value)
        if value is None:
            raise ValueError(f"new numeric feature {parameter.name} has no numeric current value.")
        validate_operation_value(value, parameter, index=1)
    return parameter, rationale


def apply_operations_to_train_text(
    text: str,
    operations: list[ValidatedOperation],
    schema: OperationSchema,
) -> OperationApplyResult:
    line_matches = find_top_level_assignment_lines(text, schema)
    lines = text.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    for operation in operations:
        if operation.name not in line_matches:
            raise ValueError(f"Cannot apply {operation.name}: no unique top-level assignment found.")
        index, old_value, prefix, suffix = line_matches[operation.name]
        new_value_text = format_operation_value(operation.value, schema.parameters[operation.name])
        newline = "\n" if lines[index].endswith("\n") else ""
        new_line = f"{prefix}{new_value_text}{suffix}{newline}"
        if new_line == lines[index]:
            raise ValueError(f"Operation for {operation.name} produced no change.")
        lines[index] = new_line
        records.append(
            {
                "name": operation.name,
                "op": operation.op,
                "old_value": old_value,
                "new_value": operation.value,
                "rationale": operation.rationale,
                "line": index + 1,
            }
        )
    new_text = "".join(lines)
    ast.parse(new_text)
    patch = "".join(
        difflib.unified_diff(
            text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile="parent/train.py",
            tofile="child/train.py",
        )
    )
    if not patch:
        raise ValueError("Operation application produced no diff.")
    return OperationApplyResult(text=new_text, patch=patch, records=records)


def find_top_level_assignment_lines(
    text: str,
    schema: OperationSchema,
) -> dict[str, tuple[int, Any, str, str]]:
    tree = ast.parse(text)
    line_by_name: dict[str, tuple[int, Any]] = {}
    duplicate_names: set[str] = set()
    for node in tree.body:
        target_name: str | None = None
        value_node: ast.AST | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = canonical_name(node.targets[0].id)
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = canonical_name(node.target.id)
            value_node = node.value
        if target_name is None or target_name not in schema.parameters:
            continue
        if target_name in line_by_name:
            duplicate_names.add(target_name)
            continue
        old_value = literal_assignment_value(value_node)
        line_by_name[target_name] = (int(node.lineno) - 1, old_value)
    if duplicate_names:
        raise ValueError(f"Duplicate top-level assignments for schema parameters: {sorted(duplicate_names)}")

    lines = text.splitlines(keepends=True)
    result: dict[str, tuple[int, Any, str, str]] = {}
    for name, (line_index, old_value) in line_by_name.items():
        line = lines[line_index].rstrip("\n")
        match = re.match(rf"^(\s*{re.escape(name)}\s*(?::[^=]+)?=\s*)(.*?)(\s*(?:#.*)?)$", line)
        if not match:
            raise ValueError(f"Cannot safely rewrite assignment line for {name}: {line!r}")
        result[name] = (line_index, old_value, match.group(1), match.group(3))
    return result


def format_operation_value(value: Any, parameter: OperationParameter) -> str:
    if parameter.kind == "choice":
        return json.dumps(value, ensure_ascii=False)
    if parameter.kind == "int":
        return str(int(value))
    return repr(float(value))


def operation_summary(operations: list[ValidatedOperation], records: list[dict[str, Any]]) -> str:
    by_name = {record.get("name"): record for record in records}
    parts = []
    for operation in operations:
        record = by_name.get(operation.name, {})
        old_value = record.get("old_value")
        new_value = record.get("new_value", operation.value)
        parts.append(f"set {operation.name} {old_value!r} -> {new_value!r}")
    return "; ".join(parts)


def feedback_action_summary(state: SearchState) -> str:
    operations = state.metrics.get("operations")
    if isinstance(operations, list) and operations:
        parts = []
        for operation in operations:
            if not isinstance(operation, dict):
                continue
            name = operation.get("name")
            old_value = operation.get("old_value")
            new_value = operation.get("new_value")
            if name:
                parts.append(f"set {name} {old_value!r} -> {new_value!r}")
        if parts:
            return "; ".join(parts)
    if state.description.strip():
        return state.description.strip()
    descriptions = [
        str(edit.get("description") or "").strip()
        for edit in state.edits
        if isinstance(edit, dict) and str(edit.get("description") or "").strip()
    ]
    if descriptions:
        return "; ".join(descriptions)
    return "seed train.py" if state.parent_id is None else "no action summary"


def feedback_path_summary(state: SearchState) -> str:
    if not state.edits:
        return feedback_action_summary(state)
    parts = []
    for edit in state.edits:
        if not isinstance(edit, dict):
            continue
        description = str(edit.get("description") or "").strip()
        if description:
            parts.append(description)
    return " -> ".join(parts) if parts else feedback_action_summary(state)


def format_tsv_float(value: Any) -> str:
    number = as_float(value)
    return "" if number is None else f"{number:.10g}"


def clean_tsv_text(value: Any) -> str:
    text = str(value)
    return re.sub(r"[\t\r\n]+", " ", text).strip()


def make_mock_operation_proposal(
    text: str,
    schema: OperationSchema,
    max_operations: int,
    counter: int,
) -> list[ValidatedOperation]:
    current = extract_top_level_assignment_values(text)
    targets = mock_operation_targets(schema)
    candidates: list[ValidatedOperation] = []
    names = list(schema.parameters)
    if names:
        offset = counter % len(names)
        names = names[offset:] + names[:offset]
    for name in names:
        if name not in targets or name not in schema.parameters:
            continue
        parameter = schema.parameters[name]
        target_value = targets[name]
        if parameter.kind == "choice":
            try:
                value = validate_operation_value(target_value, parameter, index=1)
            except ValueError:
                continue
        elif parameter.kind == "int":
            value = int(round(float(target_value)))
            value = max(int(parameter.min_value), min(int(parameter.max_value), value))
        else:
            value = float(target_value)
            value = max(float(parameter.min_value), min(float(parameter.max_value), value))
        if current.get(name) is not None and choice_values_equal(current.get(name), value):
            continue
        op = "set_choice" if parameter.kind == "choice" else "set_numeric"
        candidates.append(
            ValidatedOperation(
                name=name,
                op=op,
                value=value,
                rationale="Move this fixed schema knob toward the mock target.",
            )
        )
        if len(candidates) >= max(1, int(max_operations)):
            break
    if candidates:
        return candidates
    for name, parameter in schema.parameters.items():
        current_value = current.get(name)
        value = fallback_operation_value(current_value, parameter)
        if current_value is not None and choice_values_equal(current_value, value):
            continue
        return [
            ValidatedOperation(
                name=name,
                op="set_choice" if parameter.kind == "choice" else "set_numeric",
                value=value,
                rationale="Fallback valid schema operation.",
            )
        ]
    raise ValueError("operation_mock could not find any valid operation to apply.")


def make_mock_generator_action(
    text: str,
    engine: OperationSearchEngine,
    counter: int,
) -> GeneratorAction:
    expand_every = max(0, int(getattr(engine.args, "mock_expand_every", 0) or 0))
    if expand_every > 0 and counter > 0 and counter % expand_every == 0 and engine.feature_expansion_available():
        inactive = list(engine.inactive_operation_schema().parameters.values())
        if inactive:
            parameter = inactive[(counter // expand_every - 1) % len(inactive)]
            return GeneratorAction(
                kind="feature",
                feature=parameter,
                rationale="Mock reservoir expansion schema update.",
                source="operation_mock",
            )
    operations = make_mock_operation_proposal(
        text,
        engine.operation_schema,
        engine.max_operations_per_step,
        counter,
    )
    return GeneratorAction(
        kind="operations",
        operations=operations,
        source="operation_mock",
    )


def resolve_warmup_strategy(args: argparse.Namespace, engine: SearchEngine) -> str:
    strategy = str(args.warmup_strategy)
    if strategy == "auto":
        return "random_operation" if isinstance(engine, OperationSearchEngine) else "agent"
    return strategy


def create_random_operation_warmup_state(
    engine: OperationSearchEngine,
    root: SearchState,
    rng: random.Random,
    *,
    warmup_index: int,
    total: int,
) -> SearchState:
    state = engine._new_state(parent=root, depth=1)
    current_text = root.train_path.read_text(encoding="utf-8")
    current_values = extract_top_level_assignment_values(current_text)
    operations = sample_random_operations(
        engine.operation_schema,
        current_values,
        max_operations=engine.max_operations_per_step,
        rng=rng,
    )
    apply_result = apply_operations_to_train_text(current_text, operations, engine.operation_schema)
    prompt = build_random_warmup_prompt(engine, root, current_values, warmup_index=warmup_index, total=total)
    state.prompt_path = state.workdir / "prompt.md"
    state.prompt_path.write_text(prompt, encoding="utf-8")
    response_payload = {
        "summary": operation_summary(operations, apply_result.records),
        "warmup_strategy": "random_operation",
        "schema_version": engine.operation_schema.version,
        "operations": [
            {
                "name": operation.name,
                "op": operation.op,
                "value": operation.value,
                "rationale": operation.rationale,
            }
            for operation in operations
        ],
    }
    response = json.dumps(response_payload, indent=2, sort_keys=True)
    state.response_path = state.workdir / "response.md"
    state.response_path.write_text(response, encoding="utf-8")
    state.llm_response, state.llm_response_truncated = engine._inline_llm_response(response)
    state.patch_path = state.workdir / "patch.diff"
    state.patch_path.write_text(apply_result.patch, encoding="utf-8")
    operations_payload = {
        "summary": operation_summary(operations, apply_result.records),
        "schema_version": engine.operation_schema.version,
        "max_operations_per_step": engine.max_operations_per_step,
        "warmup_strategy": "random_operation",
        "operations": response_payload["operations"],
        "applied": apply_result.records,
        "validation_log": [{"attempt": 1, "status": "accepted", "source": "random_operation_warmup"}],
    }
    (state.workdir / "operations.json").write_text(
        json.dumps(operations_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    state.train_path.write_text(apply_result.text, encoding="utf-8")
    state.description = operation_summary(operations, apply_result.records)[:300]
    state.edits.append(
        {
            "edit_index": 1,
            "total_edits": 1,
            "description": state.description,
            "prompt_path": str(state.prompt_path),
            "response_path": str(state.response_path),
            "patch_path": str(state.patch_path),
            "operations_path": str(state.workdir / "operations.json"),
            "llm_response": state.llm_response,
            "llm_response_truncated": state.llm_response_truncated,
            "token_usage": 0,
            "token_usage_detail": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "edit_source": "random_operation_warmup",
            "detected_edit_count": len(operations),
            "operations": response_payload["operations"],
            "applied_operations": apply_result.records,
            "status": "applied",
        }
    )
    state.metrics.update(
        {
            "warmup_index": warmup_index,
            "warmup_strategy": "random_operation",
            "operation_schema_version": engine.operation_schema.version,
            "operation_schema_feature_names": list(engine.operation_schema.parameters),
            "operation_schema_feature_count": len(engine.operation_schema.parameters),
            "operations": apply_result.records,
        }
    )
    state.status = "generated"
    engine._write_state_meta(state)
    engine._record_manifest(state)
    return state


def sample_random_operations(
    schema: OperationSchema,
    current_values: dict[str, Any],
    *,
    max_operations: int,
    rng: random.Random,
) -> list[ValidatedOperation]:
    parameters = list(schema.parameters.values())
    rng.shuffle(parameters)
    operation_count = rng.randint(1, max(1, min(max_operations, len(parameters))))
    operations: list[ValidatedOperation] = []
    for parameter in parameters:
        value = sample_random_parameter_value(parameter, current_values.get(parameter.name), rng)
        if current_values.get(parameter.name) is not None and choice_values_equal(current_values.get(parameter.name), value):
            continue
        operations.append(
            ValidatedOperation(
                name=parameter.name,
                op="set_choice" if parameter.kind == "choice" else "set_numeric",
                value=value,
                rationale="Random warm-up sample for GP training.",
            )
        )
        if len(operations) >= operation_count:
            break
    if operations:
        return operations
    for parameter in parameters:
        value = fallback_operation_value(current_values.get(parameter.name), parameter)
        if current_values.get(parameter.name) is None or not choice_values_equal(current_values.get(parameter.name), value):
            return [
                ValidatedOperation(
                    name=parameter.name,
                    op="set_choice" if parameter.kind == "choice" else "set_numeric",
                    value=value,
                    rationale="Fallback warm-up sample for GP training.",
                )
            ]
    raise ValueError("Could not sample a non-noop warm-up operation from schema.")


def sample_random_parameter_value(parameter: OperationParameter, current_value: Any, rng: random.Random) -> Any:
    if parameter.kind == "choice":
        choices = list(parameter.choices)
        rng.shuffle(choices)
        for choice in choices:
            if current_value is None or not choice_values_equal(current_value, choice):
                return choice
        return choices[0]
    lo = float(parameter.min_value)
    hi = float(parameter.max_value)
    if parameter.kind == "int":
        low = int(math.ceil(lo))
        high = int(math.floor(hi))
        if high <= low:
            return low
        for _attempt in range(20):
            value = rng.randint(low, high)
            if current_value is None or not choice_values_equal(current_value, value):
                return value
        return low if not choice_values_equal(current_value, low) else high
    if parameter.scale == "log":
        value = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    else:
        value = rng.uniform(lo, hi)
    return float(value)


def build_random_warmup_prompt(
    engine: OperationSearchEngine,
    root: SearchState,
    current_values: dict[str, Any],
    *,
    warmup_index: int,
    total: int,
) -> str:
    schema_lines = []
    for parameter in engine.operation_schema.parameters.values():
        if parameter.kind == "choice":
            schema_lines.append(f"- {parameter.name}: choice in {list(parameter.choices)!r}")
        else:
            schema_lines.append(
                f"- {parameter.name}: {parameter.kind} in [{parameter.min_value}, {parameter.max_value}], "
                f"scale={parameter.scale}"
            )
    schema_values = {
        name: current_values.get(name, "<missing>")
        for name in engine.operation_schema.parameters
    }
    return f"""Random fixed-operation warm-up sample.

Root state: {root.state_id}
Warm-up candidate: {warmup_index}/{total}
Schema version: {engine.operation_schema.version}

The runner sampled valid schema operations directly to collect real evaluated data for GP learning.

Allowed schema:
{chr(10).join(schema_lines)}

Root schema values:
```json
{json.dumps(schema_values, indent=2, sort_keys=True)}
```
"""


def mock_operation_targets(schema: OperationSchema) -> dict[str, Any]:
    names = set(schema.parameters)
    mock_targets = {
        "DEPTH": 10,
        "WIDTH": 704,
        "MATRIX_LR": 0.022,
        "EMBEDDING_LR": 0.55,
        "WEIGHT_DECAY": 0.16,
        "VALUE_GATE_CHANNELS": 64,
        "WARMDOWN_RATIO": 0.45,
        "NGRAM_SCALE": 0.18,
    }
    real_targets = {
        "ASPECT_RATIO": 64,
        "HEAD_DIM": 128,
        "WINDOW_PATTERN": "SSLL",
        "TOTAL_BATCH_SIZE": 524288,
        "EMBEDDING_LR": 0.55,
        "UNEMBEDDING_LR": 0.004,
        "MATRIX_LR": 0.035,
        "SCALAR_LR": 0.45,
        "WEIGHT_DECAY": 0.15,
        "WARMUP_RATIO": 0.02,
        "WARMDOWN_RATIO": 0.45,
        "FINAL_LR_FRAC": 0.02,
        "DEPTH": 6,
        "DEVICE_BATCH_SIZE": 128,
    }
    targets = mock_targets if "WIDTH" in names else real_targets
    return {name: value for name, value in targets.items() if name in names}


def fallback_operation_value(current_value: Any, parameter: OperationParameter) -> Any:
    if parameter.kind == "choice":
        for choice in parameter.choices:
            if not choice_values_equal(current_value, choice):
                return choice
        return parameter.choices[0]
    lo = float(parameter.min_value)
    hi = float(parameter.max_value)
    if parameter.kind == "int":
        current = int(round(as_float(current_value) or lo))
        candidate = current + 1 if current < int(hi) else current - 1
        return max(int(lo), min(int(hi), candidate))
    current = as_float(current_value)
    if current is None:
        return (lo + hi) / 2.0
    if parameter.scale == "log":
        midpoint = math.exp((math.log(lo) + math.log(hi)) / 2.0)
    else:
        midpoint = (lo + hi) / 2.0
    if abs(float(current) - midpoint) > 1e-12:
        return midpoint
    return lo if abs(midpoint - lo) > abs(midpoint - hi) else hi


class GPSurrogate:
    def __init__(
        self,
        entries: list[BufferEntry],
        *,
        lengthscale: float,
        noise: float,
        prior_score: float,
        prior_std: float,
        minimize: bool,
    ):
        self.entries = entries
        self.lengthscale = max(1e-6, float(lengthscale))
        self.noise = max(1e-9, float(noise))
        self.prior_score = float(prior_score)
        self.prior_std = max(1e-9, float(prior_std))
        self.minimize = minimize
        self.ready = False
        self.fit_status = "prior" if not entries else "fallback"
        self.fit_error: str | None = None
        self.fit_metrics: dict[str, Any] = {}
        self._fit()

    def _fit(self) -> None:
        np = require_numpy()
        self.ready = False
        self.fit_status = "prior" if not self.entries else "fallback"
        self.fit_error = None
        self.fit_metrics = {}
        if len(self.entries) < 2:
            scores = [entry.score for entry in self.entries]
            self.fit_metrics = {
                "history_size": len(self.entries),
                "fit_status": self.fit_status,
                "ready": False,
                "best_observed": best_observed(scores, minimize=self.minimize),
                "mean_observed": None if not scores else float(np.mean(scores)),
                "std_observed": None if not scores else float(np.std(scores)),
            }
            return
        self.X = np.array([entry.feature_vector for entry in self.entries], dtype=float)
        self.y = np.array([entry.score for entry in self.entries], dtype=float)
        self.x_mean = self.X.mean(axis=0)
        self.x_std = self.X.std(axis=0) + 1e-8
        Xz = (self.X - self.x_mean) / self.x_std
        self.y_mean = float(self.y.mean())
        self.y_std = float(self.y.std() + 1e-9)
        yz = (self.y - self.y_mean) / self.y_std
        K = rbf_kernel(Xz, Xz, self.lengthscale) + self.noise * np.eye(len(Xz))
        try:
            self.L = np.linalg.cholesky(K + 1e-8 * np.eye(len(Xz)))
            self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, yz))
            self.Xz = Xz
            fitted_z = K @ self.alpha
            residual_z = yz - fitted_z
            nlml_z = 0.5 * float(yz @ self.alpha)
            nlml_z += float(np.log(np.diag(self.L)).sum())
            nlml_z += 0.5 * len(yz) * math.log(2.0 * math.pi)
            self.ready = True
            self.fit_status = "fitted"
            self.fit_metrics = {
                "history_size": len(self.entries),
                "fit_status": self.fit_status,
                "ready": True,
                "best_observed": best_observed(self.y.tolist(), minimize=self.minimize),
                "mean_observed": float(self.y.mean()),
                "std_observed": float(self.y.std()),
                "nlml_z": float(nlml_z),
                "train_rmse": float(math.sqrt(np.mean((residual_z * self.y_std) ** 2))),
                "train_mae": float(np.mean(np.abs(residual_z * self.y_std))),
                "lengthscale": self.lengthscale,
                "noise": self.noise,
            }
        except np.linalg.LinAlgError:
            self.ready = False
            self.fit_status = "fallback"
            self.fit_error = "cholesky_failed"
            self.fit_metrics = {
                "history_size": len(self.entries),
                "fit_status": self.fit_status,
                "ready": False,
                "fit_error": self.fit_error,
                "best_observed": best_observed(self.y.tolist(), minimize=self.minimize),
                "mean_observed": float(self.y.mean()),
                "std_observed": float(self.y.std()),
            }

    def summary(self) -> dict[str, Any]:
        if self.fit_metrics:
            return dict(self.fit_metrics)
        scores = [entry.score for entry in self.entries]
        return {
            "history_size": len(self.entries),
            "fit_status": self.fit_status,
            "ready": self.ready,
            "best_observed": best_observed(scores, minimize=self.minimize),
            "mean_observed": None if not scores else float(np.mean(scores)),
            "std_observed": None if not scores else float(np.std(scores)),
        }

    def predict(self, vector: list[float], *, mode: str, beta: float, xi: float) -> Prediction:
        np = require_numpy()
        if not self.entries:
            mean = self.prior_score
            std = self.prior_std
        elif not self.ready:
            scores = np.array([entry.score for entry in self.entries], dtype=float)
            mean = float(scores.mean())
            std = float(max(scores.std(), self.prior_std))
        else:
            x = np.array(vector, dtype=float)[None, :]
            xz = (x - self.x_mean) / self.x_std
            Ks = rbf_kernel(xz, self.Xz, self.lengthscale)
            mu_z = float(np.asarray(Ks @ self.alpha).ravel()[0])
            V = np.linalg.solve(self.L, Ks.T)
            var_z = max(1.0 - float((V * V).sum()), 1e-9)
            mean = self.y_mean + mu_z * self.y_std
            std = math.sqrt(var_z) * self.y_std

        observed = [entry.score for entry in self.entries]
        best = min(observed) if self.minimize and observed else max(observed) if observed else mean
        acquisition = make_acquisition(
            mode,
            minimize=(self.minimize,),
            beta=beta,
            xi=xi,
        )
        acquisition_score = float(acquisition.score(mean, std, best=best))
        ei = float(
            make_acquisition("ei", minimize=(self.minimize,), xi=xi).score(
                mean,
                std,
                best=best,
            )
        )
        lcb = mean - beta * std if self.minimize else mean + beta * std
        selection_score = -acquisition_score
        return Prediction(mean=mean, std=std, ei=ei, lcb=lcb, selection_score=selection_score)


def featurize_train_py(path: Path, hash_dims: int) -> Features:
    text = path.read_text(encoding="utf-8", errors="replace")
    source_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    params, numeric_values = extract_numeric_assignments(text)
    vector: list[float] = []
    for name in COMMON_NUMERIC_NAMES:
        value = params.get(name)
        vector.append(signed_log1p(value) if value is not None else 0.0)
        vector.append(1.0 if value is not None else 0.0)
    vector.extend(numeric_stats(numeric_values))
    vector.extend(hash_text_features(text, hash_dims))
    return Features(vector=vector, params=params, source_hash=source_hash)


def extract_numeric_assignments(text: str) -> tuple[dict[str, float], list[float]]:
    params: dict[str, float] = {}
    values: list[float] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return params, values

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            numeric_values = literal_numbers(node.value)
            if not numeric_values:
                continue
            values.extend(numeric_values)
            scalar_value = literal_number(node.value)
            for target in node.targets:
                name = assignment_name(target)
                if name:
                    record_param_values(params, canonical_name(name), scalar_value, numeric_values)
        elif isinstance(node, ast.AnnAssign):
            numeric_values = literal_numbers(node.value)
            name = assignment_name(node.target)
            if not numeric_values or not name:
                continue
            values.extend(numeric_values)
            record_param_values(params, canonical_name(name), literal_number(node.value), numeric_values)
    return params, values


def record_param_values(
    params: dict[str, float],
    canonical: str,
    scalar_value: float | None,
    numeric_values: list[float],
) -> None:
    if scalar_value is not None and canonical in COMMON_NUMERIC_NAMES:
        params[canonical] = scalar_value
    if canonical == "ADAM_BETAS":
        for key, value in zip(("ADAM_BETA1", "ADAM_BETA2"), numeric_values):
            params[key] = value


def literal_numbers(node: ast.AST | None) -> list[float]:
    value = literal_number(node)
    if value is not None:
        return [value]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values: list[float] = []
        for elt in node.elts:
            values.extend(literal_numbers(elt))
        return values
    if isinstance(node, ast.Dict):
        values = []
        for item in node.values:
            values.extend(literal_numbers(item))
        return values
    return []


def literal_number(node: ast.AST | None) -> float | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = literal_number(node.operand)
        if value is None:
            return None
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = literal_number(node.left)
        right = literal_number(node.right)
        if left is None or right is None:
            return None
        try:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return None if right == 0 else left / right
            if isinstance(node.op, ast.FloorDiv):
                return None if right == 0 else left // right
            if isinstance(node.op, ast.Mod):
                return None if right == 0 else left % right
            if isinstance(node.op, ast.Pow):
                return left**right
        except (OverflowError, ValueError):
            return None
    return None


def assignment_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def numeric_stats(values: list[float]) -> list[float]:
    if not values:
        return [0.0] * 6
    np = require_numpy()
    arr = np.array(values, dtype=float)
    logged = np.log1p(np.abs(arr))
    return [
        math.log1p(len(values)),
        float(np.mean(logged)),
        float(np.std(logged)),
        signed_log1p(float(np.min(arr))),
        signed_log1p(float(np.max(arr))),
        float(np.mean(arr == 0.0)),
    ]


def hash_text_features(text: str, dims: int) -> list[float]:
    dims = max(0, int(dims))
    if dims == 0:
        return []
    np = require_numpy()
    vec = np.zeros(dims, dtype=float)
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[-+]?\d+(?:\.\d+)?", text)
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8", errors="replace"), digest_size=8).digest()
        raw = int.from_bytes(digest, "little")
        sign = 1.0 if raw & 1 else -1.0
        vec[(raw >> 1) % dims] += sign
    norm = math.sqrt(max(len(tokens), 1))
    return (vec / norm).tolist()


def signed_log1p(value: float | None) -> float:
    if value is None:
        return 0.0
    value = float(value)
    return math.copysign(math.log1p(abs(value)), value)


def load_buffer(
    path: Path,
    expected_dim: int,
    expected_feature_version: str | None = None,
    args: argparse.Namespace | None = None,
) -> list[BufferEntry]:
    if not path.exists():
        return []
    entries: list[BufferEntry] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        entry = parse_buffer_entry(
            data,
            expected_dim,
            expected_feature_version=expected_feature_version,
            args=args,
        )
        if entry is not None:
            entries.append(entry)
    return entries


def load_projected_buffer(
    path: Path,
    args: argparse.Namespace,
    schema: OperationSchema | None,
) -> list[BufferEntry]:
    if not use_operation_features(args, schema):
        return load_buffer(
            path,
            surrogate_feature_dim(args, schema),
            expected_feature_version=feature_version_for_args(args, schema),
            args=args,
        )
    assert schema is not None
    raw_entries = load_raw_buffer(path, args=args)
    return project_buffer_entries(raw_entries, schema, args)


def load_raw_buffer(path: Path, *, args: argparse.Namespace | None = None) -> list[BufferEntry]:
    if not path.exists():
        return []
    entries: list[BufferEntry] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        entry = parse_raw_buffer_entry(data, args=args)
        if entry is not None:
            entries.append(entry)
    return entries


def parse_raw_buffer_entry(data: Any, *, args: argparse.Namespace | None = None) -> BufferEntry | None:
    if not isinstance(data, dict):
        return None
    score = as_float(data.get("score"))
    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    if not should_use_score_for_gp(
        score,
        status=str(metrics.get("status") or data.get("status") or ""),
        metrics=metrics,
        args=args,
    ):
        return None
    vector = data.get("feature_vector")
    try:
        feature_vector = [float(value) for value in vector] if isinstance(vector, list) else []
    except (TypeError, ValueError):
        feature_vector = []
    assert score is not None
    return BufferEntry(
        score=score,
        feature_vector=feature_vector,
        feature_version=str(data.get("feature_version") or ""),
        source_hash=str(data.get("source_hash") or ""),
        params=data.get("params") if isinstance(data.get("params"), dict) else {},
        metrics=metrics,
        state_id=str(data.get("state_id") or ""),
        iteration=int(data.get("iteration") or 0),
        train_path=str(data.get("train_path") or ""),
        run_name=str(data.get("run_name") or ""),
    )


def project_buffer_entries(
    entries: list[BufferEntry],
    schema: OperationSchema,
    args: argparse.Namespace,
) -> list[BufferEntry]:
    projected: list[BufferEntry] = []
    expected_dim = operation_feature_dim(schema)
    expected_version = operation_feature_version(schema)
    for entry in entries:
        features = features_for_buffer_entry(entry, schema, args)
        if features is None:
            continue
        if len(features.vector) != expected_dim:
            continue
        metrics = dict(entry.metrics)
        metrics["feature_projected_from_version"] = entry.feature_version
        metrics["feature_projected_to_version"] = expected_version
        projected.append(
            BufferEntry(
                score=entry.score,
                feature_vector=features.vector,
                feature_version=expected_version,
                source_hash=features.source_hash,
                params=features.params,
                metrics=metrics,
                state_id=entry.state_id,
                iteration=entry.iteration,
                train_path=entry.train_path,
                run_name=entry.run_name,
            )
        )
    return projected


def refresh_projected_buffer_entries(
    entries: list[BufferEntry],
    schema: OperationSchema,
    args: argparse.Namespace,
) -> None:
    projected = project_buffer_entries(entries, schema, args)
    entries[:] = projected


def features_for_buffer_entry(
    entry: BufferEntry,
    schema: OperationSchema,
    args: argparse.Namespace,
) -> Features | None:
    train_path = resolve_entry_train_path(entry.train_path, args)
    if train_path is not None and train_path.exists():
        try:
            return featurize_operation_schema(train_path, schema)
        except OSError:
            pass
    if isinstance(entry.params, dict) and entry.params:
        return featurize_operation_params(entry.params, schema, source_hash=entry.source_hash)
    expected_version = operation_feature_version(schema)
    if entry.feature_version == expected_version and len(entry.feature_vector) == operation_feature_dim(schema):
        return Features(vector=list(entry.feature_vector), params=dict(entry.params), source_hash=entry.source_hash)
    return None


def resolve_entry_train_path(train_path: str, args: argparse.Namespace) -> Path | None:
    if not train_path:
        return None
    path = Path(train_path)
    if path.is_absolute():
        return path
    project_root = Path(getattr(args, "project_root", Path.cwd())).resolve()
    return project_root / path


def featurize_operation_params(params: dict[str, Any], schema: OperationSchema, *, source_hash: str = "") -> Features:
    values = {canonical_name(str(name)): value for name, value in params.items()}
    vector: list[float] = []
    projected_params: dict[str, Any] = {}
    for parameter in schema.parameters.values():
        present = parameter.name in values
        raw_value = values.get(parameter.name)
        if present:
            projected_params[parameter.name] = raw_value
        if parameter.kind == "choice":
            vector.extend(1.0 if present and choice_values_equal(raw_value, choice) else 0.0 for choice in parameter.choices)
            vector.append(1.0 if present else 0.0)
            continue
        value = as_float(raw_value)
        if value is None:
            vector.append(0.0)
            vector.append(0.0)
            continue
        vector.append(normalize_operation_numeric(value, parameter))
        vector.append(1.0)
    return Features(vector=vector, params=projected_params, source_hash=source_hash)


def parse_buffer_entry(
    data: Any,
    expected_dim: int,
    *,
    expected_feature_version: str | None = None,
    args: argparse.Namespace | None = None,
) -> BufferEntry | None:
    if not isinstance(data, dict):
        return None
    score = as_float(data.get("score"))
    vector = data.get("feature_vector")
    if score is None or not isinstance(vector, list) or len(vector) != expected_dim:
        return None
    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    if not should_use_score_for_gp(
        score,
        status=str(metrics.get("status") or data.get("status") or ""),
        metrics=metrics,
        args=args,
    ):
        return None
    feature_version = str(data.get("feature_version") or "")
    if expected_feature_version is not None and feature_version != expected_feature_version:
        return None
    try:
        feature_vector = [float(value) for value in vector]
    except (TypeError, ValueError):
        return None
    return BufferEntry(
        score=score,
        feature_vector=feature_vector,
        feature_version=feature_version,
        source_hash=str(data.get("source_hash") or ""),
        params=data.get("params") if isinstance(data.get("params"), dict) else {},
        metrics=metrics,
        state_id=str(data.get("state_id") or ""),
        iteration=int(data.get("iteration") or 0),
        train_path=str(data.get("train_path") or ""),
        run_name=str(data.get("run_name") or ""),
    )


def append_buffer_entry(path: Path, entry: BufferEntry, *, mirror_path: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "score": entry.score,
        "feature_vector": entry.feature_vector,
        "feature_version": entry.feature_version,
        "source_hash": entry.source_hash,
        "params": entry.params,
        "metrics": entry.metrics,
        "state_id": entry.state_id,
        "iteration": entry.iteration,
        "train_path": entry.train_path,
        "run_name": entry.run_name,
    }
    line = json.dumps(payload, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as file:
        file.write(line)
    if mirror_path is not None and mirror_path.resolve() != path.resolve():
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        with mirror_path.open("a", encoding="utf-8") as file:
            file.write(line)


def should_use_score_for_gp(
    score: float | None,
    *,
    status: str = "",
    metrics: dict[str, Any] | None = None,
    args: argparse.Namespace | None = None,
) -> bool:
    if score is None or not finite_score(score):
        return False
    metrics = metrics if isinstance(metrics, dict) else {}
    status = str(status or metrics.get("status") or metrics.get("state_status") or "").strip()
    if args is not None and not bool(getattr(args, "gp_allow_failure_status", False)):
        if status in DEFAULT_GP_REJECT_STATUSES:
            return False
    elif args is None and status in DEFAULT_GP_REJECT_STATUSES:
        return False

    reject_at_or_above = None if args is None else as_float(getattr(args, "gp_reject_score_at_or_above", None))
    reject_at_or_below = None if args is None else as_float(getattr(args, "gp_reject_score_at_or_below", None))
    failure_score = 1.0e9 if args is None else as_float(getattr(args, "failure_score", None))
    minimize = True if args is None else not bool(getattr(args, "maximize", False))
    if reject_at_or_above is None and minimize and failure_score is not None:
        reject_at_or_above = failure_score
    if reject_at_or_below is None and not minimize and failure_score is not None:
        reject_at_or_below = failure_score
    score = float(score)
    if reject_at_or_above is not None and score >= float(reject_at_or_above):
        return False
    if reject_at_or_below is not None and score <= float(reject_at_or_below):
        return False
    return True


def make_buffer_entry(
    state: SearchState,
    args: argparse.Namespace,
    *,
    iteration: int,
    run_name: str,
    score_key: str,
) -> BufferEntry | None:
    score = as_float(state.metrics.get(score_key, state.score))
    if not should_use_score_for_gp(score, status=state.status, metrics=state.metrics, args=args):
        return None
    operation_schema = getattr(args, "operation_schema_object", None)
    features = featurize_for_surrogate(state.train_path, args, operation_schema)
    feature_version = feature_version_for_args(args, operation_schema)
    metrics = dict(state.metrics)
    metrics.setdefault("state_status", state.status)
    return BufferEntry(
        score=score,
        feature_vector=features.vector,
        feature_version=feature_version,
        source_hash=features.source_hash,
        params=features.params,
        metrics=metrics,
        state_id=state.state_id,
        iteration=iteration,
        train_path=str(state.train_path),
        run_name=run_name,
    )


def rbf_kernel(left: np.ndarray, right: np.ndarray, lengthscale: float) -> np.ndarray:
    np = require_numpy()
    diff = left[:, None, :] - right[None, :, :]
    return np.exp(-0.5 * np.sum(diff * diff, axis=-1) / (lengthscale * lengthscale))


def best_observed(scores: list[float], *, minimize: bool) -> float | None:
    finite = [float(score) for score in scores if finite_score(score)]
    if not finite:
        return None
    return min(finite) if minimize else max(finite)


def best_score_from_states(states: list[SearchState], *, minimize: bool) -> float | None:
    scores = [float(state.score) for state in states if state.score is not None and finite_score(state.score)]
    if not scores:
        return None
    return min(scores) if minimize else max(scores)


def updated_best_score(previous: float | None, current: float | None, *, minimize: bool) -> float | None:
    if current is None:
        return previous
    if previous is None:
        return current
    return current if is_better(current, previous, minimize=minimize) else previous


def format_gp_progress(summary: dict[str, Any]) -> str:
    status = str(summary.get("fit_status") or "unknown")
    history_size = summary.get("history_size")
    parts = [f"{status} n={history_size}"]
    nlml = as_float(summary.get("nlml_z"))
    rmse = as_float(summary.get("train_rmse"))
    mae = as_float(summary.get("train_mae"))
    best = as_float(summary.get("best_observed"))
    if nlml is not None:
        parts.append(f"nlml={nlml:.4g}")
    if rmse is not None:
        parts.append(f"rmse={rmse:.4g}")
    if mae is not None:
        parts.append(f"mae={mae:.4g}")
    if best is not None:
        parts.append(f"best={best:.6g}")
    return " ".join(parts)


def format_optional_float(value: Any) -> str:
    number = as_float(value)
    return "-" if number is None else f"{number:.6g}"


def format_score_delta(score: float | None, previous_best: float | None, minimize: bool) -> str:
    if score is None or previous_best is None or not finite_score(score) or not finite_score(previous_best):
        return "-"
    delta = float(score) - float(previous_best)
    improved = is_better(score, previous_best, minimize=minimize)
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.6g}{' improved' if improved else ''}"


def surrogate_sort_key(state: SearchState) -> tuple[float, int, str]:
    score = as_float(state.metrics.get("surrogate_score"))
    if score is None:
        score = float("inf")
    order, state_id = state_id_sort_key(state.state_id)
    return float(score), order, state_id


def state_id_sort_key(state_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", state_id)
    if not match:
        return 10**12, state_id
    return int(match.group(1)), state_id


def write_state_update(engine: SearchEngine, state: SearchState) -> None:
    engine._write_state_meta(state)
    engine._record_manifest(state)


def resolve_operation_schema(args: argparse.Namespace, project_root: Path) -> OperationSchema | None:
    schema_path = args.operation_schema
    resume_path = getattr(args, "resume_from", None)
    if schema_path is None and resume_path is not None:
        path = resume_path if Path(resume_path).is_absolute() else project_root / resume_path
        run_dir = path.parent if path.is_file() else path
        candidate = run_dir / "operation_schema.json"
        if candidate.exists():
            schema_path = candidate
    if schema_path is None and args.generator in OPERATION_GENERATORS:
        train_name = Path(args.train_file).name
        candidates = []
        if train_name == "mock_train.py":
            candidates.append(project_root / "ldm_task" / "operation_schema_mock_train.json")
        candidates.append(project_root / "ldm_task" / "operation_schema_real_train.json")
        for candidate in candidates:
            if candidate.exists():
                schema_path = candidate
                break
    if schema_path is None:
        return None
    return load_operation_schema(schema_path, project_root)


def resolve_buffer_path(buffer_arg: Path | None, project_root: Path, run_buffer_path: Path) -> Path:
    if buffer_arg is None:
        return run_buffer_path
    path = buffer_arg if buffer_arg.is_absolute() else project_root / buffer_arg
    if path.is_dir():
        candidate = path / "model_based_buffer.jsonl"
        if candidate.exists():
            return candidate
        summary_path = path / "model_based_summary.json"
        summary = load_json_file(summary_path)
        if summary is not None:
            buffer_value = summary.get("run_buffer") or summary.get("buffer")
            if isinstance(buffer_value, str):
                return Path(buffer_value)
    if path.name == "model_based_summary.json":
        summary = load_json_file(path)
        if summary is not None:
            buffer_value = summary.get("run_buffer") or summary.get("buffer")
            if isinstance(buffer_value, str):
                return Path(buffer_value)
    return path


def resolve_feedback_path(feedback_arg: Path | None, project_root: Path, out_dir: Path) -> Path:
    if feedback_arg is None:
        return out_dir / "iteration_feedback.tsv"
    return feedback_arg if feedback_arg.is_absolute() else project_root / feedback_arg


def load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def sync_run_buffer(source_path: Path, run_buffer_path: Path) -> None:
    run_buffer_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() == run_buffer_path.resolve():
        if not run_buffer_path.exists():
            run_buffer_path.touch()
        return
    if source_path.exists():
        shutil.copy2(source_path, run_buffer_path)
    elif not run_buffer_path.exists():
        run_buffer_path.touch()


def resolve_model_based_resume_info(resume_arg: Path, project_root: Path) -> dict[str, Any]:
    path = resume_arg if resume_arg.is_absolute() else project_root / resume_arg
    if path.is_file():
        if path.name == "model_based_summary.json":
            model_based_summary_path = path
            run_dir = path.parent
        elif path.name == "summary.json":
            run_dir = path.parent
            model_based_summary_path = run_dir / "model_based_summary.json"
        else:
            raise SystemExit("--resume-from must point to a model-based run directory, model_based_summary.json, or summary.json.")
    else:
        run_dir = path
        model_based_summary_path = run_dir / "model_based_summary.json"
    if not run_dir.exists():
        raise SystemExit(f"resume run directory does not exist: {run_dir}")
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"resume run is missing summary.json: {summary_path}")
    summary = load_json_file(summary_path) or {}
    model_based_summary = load_json_file(model_based_summary_path) or {}
    iterations = model_based_summary.get("iterations")
    if not isinstance(iterations, list):
        iterations = []
    completed_iterations = [
        int(item.get("iteration"))
        for item in iterations
        if isinstance(item, dict) and isinstance(item.get("iteration"), int)
    ]
    if completed_iterations:
        next_iteration = max(completed_iterations) + 1
    else:
        next_iteration = infer_next_iteration_from_states(run_dir)
    best_state_id = model_based_summary.get("best_state_id") or summary.get("best_state_id")
    best_score = model_based_summary.get("best_score", summary.get("best_score"))
    real_evaluations = int(summary.get("evaluation_count") or 0)
    if real_evaluations <= 0:
        real_evaluations = count_real_evaluations_from_states(run_dir)
    return {
        "run_dir": run_dir.resolve(),
        "summary": summary,
        "model_based_summary": model_based_summary,
        "warmup": model_based_summary.get("warmup"),
        "iterations": iterations,
        "next_iteration": next_iteration,
        "best_state_id": best_state_id,
        "best_score": best_score,
        "real_evaluations": real_evaluations,
    }


def apply_model_based_resume_defaults(args: argparse.Namespace, resume_info: dict[str, Any], project_root: Path) -> None:
    summary_args = resume_info.get("summary", {}).get("args")
    if not isinstance(summary_args, dict):
        summary_args = {}
    explicit = set(getattr(args, "_explicit_options", set()))
    defaults = {
        "method": "auto",
        "breadth": 2,
        "depth": 2,
        "beam_width": 0,
        "select_from": "leaves",
        "seed_policy": "best",
        "max_generated_per_iteration": 1024,
        "evaluate_root": False,
        "skip_eval": False,
        "max_real_evaluations": 0,
        "warmup": 0,
        "warmup_include_root": False,
        "warmup_strategy": "auto",
        "warmup_seed": 0,
        "warmup_updates_seed": False,
        "eval_command": "uv run python {train_path}",
        "eval_shell": False,
        "timeout_seconds": 900,
        "score_key": "val_bpb",
        "maximize": False,
        "failure_score": 1.0e9,
        "gp_reject_score_at_or_above": None,
        "gp_reject_score_at_or_below": None,
        "gp_allow_failure_status": False,
        "generator": "api",
        "llm_url": os.environ.get("TTS_LLM_URL") or os.environ.get("LLM_BASE_URL") or "",
        "llm_model_name": (
            os.environ.get("TTS_LLM_MODEL")
            or os.environ.get("LLM_MODEL_NAME")
            or os.environ.get("LLM_MODEL")
            or ""
        ),
        "api_key": (
            os.environ.get("TTS_LLM_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        ),
        "max_tokens": 4096,
        "temperature": 0.7,
        "logprobs": False,
        "top_logprobs": None,
        "disable_thinking": False,
        "concurrency": 1,
        "num_edits_per_step": 1,
        "prompt_max_chars": 100_000,
        "context_file": None,
        "context": "",
        "instruction": "",
        "feedback_max_rows": 12,
        "acquisition_feedback": "none",
        "acquisition_feedback_probes": 96,
        "acquisition_feedback_top_k": 3,
        "acquisition_feedback_max_chars": 1800,
        "response_log_chars": 20_000,
        "operation_schema": None,
        "operation_retries": 2,
        "max_operations_per_step": 2,
        "operation_features": False,
        "initial_operation_features": "5",
        "max_active_operation_features": 0,
        "allow_feature_expansion": True,
        "allow_new_feature_specs": False,
        "mock_expand_every": 0,
        "buffer": None,
        "surrogate_mode": "lcb",
        "gp_beta": 1.0,
        "gp_xi": 0.001,
        "gp_lengthscale": 1.5,
        "gp_noise": 1.0e-4,
        "prior_score": 1.0,
        "prior_std": 0.15,
        "hash_dims": 48,
    }
    for name, default in defaults.items():
        if name in explicit:
            continue
        if name not in summary_args:
            continue
        value = summary_args[name]
        if name in {"train_file", "context_file", "operation_schema", "buffer"} and value is not None:
            value = Path(value)
            if not value.is_absolute():
                value = project_root / value
        setattr(args, name, value)
    if "train_file" not in explicit:
        train_value = summary_args.get("train_file")
        if isinstance(train_value, str) and train_value:
            args.train_file = Path(train_value)
    if "feedback_tsv" not in explicit and args.feedback_tsv is None:
        feedback_value = summary_args.get("feedback_tsv") or resume_info.get("model_based_summary", {}).get("feedback_tsv")
        if isinstance(feedback_value, str) and feedback_value:
            args.feedback_tsv = Path(feedback_value)
    if "operation_schema" not in explicit and args.operation_schema is None:
        schema_value = summary_args.get("operation_schema") or summary_args.get("operation_schema_path")
        if isinstance(schema_value, str) and schema_value:
            args.operation_schema = Path(schema_value)
    if "initial_operation_features" not in explicit:
        active_names = summary_args.get("operation_schema_feature_names")
        if isinstance(active_names, list) and active_names:
            args.initial_operation_features = ",".join(str(name) for name in active_names)
    if "max_active_operation_features" not in explicit:
        max_active = summary_args.get("max_active_operation_features")
        active_count = summary_args.get("operation_schema_feature_count")
        full_json = resume_info.get("run_dir") / "operation_schema.json"
        full_schema = load_json_file(full_json)
        full_parameters = full_schema.get("parameters") if isinstance(full_schema, dict) else None
        if isinstance(max_active, int):
            args.max_active_operation_features = max_active
        elif isinstance(full_parameters, dict):
            args.max_active_operation_features = len(full_parameters)
        elif isinstance(active_count, int):
            args.max_active_operation_features = max(args.max_active_operation_features, active_count)
    if "buffer" not in explicit:
        run_buffer_value = summary_args.get("run_buffer") or resume_info.get("model_based_summary", {}).get("run_buffer")
        buffer_value = summary_args.get("buffer") or resume_info.get("model_based_summary", {}).get("buffer")
        chosen_buffer = run_buffer_value or buffer_value
        if isinstance(chosen_buffer, str) and chosen_buffer:
            args.buffer = Path(chosen_buffer)
    if "warmup" not in explicit:
        args.warmup = 0
    if "evaluate_root" not in explicit:
        args.evaluate_root = False


def explicit_options_from_argv(argv: list[str]) -> set[str]:
    mapping = {
        "--resume-from": "resume_from",
        "--train-file": "train_file",
        "--out-dir": "out_dir",
        "--run-name": "run_name",
        "--export-best": "export_best",
        "--eval-command": "eval_command",
        "--eval-shell": "eval_shell",
        "--timeout-seconds": "timeout_seconds",
        "--score-key": "score_key",
        "--maximize": "maximize",
        "--failure-score": "failure_score",
        "--gp-reject-score-at-or-above": "gp_reject_score_at_or_above",
        "--gp-reject-score-at-or-below": "gp_reject_score_at_or_below",
        "--gp-allow-failure-status": "gp_allow_failure_status",
        "--llm-url": "llm_url",
        "--llm-model-name": "llm_model_name",
        "--api-key": "api_key",
        "--max-tokens": "max_tokens",
        "--top-logprobs": "top_logprobs",
        "--disable-thinking": "disable_thinking",
        "--num-edits-per-step": "num_edits_per_step",
        "--prompt-max-chars": "prompt_max_chars",
        "--context-file": "context_file",
        "--feedback-max-rows": "feedback_max_rows",
        "--feedback-tsv": "feedback_tsv",
        "--acquisition-feedback": "acquisition_feedback",
        "--acquisition-feedback-probes": "acquisition_feedback_probes",
        "--acquisition-feedback-top-k": "acquisition_feedback_top_k",
        "--acquisition-feedback-max-chars": "acquisition_feedback_max_chars",
        "--no-progress": "no_progress",
        "--progress-width": "progress_width",
        "--response-log-chars": "response_log_chars",
        "--operation-schema": "operation_schema",
        "--operation-retries": "operation_retries",
        "--max-operations-per-step": "max_operations_per_step",
        "--operation-features": "operation_features",
        "--operation-surrogate": "operation_features",
        "--initial-operation-features": "initial_operation_features",
        "--initial-expansion-parameters": "initial_operation_features",
        "--max-active-operation-features": "max_active_operation_features",
        "--max-expansion-parameters": "max_active_operation_features",
        "--disable-feature-expansion": "allow_feature_expansion",
        "--disable-expansion-schema-updates": "allow_feature_expansion",
        "--allow-new-feature-specs": "allow_new_feature_specs",
        "--allow-new-expansion-parameters": "allow_new_feature_specs",
        "--mock-expand-every": "mock_expand_every",
        "--surrogate-mode": "surrogate_mode",
        "--gp-beta": "gp_beta",
        "--gp-xi": "gp_xi",
        "--gp-lengthscale": "gp_lengthscale",
        "--gp-noise": "gp_noise",
        "--prior-score": "prior_score",
        "--prior-std": "prior_std",
        "--hash-dims": "hash_dims",
        "--max-real-evaluations": "max_real_evaluations",
        "--max-generated-per-iteration": "max_generated_per_iteration",
        "--warmup-include-root": "warmup_include_root",
        "--warmup-strategy": "warmup_strategy",
        "--warmup-seed": "warmup_seed",
        "--warmup-updates-seed": "warmup_updates_seed",
        "--select-from": "select_from",
        "--beam-width": "beam_width",
    }
    explicit = set()
    for item in argv:
        option = item.split("=", 1)[0]
        if not option.startswith("--"):
            continue
        if option in mapping:
            explicit.add(mapping[option])
        else:
            explicit.add(option[2:].replace("-", "_"))
    return explicit


def load_existing_search_states(engine: SearchEngine, out_dir: Path) -> None:
    state_paths = sorted((out_dir / "states").glob("state_*/meta.json"), key=lambda path: state_id_sort_key(path.parent.name))
    states: list[SearchState] = []
    max_index = -1
    for meta_path in state_paths:
        data = load_json_file(meta_path) or {}
        state_id = str(data.get("state_id") or meta_path.parent.name)
        state = SearchState(
            state_id=state_id,
            parent_id=data.get("parent_id"),
            depth=int(data.get("depth") or 0),
            workdir=Path(data.get("workdir") or meta_path.parent),
            train_path=Path(data.get("train_path") or meta_path.parent / "train.py"),
            status=str(data.get("status") or "created"),
            score=data.get("score"),
            metrics=data.get("metrics") if isinstance(data.get("metrics"), dict) else {},
            token_usage=int(data.get("token_usage") or 0),
            token_usage_detail=data.get("token_usage_detail") if isinstance(data.get("token_usage_detail"), dict) else {},
            elapsed_seconds=float(data.get("elapsed_seconds") or 0.0),
            error=data.get("error"),
            description=str(data.get("description") or ""),
            prompt_path=None if data.get("prompt_path") is None else Path(data["prompt_path"]),
            response_path=None if data.get("response_path") is None else Path(data["response_path"]),
            patch_path=None if data.get("patch_path") is None else Path(data["patch_path"]),
            llm_response=str(data.get("llm_response") or ""),
            llm_response_truncated=bool(data.get("llm_response_truncated")),
            edits=data.get("edits") if isinstance(data.get("edits"), list) else [],
        )
        states.append(state)
        order, _ = state_id_sort_key(state_id)
        if order < 10**12:
            max_index = max(max_index, order)
    engine.states = states
    engine._counter = max_index + 1


def load_feedback_memory_rows(feedback_memory: FeedbackMemory, feedback_path: Path) -> None:
    if not feedback_path.exists():
        return
    lines = feedback_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return
    header = lines[0].split("\t")
    if header != feedback_memory.columns:
        return
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        values = line.split("\t")
        rows.append({column: values[index] if index < len(values) else "" for index, column in enumerate(header)})
    feedback_memory.rows = rows


def state_from_id(engine: SearchEngine, state_id: Any) -> SearchState | None:
    if not isinstance(state_id, str) or not state_id:
        return None
    for state in engine.states:
        if state.state_id == state_id:
            return state
    return None


def latest_evaluated_child_state(engine: SearchEngine) -> SearchState | None:
    candidates = [
        state
        for state in engine.states
        if state.parent_id is not None and state.score is not None and finite_score(state.score)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda state: state_id_sort_key(state.state_id))[-1]


def infer_next_iteration_from_states(run_dir: Path) -> int:
    max_iteration = 0
    for meta_path in (run_dir / "states").glob("state_*/meta.json"):
        data = load_json_file(meta_path) or {}
        metrics = data.get("metrics")
        if isinstance(metrics, dict) and isinstance(metrics.get("model_based_iteration"), int):
            max_iteration = max(max_iteration, int(metrics["model_based_iteration"]))
    return max_iteration + 1


def count_real_evaluations_from_states(run_dir: Path) -> int:
    total = 0
    for meta_path in (run_dir / "states").glob("state_*/meta.json"):
        data = load_json_file(meta_path) or {}
        status = str(data.get("status") or "")
        score = as_float(data.get("score"))
        if status in {"evaluated", "evaluated_nonzero", "score_missing", "crash", "timeout", "evaluation_error"}:
            total += 1
        elif score is not None and status not in {"seed", "generated", "surrogate_scored", "evaluation_deferred"}:
            total += 1
    return total


def choose_seed_path(
    args: argparse.Namespace,
    original: Path,
    best_actual: SearchState | None,
    latest_actual: SearchState | None,
) -> Path:
    if args.seed_policy == "best" and best_actual is not None:
        return best_actual.train_path
    if args.seed_policy == "latest" and latest_actual is not None:
        return latest_actual.train_path
    return original


def build_task_context(project_root: Path, args: argparse.Namespace) -> str:
    parts = [DEFAULT_TASK_CONTEXT]
    if args.context_file is not None:
        context_path = args.context_file if args.context_file.is_absolute() else project_root / args.context_file
        parts.append(context_path.read_text(encoding="utf-8").strip())
    if args.context.strip():
        parts.append(args.context.strip())
    parts.append(
        "Model-based search note: intermediate candidates may be valued by a GP surrogate before "
        "real execution. Propose edits that are coherent complete candidates, because a selected "
        "leaf may be executed as the next real experiment."
    )
    return "\n\n".join(part for part in parts if part)


def default_run_name(args: argparse.Namespace, train_file: Path) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    method = resolve_search_method(args)
    return "_".join(
        [
            "model_based",
            safe_path_tag(method),
            safe_path_tag(args.generator),
            safe_path_tag(train_file.stem, default="train"),
            f"b{max(1, args.breadth)}",
            f"d{max(1, args.depth)}",
            f"i{max(1, args.iterations)}",
            stamp,
        ]
    )


def resolve_search_method(args: argparse.Namespace) -> str:
    method = SEARCH_METHOD_ALIASES.get(str(getattr(args, "method", "auto")), "auto")
    if method == "auto":
        return "beam_search" if int(getattr(args, "beam_width", 0)) > 0 else "tree_search"
    return canonical_search_method(method)


def estimate_progress_total(args: argparse.Namespace, generated_per_iteration: int) -> int:
    iterations = max(0, int(args.iterations))
    generated_total = max(0, int(generated_per_iteration)) * iterations
    warmup_total = max(0, int(getattr(args, "warmup", 0)))
    if args.skip_eval:
        return generated_total + warmup_total
    real_total = iterations
    if args.evaluate_root:
        real_total += 1
    real_total += warmup_total
    if args.max_real_evaluations > 0:
        real_total = min(real_total, max(0, int(args.max_real_evaluations)))
    return generated_total + real_total


def feature_dim(hash_dims: int) -> int:
    return 2 * len(COMMON_NUMERIC_NAMES) + 6 + max(0, int(hash_dims))


def finite_score(value: Any) -> bool:
    return is_finite_number(value)


def is_better(candidate: float | None, incumbent: float | None, *, minimize: bool) -> bool:
    if candidate is None:
        return False
    if incumbent is None:
        return True
    return float(candidate) < float(incumbent) if minimize else float(candidate) > float(incumbent)


def as_float(value: Any) -> float | None:
    return shared_as_float(value)


def write_model_based_summary(
    out_dir: Path,
    summary_path: Path,
    buffer_path: Path,
    run_buffer_path: Path,
    log_path: Path,
    feedback_path: Path,
    warmup_record: dict[str, Any],
    iteration_records: list[dict[str, Any]],
    best_actual: SearchState | None,
    *,
    ldm_task_spec: dict[str, Any] | None = None,
) -> None:
    serializable_records = []
    for record in iteration_records:
        item = {
            key: value
            for key, value in record.items()
            if key not in {"actual_states", "selected_state"}
        }
        serializable_records.append(item)
    serializable_warmup = {
        key: value
        for key, value in warmup_record.items()
        if key not in {"actual_states"}
    }
    payload = {
        "summary": str(summary_path),
        "buffer": str(buffer_path),
        "run_buffer": str(run_buffer_path),
        "log": str(log_path),
        "feedback_tsv": str(feedback_path),
        "best_state_id": None if best_actual is None else best_actual.state_id,
        "best_score": None if best_actual is None else best_actual.score,
        "warmup": serializable_warmup,
        "iterations": serializable_records,
    }
    if ldm_task_spec is not None:
        payload["ldm_task_spec"] = ldm_task_spec
    (out_dir / "model_based_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
