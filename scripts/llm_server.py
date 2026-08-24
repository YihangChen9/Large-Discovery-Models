#!/usr/bin/env python3
"""OpenAI-compatible chat server for Qwen3.5-9B via transformers.

Serves GET /v1/models and POST /v1/chat/completions on the given host/port.
Loads the model in bf16 on the GPU(s) selected by CUDA_VISIBLE_DEVICES.

The Qwen3.5 chat template only honors ``enable_thinking`` when it is passed
as a direct template kwarg (nested ``chat_template_kwargs`` is not picked up
by transformers 5.15's apply_chat_template), so this server unwraps
``body["chat_template_kwargs"]["enable_thinking"]`` and forwards it directly.

Usage:
    SERVE_MODEL=/path/to/model SERVE_PORT=8012 \
        CUDA_VISIBLE_DEVICES=1 python scripts/llm_server.py
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

MODEL_PATH = os.environ.get("SERVE_MODEL", "/mnt/data0/hf_models/models/Qwen3.5-9B")
SERVED_NAME = os.environ.get(
    "SERVE_MODEL_NAME",
    os.path.basename(MODEL_PATH.rstrip("/")),
)
HOST = os.environ.get("SERVE_HOST", "127.0.0.1")
PORT = int(os.environ.get("SERVE_PORT", "8012"))
MAX_MODEL_LEN = int(os.environ.get("SERVE_MAX_MODEL_LEN", "32768"))
DTYPE = os.environ.get("SERVE_DTYPE", "bfloat16")


def _load() -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[DTYPE]
    t0 = time.monotonic()
    print(f"[llm_server] loading tokenizer from {MODEL_PATH}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"[llm_server] loading model (dtype={DTYPE})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    print(
        f"[llm_server] model ready: {type(model).__name__} in {time.monotonic() - t0:.1f}s "
        f"cuda_mem={torch.cuda.memory_allocated() / 1e9:.1f}GB",
        flush=True,
    )
    return tok, model


TOKENIZER, MODEL = _load()
GEN_LOCK = threading.Lock()


class PresencePenaltyLogitsProcessor:
    """Subtract ``presence_penalty`` from logits of tokens already emitted."""

    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.penalty == 0.0:
            return scores
        for b in range(scores.shape[0]):
            seen = torch.unique(input_ids[b])
            scores[b, seen] -= self.penalty
        return scores


def build_generation_kwargs(body: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}

    max_tokens = body.get("max_tokens") or body.get("max_new_tokens")
    if max_tokens:
        kwargs["max_new_tokens"] = int(max_tokens)

    temperature = body.get("temperature")
    if temperature is None:
        temperature = 1.0
    if isinstance(temperature, bool):
        temperature = 1.0
    do_sample = float(temperature) > 0.0
    kwargs["do_sample"] = do_sample
    if do_sample:
        kwargs["temperature"] = float(temperature)
    else:
        kwargs["temperature"] = None

    top_p = body.get("top_p")
    if top_p is not None:
        kwargs["top_p"] = float(top_p)

    top_k = body.get("top_k")
    if top_k is not None:
        kwargs["top_k"] = int(top_k)

    min_p = body.get("min_p")
    if min_p is not None:
        kwargs["min_p"] = float(min_p)

    rep_pen = body.get("repetition_penalty")
    if rep_pen is not None:
        kwargs["repetition_penalty"] = float(rep_pen)

    presence = body.get("presence_penalty")
    if presence is not None and float(presence) != 0.0:
        kwargs.setdefault("logits_processor", [])
        kwargs["logits_processor"].append(
            PresencePenaltyLogitsProcessor(float(presence))
        )

    stop = body.get("stop")
    if isinstance(stop, str):
        kwargs["stop_strings"] = [stop]
    elif isinstance(stop, list) and stop:
        kwargs["stop_strings"] = [str(s) for s in stop]

    return kwargs


def parse_tool_calls_xml(text: str) -> list[dict[str, Any]]:
    """Parse Qwen-style XML tool calls into OpenAI tool_calls entries.

    Model output format (from the Qwen3.5 chat template):
        <tool_call>
        <function=propose_operation_feature>
        <parameter=name>
        DEVICE_BATCH_SIZE
        </parameter>
        </function>
        </tool_call>

    Returns a list of OpenAI-compatible tool call dicts, or [] when the
    text contains no tool calls.
    """
    import re as _re

    calls: list[dict[str, Any]] = []
    pattern = _re.compile(
        r"<tool_call>\s*<function=(?P<name>[^>\n]+)>(?P<body>.*?)</function>\s*</tool_call>",
        flags=_re.DOTALL,
    )
    param_pattern = _re.compile(
        r"<parameter=([^>\n]+)>(.*?)</parameter>",
        flags=_re.DOTALL,
    )
    for match in pattern.finditer(text):
        name = match.group("name").strip()
        body = match.group("body")
        arguments: dict[str, Any] = {}
        for pmatch in param_pattern.finditer(body):
            key = pmatch.group(1).strip()
            value = pmatch.group(2).strip()
            try:
                arguments[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                arguments[key] = value
        if name:
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:16]}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            )
    return calls


def _normalize_tool_call_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse string tool-call arguments into dicts for the chat template.

    OpenAI-compatible clients represent assistant tool calls with
    ``function.arguments`` as a JSON string; the Qwen3.5 template iterates
    ``tool_call.arguments|items`` and needs a dict. Return a shallow copy of
    ``messages`` with arguments parsed where possible.
    """
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            normalized.append(message)
            continue
        copy = dict(message)
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            parsed_calls: list[dict[str, Any]] = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    parsed_calls.append(tool_call)
                    continue
                tc = dict(tool_call)
                function = tc.get("function")
                if isinstance(function, dict):
                    fn = dict(function)
                    arguments = fn.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            fn["arguments"] = json.loads(arguments)
                        except (json.JSONDecodeError, ValueError):
                            pass
                    tc["function"] = fn
                parsed_calls.append(tc)
            copy["tool_calls"] = parsed_calls
        normalized.append(copy)
    return normalized


app = FastAPI()


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local-transformers",
                "root": MODEL_PATH,
                "parent": None,
                "max_model_len": MAX_MODEL_LEN,
                "permission": [{"id": f"modelperm-{uuid.uuid4().hex}", "object": "model_permission", "created": int(time.time()), "allow_create_engine": False, "allow_sampling": True, "allow_logprobs": False, "allow_search_indices": False, "allow_view": True, "allow_fine_tuning": False, "organization": "*", "group": None, "is_blocking": False}],
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": {"message": f"bad json: {exc}", "type": "invalid_request_error"}})

    messages = body.get("messages") or []
    if not messages:
        return JSONResponse(status_code=400, content={"error": {"message": "no messages", "type": "invalid_request_error"}})

    # Unwrap nested chat_template_kwargs -> direct template kwargs.
    template_kwargs: dict[str, Any] = {}
    ctk = body.get("chat_template_kwargs")
    if isinstance(ctk, dict):
        template_kwargs.update(ctk)
    # Also accept enable_thinking at top level.
    if "enable_thinking" in body:
        template_kwargs.setdefault("enable_thinking", body["enable_thinking"])
    # Default to thinking disabled for this reasoning-capable model: several
    # LDM task generators (e.g. nanoGPT operation-tool proposals) request JSON
    # but do not send chat_template_kwargs, and an enabled thinking block
    # corrupts/truncates their structured output. Clients that explicitly want
    # thinking can still pass enable_thinking=true.
    template_kwargs.setdefault("enable_thinking", False)

    # OpenAI tools -> Qwen3.5 chat template tool declarations.
    tools = body.get("tools")
    tool_choice = body.get("tool_choice")
    if tools:
        template_kwargs["tools"] = tools
        if tool_choice not in (None, "auto", "none"):
            template_kwargs.setdefault("tool_choice", tool_choice)

    # The Qwen3.5 chat template iterates `tool_call.arguments|items`, which
    # requires arguments to be a dict. OpenAI-style messages carry them as a
    # JSON string, so parse assistant tool-call arguments before rendering.
    messages = _normalize_tool_call_arguments(messages)

    prompt = TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )

    gen_kwargs = build_generation_kwargs(body)
    # OpenAI `n` -> number of independent completions (default 1).
    try:
        n = max(1, int(body.get("n") or 1))
    except (TypeError, ValueError):
        n = 1
    if n > 1:
        gen_kwargs["num_return_sequences"] = n
    t0 = time.monotonic()

    with GEN_LOCK:
        inputs = TOKENIZER(prompt, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]
        if input_len > MAX_MODEL_LEN:
            return JSONResponse(status_code=400, content={"error": {"message": f"prompt too long: {input_len} > {MAX_MODEL_LEN}", "type": "invalid_request_error"}})
        with torch.inference_mode():
            outputs = MODEL.generate(**inputs, **gen_kwargs)
        output_sequences = outputs[:, input_len:] if outputs.ndim == 2 else outputs[0][:, input_len:]

    texts = [
        TOKENIZER.decode(seq, skip_special_tokens=True)
        for seq in output_sequences
    ]
    usage = {
        "prompt_tokens": input_len,
        "completion_tokens": int(output_sequences.shape[-1]),
        "total_tokens": input_len + int(output_sequences.shape[-1]),
    }
    print(
        f"[llm_server] n={n} {usage['prompt_tokens']}->{usage['completion_tokens']} tok "
        f"in {time.monotonic() - t0:.1f}s",
        flush=True,
    )

    choices = []
    for idx, text in enumerate(texts):
        message: dict[str, Any] = {"role": "assistant", "content": text}
        finish_reason = "stop"
        if tools:
            tool_calls = parse_tool_calls_xml(text)
            if tool_calls:
                # Strip the raw XML tool-call markup from the visible content.
                import re as _re

                cleaned = _re.sub(
                    r"<tool_call>.*?</tool_call>",
                    "",
                    text,
                    flags=_re.DOTALL,
                ).strip()
                message = {"role": "assistant", "content": cleaned or None, "tool_calls": tool_calls}
                finish_reason = "tool_calls"
        choices.append(
            {
                "index": idx,
                "message": message,
                "finish_reason": finish_reason,
            }
        )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": SERVED_NAME,
        "choices": choices,
        "usage": usage,
    }


if __name__ == "__main__":
    import uvicorn

    print(f"[llm_server] serving {SERVED_NAME} at http://{HOST}:{PORT}/v1", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
