"""Slime integration for the LDM environment.

Mirrors Slime's search-r1 example: a custom rollout ``generate`` function runs
one multi-turn agent-environment loop per sample, and a custom ``reward_func``
exposes the episode reward. The sample prompt is an :class:`EpisodeSpec` JSON;
the rendered policy prompt comes from ``env.reset()``. Environment feedback is
appended with ``trainable=False`` (loss-mask 0), policy turns with
``trainable=True``.

Usage in a Slime launch script::

    export PYTHONPATH=$REPO_ROOT/rl:$REPO_ROOT
    --custom-generate-function-path ldm_rl.bridge.generate
    --custom-rm-path ldm_rl.bridge.reward_func
    --prompt-data /path/to/ldm_episodes.jsonl
    --input-key prompt --label-key label

Slime imports happen lazily so the environment core stays importable without
Slime installed (unit tests inject fake dependencies instead).
"""

from __future__ import annotations

import asyncio
from typing import Any


def _load_slime_deps() -> tuple[Any, Any]:
    from slime.utils.http_utils import post
    from slime.utils.types import Sample

    return post, Sample


def _load_generate_state(args: Any) -> Any:
    from slime.rollout.sglang_rollout import GenerateState

    return GenerateState(args)


def _apply_chat_template(state: Any, text: str) -> str:
    """Wrap the rendered prompt in the tokenizer chat template when possible.

    Reasoning-capable checkpoints (e.g. Qwen3.5) emit a ``<think>...</think>``
    block by default, which consumes the rollout response budget and yields
    ``length``-truncated turns before any candidate is produced. Disable
    thinking via the ``enable_thinking`` template kwarg, mirroring
    ``scripts/llm_server.py``; tokenizers without that kwarg fall back to the
    plain template.
    """

    tokenizer = getattr(state, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return text

    def _render(**kwargs: Any) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=False,
            **kwargs,
        )

    try:
        templated = _render(enable_thinking=False)
    except TypeError:
        try:
            templated = _render()
        except Exception:  # noqa: BLE001 - fall back to the untemplated prompt
            return text
    except Exception:  # noqa: BLE001 - fall back to the untemplated prompt
        return text
    return templated if isinstance(templated, str) and templated.strip() else text


_OBS_SENTINEL = "<<<LDM_OBS>>>"
_PREV_SENTINEL = "<<<LDM_PREV>>>"


def _observation_wrapper(state: Any) -> tuple[str, str]:
    """Return the (prefix, suffix) that make env feedback a proper user turn.

    Environment feedback used to be concatenated straight onto the response,
    which puts plain text immediately after the policy's end-of-turn marker.
    For a chat model that is malformed: after `<|im_end|>` the template expects
    a new role marker, not prose. Round 0 always parsed (its prompt goes
    through `_apply_chat_template`), and every later round came back with no
    JSON at all -- 231 unparseable turns against 52 successful evaluations in
    one measured run.

    The wrapper is derived from the tokenizer's own chat template rather than
    hardcoded, by rendering an assistant turn followed by a user turn and
    slicing around the two sentinels. Returns ("", "") when the tokenizer has
    no chat template, which restores the previous behaviour.
    """

    tokenizer = getattr(state, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return "", ""
    try:
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "assistant", "content": _PREV_SENTINEL},
                {"role": "user", "content": _OBS_SENTINEL},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:  # noqa: BLE001 - fall back to raw concatenation
        return "", ""
    if not isinstance(rendered, str):
        return "", ""
    prev_at = rendered.find(_PREV_SENTINEL)
    obs_at = rendered.find(_OBS_SENTINEL)
    if prev_at < 0 or obs_at < 0 or obs_at < prev_at:
        return "", ""
    return (
        rendered[prev_at + len(_PREV_SENTINEL) : obs_at],
        rendered[obs_at + len(_OBS_SENTINEL) :],
    )


async def generate(args, sample, sampling_params, evaluation: bool = False) -> Any:
    """Slime custom generate function: one LDM campaign episode per sample."""

    post, Sample = _load_slime_deps()
    from ldm_rl.episodes import EpisodeSpec
    from ldm_rl.factories import build_env

    try:
        spec = EpisodeSpec.from_json(str(sample.prompt))
        if spec.mode == "real":
            from ldm_rl.remote_env import RemoteLDMEnv

            task_python = str(
                spec.real.get("task_python")
                or "/mnt/data0/ys/LDM/tasks/small_molecule/.venv/bin/python"
            )
            env = RemoteLDMEnv(spec, task_python)
        else:
            env = build_env(
                spec.task,
                mode=spec.mode,
                config=spec.to_env_config(),
                context=spec.context,
                seed=spec.seed,
                **spec.real,
            )
    except Exception as exc:  # noqa: BLE001 - rollout must not crash the pool
        sample.status = Sample.Status.FAILED
        sample.reward = 0.0
        sample.metadata["env_error"] = f"{type(exc).__name__}: {exc}"
        return sample

    state = _load_generate_state(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # The sample prompt is the raw EpisodeSpec JSON; render the policy prompt
    # here so --apply-chat-template on the data source cannot corrupt the spec.
    prompt_text = env.reset()
    prompt_text = _apply_chat_template(state, prompt_text)
    prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)[
        "input_ids"
    ]
    sample.prompt = prompt_text
    sample.metadata["episode_spec"] = spec.to_dict()
    sample.metadata["env_steps"] = []
    sample.tokens = list(prompt_token_ids)
    sample.loss_mask = []

    response = ""
    response_token_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] = []
    total_reward = 0.0
    last_step = None
    last_finish = "stop"
    output = None

    try:
        for _turn_idx in range(spec.iterations):
            payload: dict[str, Any] = {
                "text": prompt_text + response,
                "sampling_params": sampling_params,
                "return_logprob": True,
            }
            output = await post(url, payload)
            meta_info = output["meta_info"]
            last_finish = meta_info["finish_reason"]["type"]
            if last_finish == "abort":
                sample.status = Sample.Status.ABORTED
                return sample

            cur_response = output["text"]
            if "output_token_logprobs" not in meta_info:
                raise RuntimeError(
                    "output_token_logprobs missing from sglang meta_info; the "
                    "custom LDM generate function requires return_logprob"
                )
            cur_token_ids = [
                item[1] for item in meta_info["output_token_logprobs"]
            ]
            cur_log_probs = [
                item[0] for item in meta_info["output_token_logprobs"]
            ]

            response += cur_response
            response_token_ids += cur_token_ids
            loss_mask += [1] * len(cur_token_ids)
            rollout_log_probs += cur_log_probs
            sample.append_response_tokens(
                args,
                tokens=cur_token_ids,
                log_probs=cur_log_probs,
                trainable=True,
                meta_info=meta_info,
            )

            if last_finish == "length":
                sample.status = Sample.Status.TRUNCATED
                break

            # Slime schedules the trajectories of one rollout batch with
            # ``asyncio.gather`` (sglang_rollout.py). ``env.step`` is blocking
            # I/O -- ``RemoteLDMEnv.step`` writes to a subprocess stdin and
            # blocks on ``stdout.readline`` -- so calling it directly holds the
            # event loop for its full duration and every other trajectory in
            # the batch stops. Measured on this cluster a single ``env.step``
            # takes 4.09 s (docking dominates) and accounts for ~98% of a
            # training step, so the loop spends nearly all of its time
            # serialised on one trajectory.
            #
            # Each sample builds its own env above, so there is no shared
            # state between the threads; the shared GP history file is
            # guarded by flock in ``rl_real_shared.py``.
            step = await asyncio.get_running_loop().run_in_executor(
                None, env.step, cur_response
            )
            last_step = step
            total_reward += step.reward
            sample.metadata["env_steps"].append(step.info)

            if step.observation:
                # Wrap the feedback as its own user turn, then reopen an
                # assistant turn. Without this the text lands right after the
                # policy's end-of-turn marker and the conversation stops being
                # valid chat, which is why every round after the first came
                # back unparseable.
                obs_prefix, obs_suffix = _observation_wrapper(state)
                obs_text = obs_prefix + step.observation + obs_suffix
                obs_token_ids = state.tokenizer(
                    obs_text, add_special_tokens=False
                )["input_ids"]
                response += obs_text
                response_token_ids += obs_token_ids
                loss_mask += [0] * len(obs_token_ids)
                # Environment feedback tokens are non-trainable; pad the
                # rollout log-probs with zeros so their length stays aligned
                # with response_length (the actor slices log_probs ==
                # response_length during training).
                rollout_log_probs += [0.0] * len(obs_token_ids)
                sample.append_response_tokens(
                    args, tokens=obs_token_ids, trainable=False
                )

            if step.done:
                break
    except Exception as exc:  # noqa: BLE001 - keep partial credit on env errors
        sample.metadata["env_error"] = f"{type(exc).__name__}: {exc}"
        if sample.status != Sample.Status.ABORTED:
            sample.status = Sample.Status.FAILED
    else:
        if last_step is not None and last_step.done:
            sample.status = Sample.Status.COMPLETED
        else:
            sample.status = Sample.Status.TRUNCATED

    sample.response = response
    sample.response_length = len(response_token_ids)
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = rollout_log_probs
    sample.reward = total_reward
    sample.metadata["env_total_reward"] = total_reward
    if last_step is not None:
        sample.metadata["env_terminated"] = last_step.terminated
        sample.metadata["env_truncated"] = last_step.truncated
        sample.metadata["env_incumbent"] = last_step.info.get("incumbent")
    _close = getattr(env, "close", None)
    if _close is not None:
        _close()
    return sample


def reward_func(args, sample, **kwargs) -> float:
    """Slime custom reward function; reads the reward filled during rollout."""

    if sample.reward is not None:
        return float(sample.reward)
    return float(sample.metadata.get("env_total_reward", 0.0))


__all__ = ["generate", "reward_func"]
