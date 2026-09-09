#!/usr/bin/env python3
"""Guard the three fixes in this PR against silent regression.

All three failures shared a property: the code kept running and produced
plausible output, so nothing surfaced until a training run had already burned
hours. None of them would be caught by the existing tests, because none of
them raise.

  1. bridge.py -- `env.step` called directly inside the `async def generate`
     body holds the event loop, serialising a rollout batch that reads as
     concurrent. Must go through an executor.
  2. parsing.py -- locating the end of a JSON object by searching for the last
     `}` swallows trailing text and discards otherwise-good replies. Must use
     a real decoder that reports where the first object ends.
  3. prompts.py -- the prompt must show an instance of the expected reply, not
     only its JSON Schema, or the model answers with the schema.

Standard library only, AST-based, no repo imports. Run `--self-test` to check
the checker against known-good and known-bad snippets.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- check 1

def _blocking_env_step_calls(tree: ast.AST) -> list[int]:
    """Line numbers where `env.step(...)` is *called* inside an async def.

    Passing the method as a value -- `run_in_executor(None, env.step, x)` --
    is the fix, and appears in the AST as a bare Attribute rather than the
    func of a Call, so it is not reported.
    """
    bad: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "step"
                and isinstance(inner.func.value, ast.Name)
                and inner.func.value.id == "env"
            ):
                bad.append(inner.lineno)
    return bad


# ---------------------------------------------------------------- check 2

def _rfind_brace_in(tree: ast.AST, func_name: str) -> list[int]:
    """Line numbers where `func_name` locates JSON's end with `.rfind("}")`."""
    bad: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != func_name:
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "rfind"
                and inner.args
                and isinstance(inner.args[0], ast.Constant)
                and inner.args[0].value == "}"
            ):
                bad.append(inner.lineno)
    return bad


def _calls_raw_decode(tree: ast.AST, func_name: str) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != func_name:
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "raw_decode"
            ):
                return True
    return False


# ---------------------------------------------------------------- check 3

def _defines(tree: ast.AST, name: str) -> bool:
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
        for n in ast.walk(tree)
    )


def _references(tree: ast.AST, name: str) -> bool:
    """True if `name` is used somewhere other than its own definition."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            return True
    return False


# ---------------------------------------------------------------- driver

def _parse(rel: str) -> ast.AST | None:
    p = ROOT / rel
    if not p.exists():
        print(f"  SKIP {rel} (not in this checkout)")
        return None
    return ast.parse(p.read_text(encoding="utf-8"), filename=str(p))


def run_checks() -> int:
    failures = 0

    tree = _parse("rl/ldm_rl/bridge.py")
    if tree is not None:
        lines = _blocking_env_step_calls(tree)
        if lines:
            print(
                "FAIL rl/ldm_rl/bridge.py: env.step() called directly inside "
                f"an async def at line(s) {lines}.\n"
                "     That blocks the event loop for the whole call, so the "
                "trajectories asyncio.gather is running make no progress.\n"
                "     Use: await asyncio.get_running_loop()"
                ".run_in_executor(None, env.step, ...)"
            )
            failures += 1
        else:
            print("ok   bridge.py: env.step does not block the event loop")

    tree = _parse("ldm_tts/transport/parsing.py")
    if tree is not None:
        fn = "extract_json_object_text"
        lines = _rfind_brace_in(tree, fn)
        if lines:
            print(
                f"FAIL ldm_tts/transport/parsing.py: {fn} locates the end of "
                f"the object with rfind('}}') at line(s) {lines}.\n"
                "     That spans any trailing text (chat-template markers, a "
                "second object) and loses the whole reply."
            )
            failures += 1
        elif not _calls_raw_decode(tree, fn):
            print(
                f"FAIL ldm_tts/transport/parsing.py: {fn} no longer uses "
                "raw_decode; the end of the first object is not being "
                "determined by a real decoder."
            )
            failures += 1
        else:
            print(f"ok   parsing.py: {fn} decodes the first complete object")

    tree = _parse("rl/ldm_rl/prompts.py")
    if tree is not None:
        if not _defines(tree, "_schema_skeleton"):
            print(
                "FAIL rl/ldm_rl/prompts.py: _schema_skeleton is gone. The "
                "prompt would show only the JSON Schema, and the model "
                "answers with the schema instead of an instance."
            )
            failures += 1
        elif not _references(tree, "_schema_skeleton"):
            print(
                "FAIL rl/ldm_rl/prompts.py: _schema_skeleton is defined but "
                "never used, so the prompt does not carry the example."
            )
            failures += 1
        else:
            print("ok   prompts.py: the prompt carries an instance skeleton")

    return failures


# ---------------------------------------------------------------- self-test

GOOD_BRIDGE = """
import asyncio
async def generate(args, sample):
    env = make()
    step = await asyncio.get_running_loop().run_in_executor(
        None, env.step, "x")
    return step
"""
BAD_BRIDGE = """
async def generate(args, sample):
    env = make()
    step = env.step("x")
    return step
"""
# env.step called from a *sync* helper is not this bug; must not be reported.
SYNC_BRIDGE = """
def helper(env):
    return env.step("x")
"""
GOOD_PARSE = """
import json
def extract_json_object_text(text):
    _, end = json.JSONDecoder().raw_decode(text)
    return text[:end]
"""
BAD_PARSE = """
def extract_json_object_text(text):
    return text[text.find("{"): text.rfind("}") + 1]
"""


def self_test() -> int:
    bad = 0

    def expect(cond: bool, label: str) -> None:
        nonlocal bad
        print(f"  {'ok  ' if cond else 'FAIL'} {label}")
        if not cond:
            bad += 1

    expect(_blocking_env_step_calls(ast.parse(BAD_BRIDGE)) == [4],
           "flags a direct env.step() inside async def")
    expect(_blocking_env_step_calls(ast.parse(GOOD_BRIDGE)) == [],
           "accepts env.step passed to run_in_executor")
    expect(_blocking_env_step_calls(ast.parse(SYNC_BRIDGE)) == [],
           "ignores env.step in a synchronous function")
    expect(_rfind_brace_in(ast.parse(BAD_PARSE), "extract_json_object_text") == [3],
           "flags rfind('}') as the object terminator")
    expect(_rfind_brace_in(ast.parse(GOOD_PARSE), "extract_json_object_text") == [],
           "accepts raw_decode")
    expect(_calls_raw_decode(ast.parse(GOOD_PARSE), "extract_json_object_text"),
           "detects raw_decode")
    expect(not _calls_raw_decode(ast.parse(BAD_PARSE), "extract_json_object_text"),
           "does not hallucinate raw_decode")
    expect(_defines(ast.parse("def _schema_skeleton(s): pass"), "_schema_skeleton"),
           "detects the skeleton helper")
    expect(not _defines(ast.parse("def other(): pass"), "_schema_skeleton"),
           "does not hallucinate the skeleton helper")

    print(f"\nself-test: {'all checks behave correctly' if not bad else f'{bad} broken'}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true",
                    help="check the checker against known-good/bad snippets")
    args = ap.parse_args()

    if args.self_test:
        return 1 if self_test() else 0

    print("rollout invariants")
    n = run_checks()
    if n:
        print(f"\n{n} check(s) failed")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
