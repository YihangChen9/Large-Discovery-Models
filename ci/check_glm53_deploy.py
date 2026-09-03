#!/usr/bin/env python3
"""Guard the GLM-5.3-Flash serving flags against being "tidied up".

Every constraint below cost hours to find and each one looks removable to
someone reading the script cold:

  - `--kv-cache-dtype fp8` is the obvious memory win and it makes the server
    fail to start on this model (fp8_ds_mla asserts pe_dim == 64, the MLA here
    is NoPE).
  - `TMPDIR=/tmp` looks like boilerplate. Without it DeepGEMM's JIT cannot
    write its nvcc output, and it fails long after startup looked fine.
  - `--max-num-seqs` looks like a throughput knob to raise. The number of KDA
    state blocks available varies per allocation (512 and 136 both observed);
    256 worked on one and failed outright on the other.
  - `--max-model-len 65536` is what the model supports and does not fit at TP4
    on 96 GB cards.

None of these produce a clear error at the point of the change, which is why
they are worth a check rather than a comment. Also verifies no absolute
cluster paths were left in, since these scripts are meant to be portable.

Standard library only. Run `--self-test` to check the checker itself.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy" / "glm53-flash-gh200"

# Paths specific to the machine this was developed on; portable scripts take
# these from the environment instead.
HOST_PATH = re.compile(r"/lus/lfs1aip2|/home/u6gb|/projects/public/u6gb")


def has_fp8_kv(text: str) -> bool:
    """True if fp8 KV cache is actually enabled (not just mentioned)."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):          # a comment explaining why not
            continue
        if re.search(r"--kv-cache-dtype\s+fp8", stripped):
            return True
    return False


def sets_tmpdir(text: str) -> bool:
    return bool(re.search(r"--env\s+TMPDIR=/tmp", text))


def max_num_seqs_value(text: str) -> int | None:
    """The default MAXSEQS the script falls back to, if it declares one."""
    m = re.search(r"MAXSEQS=\$\{MAXSEQS:-(\d+)\}", text)
    return int(m.group(1)) if m else None


def max_model_len_value(text: str) -> int | None:
    m = re.search(r"MAXLEN=\$\{MAXLEN:-(\d+)\}", text)
    return int(m.group(1)) if m else None


def host_paths(text: str) -> list[int]:
    """1-indexed line numbers carrying a machine-specific absolute path."""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith("#"):
            continue
        if HOST_PATH.search(line):
            out.append(i)
    return out


def run_checks() -> int:
    failures = 0
    serve = DEPLOY / "serve_vllm.sh"
    if not serve.exists():
        print(f"SKIP {serve} (not in this checkout)")
        return 0
    text = serve.read_text(encoding="utf-8")

    if has_fp8_kv(text):
        print("FAIL serve_vllm.sh enables --kv-cache-dtype fp8.\n"
              "     That routes to fp8_ds_mla, which asserts pe_dim == 64; this\n"
              "     model's MLA is NoPE, so the server will not start.")
        failures += 1
    else:
        print("ok   KV cache stays bf16 (fp8 would hit the NoPE assertion)")

    if not sets_tmpdir(text):
        print("FAIL serve_vllm.sh does not set TMPDIR=/tmp inside the container.\n"
              "     The host TMPDIR does not exist there and DeepGEMM's JIT nvcc\n"
              "     call fails with 'Could not open output file'.")
        failures += 1
    else:
        print("ok   TMPDIR is redirected to node-local /tmp")

    seqs = max_num_seqs_value(text)
    if seqs is None:
        print("FAIL serve_vllm.sh has no explicit --max-num-seqs default.\n"
              "     vLLM's default (1024) exceeds the KDA state-block budget.")
        failures += 1
    elif seqs > 128:
        print(f"FAIL --max-num-seqs default is {seqs}.\n"
              "     Available KDA state blocks vary per allocation (512 and 136\n"
              "     both observed); 256 failed outright on the smaller one.")
        failures += 1
    else:
        print(f"ok   --max-num-seqs default is {seqs}, below the observed budget")

    mlen = max_model_len_value(text)
    if mlen is None:
        print("FAIL serve_vllm.sh has no explicit --max-model-len default.")
        failures += 1
    elif mlen > 32768:
        print(f"FAIL --max-model-len default is {mlen}.\n"
              "     At TP4 on 96 GB cards the KV reservation squeezes out the\n"
              "     cuBLAS workspace above 32768.")
        failures += 1
    else:
        print(f"ok   --max-model-len default is {mlen}, fits at TP4")

    for name in ("serve_vllm.sh", "attach_serve.sh", "fetch_weights.sh"):
        p = DEPLOY / name
        if not p.exists():
            continue
        lines = host_paths(p.read_text(encoding="utf-8"))
        if lines:
            print(f"FAIL {name} hardcodes a machine-specific path at line(s) "
                  f"{lines}; take it from the environment instead.")
            failures += 1
    if not failures:
        print("ok   no machine-specific absolute paths in the scripts")

    return failures


GOOD = '''
MAXLEN=${MAXLEN:-32768}
MAXSEQS=${MAXSEQS:-32}
exec apptainer exec --nv --env TMPDIR=/tmp "$SIF" vllm serve "$MODEL" \\
  --max-num-seqs "$MAXSEQS"
# Deliberately not set: --kv-cache-dtype fp8 (NoPE MLA)
'''
BAD_FP8 = '''
MAXLEN=${MAXLEN:-32768}
MAXSEQS=${MAXSEQS:-32}
exec apptainer exec --nv --env TMPDIR=/tmp "$SIF" vllm serve "$MODEL" \\
  --kv-cache-dtype fp8
'''
BAD_NO_TMPDIR = '''
MAXLEN=${MAXLEN:-32768}
MAXSEQS=${MAXSEQS:-32}
exec apptainer exec --nv "$SIF" vllm serve "$MODEL"
'''
BAD_SEQS = 'MAXSEQS=${MAXSEQS:-1024}\n'
BAD_LEN = 'MAXLEN=${MAXLEN:-65536}\n'
BAD_PATH = 'MODEL=/lus/lfs1aip2/projects/public/u6gb/models/GLM\n'


def self_test() -> int:
    bad = 0

    def expect(cond: bool, label: str) -> None:
        nonlocal bad
        print(f"  {'ok  ' if cond else 'FAIL'} {label}")
        if not cond:
            bad += 1

    expect(has_fp8_kv(BAD_FP8), "flags an enabled --kv-cache-dtype fp8")
    expect(not has_fp8_kv(GOOD), "accepts fp8 mentioned only in a comment")
    expect(sets_tmpdir(GOOD), "detects TMPDIR=/tmp")
    expect(not sets_tmpdir(BAD_NO_TMPDIR), "notices a missing TMPDIR")
    expect(max_num_seqs_value(GOOD) == 32, "reads the MAXSEQS default")
    expect(max_num_seqs_value(BAD_SEQS) == 1024, "reads a too-large MAXSEQS")
    expect(max_model_len_value(GOOD) == 32768, "reads the MAXLEN default")
    expect(max_model_len_value(BAD_LEN) == 65536, "reads a too-large MAXLEN")
    expect(host_paths(BAD_PATH) == [1], "flags a hardcoded cluster path")
    expect(host_paths(GOOD) == [], "accepts scripts with no absolute paths")

    print(f"\nself-test: {'all checks behave correctly' if not bad else f'{bad} broken'}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true",
                    help="check the checker against known-good/bad snippets")
    args = ap.parse_args()

    if args.self_test:
        return 1 if self_test() else 0

    print("glm53 deploy invariants")
    n = run_checks()
    if n:
        print(f"\n{n} check(s) failed")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
