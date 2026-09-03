# GLM-5.3-Flash on GH200 (aarch64, driver 565 / CUDA 12.7)

An OpenAI-compatible vLLM server for GLM-5.3-Flash on Grace Hopper nodes,
running from the official container image. Used here as the LLM behind
proposal-generating campaigns and IR augmentation.

```bash
GLM53_MODEL=/path/to/GLM-5.3-Flash bash fetch_weights.sh      # ~306 GiB, login node
apptainer pull glm53.sif docker://vllm/vllm-openai:glm53-flash-arm64-cu129

GLM53_MODEL=/path/to/GLM-5.3-Flash GLM53_SIF=/path/to/glm53.sif \
  bash attach_serve.sh <jobid> <node>                          # TP4, port 8383
```

## Why a container and not pip or a source build

Both were tried first. The failures are specific enough to be worth writing
down, because each one wastes hours before it looks like a failure.

**pip.** The vLLM wheels on PyPI (0.27, 0.28) pin torch 2.13+**cu130**. On a
565 driver (CUDA 12.7) that crosses a major version and `cuda_available()` is
simply `False`. GitHub releases do carry `+cu129` aarch64 variants that install
and import cleanly — but **no released version has the `Glm5Next`
architecture**. vLLM then falls back to the transformers backend without
saying so, and dies much later on the KDA parameter `k_conv1d`. The
implementation lives in PR #53906 (93 files, including CUDA kernels), unmerged
at the time of writing.

**Source build of that PR.** Six rounds on this machine, none successful:

| round | wall | what broke |
|---|---|---|
| 1–4 | hours | conda CUDA layout: empty top-level `include`, missing `libnvrtc.so` link name; fixed incrementally |
| 5 | ~1 h | CUTLASS headers report `std::is_signed_v` missing and fold expressions ill-formed |
| 6 | ~1 h | same, after upgrading nvcc to 12.9 — the upgrade was not the problem |

The round-5/6 error reads like a CUDA toolkit version issue and is not:
`-std=c++20` was being passed, but **`-ccbin` was not**, so the host compiler
fell through to the system gcc-7. Chasing the nvcc version was the wrong
diagnosis for two rounds. The image sidesteps all of it.

## Four things that will bite inside the container

Each of these is already handled in `serve_vllm.sh`; they are listed so the
flags do not look arbitrary.

**1. `TMPDIR` must be set to `/tmp`.** This cluster gives each job a TMPDIR
under a host path that does not exist inside the container. DeepGEMM's JIT
shells out to `nvcc -o $TMPDIR/...` and fails with "Could not open output
file" — well after startup already looked healthy.

**2. `--max-num-seqs` must be small, and the right value is not fixed.** KDA
linear layers hold one fixed-size state block per concurrent sequence. The
number of blocks the engine finds available **varies between allocations**:
512 and 136 were both observed on the same cluster. 256 started fine on the
512-block allocation and failed outright on the 136-block one. The default
here is 32. Treat any value as allocation-dependent and check that the server
actually came up.

**3. `--kv-cache-dtype fp8` does not work with this model.** It routes to the
`fp8_ds_mla` kernel, which asserts `pe_dim == 64`. This model's MLA is
**NoPE**, so the assertion fires. KV stays bf16; MLA already compresses to 512
dims, so the cost is small.

**4. `--max-model-len 65536` does not fit at TP4 on 96 GB cards.** The KV
reservation squeezes out the cuBLAS workspace. 32768 is what fits.

## What it looks like when it works

```
TP4 · max-model-len 32768 · gpu-memory-utilization 0.92 · max-num-seqs 32 · KV bf16
FP8 weights 306 GiB, ~77 GiB per card, warm cache load under a minute
OpenAI-compatible on port 8383
```

`--tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice`
separate the chain of thought from the answer. Without them the server still
works, but the reasoning arrives inside `content`, delimited by `</think>`,
and every caller has to strip it.

One version note: vLLM 0.28.0 pins flashinfer 0.6.16.post3, below the
0.6.17 the deployment page asks for. Nothing has required the newer one so
far; if a KDA or sparse-MLA kernel starts complaining, that pin is the first
thing to check.

## Files

| file | what it does |
|---|---|
| `fetch_weights.sh` | pull the 62 shards from ModelScope, verify the count |
| `attach_serve.sh` | per-GPU emptiness check, then start the server on a node of an allocation you hold |
| `serve_vllm.sh` | the container invocation and every flag above |
