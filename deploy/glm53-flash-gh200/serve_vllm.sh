#!/bin/bash
# Serve GLM-5.3-Flash with vLLM inside the official arm64/cu129 container.
#
# Runs on a compute node, normally via attach_serve.sh. See README.md for why
# this goes through a container image rather than pip or a source build.
#
# Required:
#   GLM53_MODEL   path to the weights (see fetch_weights.sh)
#   GLM53_SIF     path to the .sif built from the official image
# Optional:
#   PORT          default 8383
#   MAXLEN        default 32768
#   MAXSEQS       default 32   -- read the note below before raising this
set -u

MODEL=${GLM53_MODEL:?set GLM53_MODEL to the weights directory}
SIF=${GLM53_SIF:?set GLM53_SIF to the .sif path}
PORT=${PORT:-8383}

# 64k of KV reservation squeezed out the cuBLAS workspace on 96 GB cards at
# TP4. 32k is what fits; raise it only after checking the server actually
# starts, not just that the flag is accepted.
MAXLEN=${MAXLEN:-32768}

# The KDA linear layers hold one fixed-size state block per concurrent
# sequence, and the number of blocks the engine reports available VARIES
# between allocations -- 512 and 136 were both observed on the same cluster.
# 256 started fine on a 512-block allocation and failed outright on the
# 136-block one, so this default is deliberately far below either. Treat any
# value you pick here as allocation-dependent and verify startup each time.
MAXSEQS=${MAXSEQS:-32}

[ -f "$SIF" ] || { echo "FATAL: no SIF at $SIF" >&2; exit 6; }
[ -d "$MODEL" ] || { echo "FATAL: no weights at $MODEL" >&2; exit 6; }

echo "[serve] host=$(hostname) port=$PORT max_len=$MAXLEN max_seqs=$MAXSEQS tp=4"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# Every compile/download cache is pointed at node-local /tmp, which apptainer
# binds by default. The host TMPDIR (a per-job path on this cluster) does not
# exist inside the container, and DeepGEMM's JIT invokes nvcc with -o into it,
# which fails as "Could not open output file" long after startup looks healthy.
exec apptainer exec --nv \
    --bind "$(dirname "$MODEL")":"$(dirname "$MODEL")" \
    --env TMPDIR=/tmp \
    --env VLLM_CACHE_ROOT=/tmp/glm53_vllm_cache_$USER \
    --env TRITON_CACHE_DIR=/tmp/glm53_triton_$USER \
    --env DG_JIT_CACHE_DIR=/tmp/glm53_dgjit_$USER \
    --env HF_HUB_OFFLINE=1 \
    "$SIF" \
    vllm serve "$MODEL" \
      --served-model-name GLM-5.3-Flash \
      --tensor-parallel-size 4 \
      --max-model-len "$MAXLEN" \
      --gpu-memory-utilization 0.92 \
      --max-num-seqs "$MAXSEQS" \
      --tool-call-parser glm47 \
      --reasoning-parser glm45 \
      --enable-auto-tool-choice \
      --host 0.0.0.0 \
      --port "$PORT"
#
# Deliberately NOT set: --kv-cache-dtype fp8. It routes to the fp8_ds_mla
# kernel, which asserts pe_dim == 64. This model's MLA is NoPE, so the
# assertion fires at startup. KV stays bf16; MLA already compresses to 512
# dims so the memory cost of that is small.
