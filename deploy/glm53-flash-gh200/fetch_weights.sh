#!/bin/bash
# Download the GLM-5.3-Flash weights from ModelScope.
#
#   GLM53_MODEL=/path/to/GLM-5.3-Flash bash fetch_weights.sh
#
# FP8, ~306 GiB across 62 safetensors shards. Pure sequential network I/O, so
# it is one of the few things that belongs on a login node rather than a
# compute node.
#
# ModelScope's CLI is used instead of huggingface_hub because it does not go
# through hf_xet, whose Rust thread pool has exhausted the process quota on
# this cluster when pulling models of this size.
set -u

DEST=${GLM53_MODEL:?set GLM53_MODEL to the destination directory}
MS=${MODELSCOPE_BIN:-modelscope}
EXPECTED_SHARDS=${EXPECTED_SHARDS:-62}

command -v "$MS" >/dev/null || { echo "FATAL: $MS not on PATH (pip install modelscope)" >&2; exit 3; }
mkdir -p "$DEST"

# Retry loop for network flakiness; the CLI resumes, so a retry is cheap.
for i in $(seq 1 30); do
    "$MS" download --model ZhipuAI/GLM-5.3-Flash --local_dir "$DEST" \
        && { echo "[fetch] done on attempt $i"; break; }
    echo "[fetch] attempt $i failed (rc=$?), retrying"; sleep 15
done

# Count shards rather than trust the exit code: a partial download exits 0 on
# the attempt that happened to complete, and the missing shard only surfaces
# when vLLM fails to load it.
n=$(ls -1 "$DEST"/*.safetensors 2>/dev/null | wc -l)
sz=$(du -sh --apparent-size "$DEST" 2>/dev/null | cut -f1)
echo "[fetch] shards=$n (expected $EXPECTED_SHARDS)  total=$sz"
if [ "$n" -eq "$EXPECTED_SHARDS" ]; then
    echo "[fetch] COMPLETE"
else
    echo "[fetch] INCOMPLETE -- rerun this script, it resumes" >&2
    exit 1
fi
