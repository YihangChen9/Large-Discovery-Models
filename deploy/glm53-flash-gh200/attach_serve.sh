#!/bin/bash
# Start the GLM-5.3-Flash server on one node of an allocation you already hold.
#
#   bash attach_serve.sh <jobid> <node>
#
# Attaching to a running allocation rather than submitting a fresh job avoids
# queueing behind it, but it also means the node may be shared. The GPU check
# below is not a formality: a neighbour holding even a few GB will not stop
# vLLM from starting, it will stop it ~10 minutes later during weight load,
# by which point the failure looks like a model problem.
set -u

JOBID=${1:?usage: attach_serve.sh <jobid> <node>}
NODE=${2:?usage: attach_serve.sh <jobid> <node>}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

: "${GLM53_MODEL:?set GLM53_MODEL to the weights directory}"
: "${GLM53_SIF:?set GLM53_SIF to the .sif path}"

# Per-GPU, not the node total: a node with 3 idle cards and 1 busy one reads
# as mostly free in aggregate and still cannot host TP4.
BUSY=$(srun --overlap --jobid="$JOBID" -w "$NODE" --ntasks=1 --gres=gpu:4 \
    --job-name=glm53-probe \
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk '$1 > 2000 {n++} END {print n+0}')

if [ "$BUSY" != "0" ]; then
    echo "FATAL: $NODE has $BUSY GPU(s) above 2 GB in use; refusing to start" >&2
    exit 7
fi
echo "[attach] $NODE: all 4 GPUs free, starting vLLM under job $JOBID"

exec srun --overlap --jobid="$JOBID" -w "$NODE" \
    --ntasks=1 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=64 \
    --job-name=glm53-vllm --cpu-bind=none \
    env GLM53_MODEL="$GLM53_MODEL" GLM53_SIF="$GLM53_SIF" \
        PORT="${PORT:-8383}" MAXLEN="${MAXLEN:-32768}" MAXSEQS="${MAXSEQS:-32}" \
    bash "$HERE/serve_vllm.sh"
