#!/usr/bin/env bash
# Functional parallelism matrix on CPU/gloo: every layout must reproduce the
# single-rank loss curve. Usage: PYTHON=/path/to/venv/bin/python tests/run_cpu_matrix.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
TORCHRUN="$(dirname "$PYTHON")/torchrun"
OUT=${OUT:-/tmp/vit3d_cpu_matrix}
mkdir -p "$OUT"
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo0}

port=29600
run() {  # run <config> <nranks> [rank to log: the last PP stage prints the loss]
  port=$((port + 1))
  "$TORCHRUN" --nnodes 1 --nproc_per_node="$2" --master_addr 127.0.0.1 \
    --master_port $port --local-ranks-filter "${3:-0}" tests/cpu_train.py --module vit3d.recipes --config "$1" \
    --output-dir "$OUT/$1" > "$OUT/$1.log" 2>&1
  local losses
  losses=$(grep -a "step:" "$OUT/$1.log" | sed -E 's/.*loss: +(-?[0-9.]+).*/\1/' | tr '\n' ' ')
  printf '%-30s ranks=%s  loss: %s\n' "$1" "$2" "${losses:-FAILED (see $OUT/$1.log)}"
}

[[ -d "$OUT/seed/checkpoint/step-0" ]] || "$TORCHRUN" --nnodes 1 --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 29599 \
  tests/cpu_train.py --module vit3d.recipes --config vit3d_debug_seed_checkpoint \
  --output-dir "$OUT/seed" > "$OUT/seed.log" 2>&1 || { echo "seed checkpoint failed"; exit 1; }
export VIT_SEED_CHECKPOINT="$OUT/seed/checkpoint/step-0"

# Optional args select a subset, e.g. tests/run_cpu_matrix.sh pp cp_ulysses
ranks() {
  case $1 in debug|one_sample) echo 1 ;; 4d) echo 8 ;; *) echo 2 ;; esac
}
for case in ${@:-debug one_sample fsdp ddp tp pp cp_ulysses cp_allgather_kv 4d}; do
  config=vit3d_debug_$case; [[ $case == debug ]] && config=vit3d_debug
  log_rank=0; [[ $case == pp ]] && log_rank=1
  run "$config" "$(ranks "$case")" "$log_rank"
done
