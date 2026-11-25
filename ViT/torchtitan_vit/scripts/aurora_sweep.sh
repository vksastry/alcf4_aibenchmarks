#!/bin/bash
# Throughput sweeps of the 3D ViT on Aurora. Run from an interactive job
# (qsub -I) in the torchtitan_vit directory, venv not activated:
#
#   scripts/aurora_sweep.sh layouts    # each parallelism layout, fixed problem
#   scripts/aurora_sweep.sh model      # model size, fixed context and layout
#   scripts/aurora_sweep.sh context    # context length (volume / patch), fixed model
#   scripts/aurora_sweep.sh all        # the three in turn
#
# Each case is one scripts/aurora_run.sh run (synthetic data by default). The
# metrics of the steps after WARMUP are averaged per run (summary.json in the
# run directory) and appended as a row to <sweep dir>/results.csv; a table is
# printed at the end. A failed case (e.g. out of memory) is recorded and the
# sweep continues.
#
# Everything below can be overridden from the command line, e.g.
#   TILES=12 STEPS=40 FLAVOR=w2304 scripts/aurora_sweep.sh layouts
#   VIT_DATA=hdf5 VIT_DATA_GLOB='/flare/.../*.h5' scripts/aurora_sweep.sh layouts
# Other VIT_* variables (VIT_REMAT, VIT_DATA_DTYPE, ...) apply to every case.
set -uo pipefail

SWEEP=${1:-all}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TILES=${TILES:-12}
STEPS=${STEPS:-30}
WARMUP=${WARMUP:-10}
CASE_TIMEOUT=${CASE_TIMEOUT:-1800}   # seconds per case
export VIT_DATA=${VIT_DATA:-synthetic}

# Fixed problem for the layout sweep, fixed model for the context sweep.
FLAVOR=${FLAVOR:-w1536}
VOLUME=${VOLUME:-192x256x256}
PATCH=${PATCH:-8}
# Layout for the model and context sweeps: Ulysses CP over the node.
SWEEP_LAYOUT=${SWEEP_LAYOUT:-"VIT_CP=${TILES}"}
# Model sizes (vit3d/flavors.py): 92M, 651M, 2.0B, 4.6B parameters.
MODELS=${MODELS:-"w768 w1536 w2304 w3072"}
# Context lengths as volume:patch; tokens = prod(volume / patch). Every count
# below divides by 12 and 24.
CONTEXTS=${CONTEXTS:-"384:16 192x256x256:8 384x256x256:8 384:8 192:4 480:8 768x512x512:8"}

OUT=${SWEEP_OUT:-${HERE}/outputs/sweep_${SWEEP}_${TILES}tiles_$(date +%Y%m%d_%H%M%S)}
CSV=${OUT}/results.csv
mkdir -p "${OUT}"

# Ctrl-C stops the running case (timeout passes the signal to its process
# group, which includes mpiexec and the ranks) and the whole sweep.
CASE_PID=
stop_sweep() {
  echo "sweep interrupted; stopping the running case" >&2
  [[ -n ${CASE_PID} ]] && kill -TERM "${CASE_PID}" 2>/dev/null && wait "${CASE_PID}"
  exit 130
}
trap stop_sweep INT TERM

run_case() {  # run_case <name> [VIT_X=value ...]
  local name=$1; shift
  local dir=${OUT}/${name}
  mkdir -p "${dir}"
  echo "=== ${name}: $*"
  # timeout puts the case in its own process group, out of reach of Ctrl-C;
  # run it in the background so the trap below can stop it.
  env VIT_STEPS="${STEPS}" VIT_WARMUP="${WARMUP}" VIT_OUT="${dir}" \
      VIT_RESULTS_CSV="${CSV}" "$@" \
    timeout "${CASE_TIMEOUT}" "${HERE}/scripts/aurora_run.sh" "${TILES}" vit3d_scaling \
    > "${dir}/console.log" 2>&1 &
  CASE_PID=$!
  wait "${CASE_PID}"
  local status=$?
  CASE_PID=
  # aurora_run.sh writes the CSV row itself; record cases that never got that far.
  if ! grep -q "^${name}," "${CSV}" 2>/dev/null; then
    echo "${name},FAILED (exit ${status}; see ${dir}/console.log)" >> "${CSV}"
  fi
  tail -n 1 "${dir}/console.log"
}

sweep_layouts() {
  local problem=(VIT_FLAVOR="${FLAVOR}" VIT_VOLUME="${VOLUME}" VIT_PATCH="${PATCH}")
  local t=${TILES}
  run_case layout_fsdp${t}           "${problem[@]}"
  run_case layout_ddp${t}            "${problem[@]}" VIT_DP_REPLICATE="${t}" VIT_DP_SHARD=1
  run_case layout_hsdp2x$((t / 2))   "${problem[@]}" VIT_DP_REPLICATE=2
  run_case layout_tp${t}             "${problem[@]}" VIT_TP="${t}"
  run_case layout_cp${t}_ulysses     "${problem[@]}" VIT_CP="${t}"
  run_case layout_cp${t}_allgatherkv "${problem[@]}" VIT_CP="${t}" VIT_CP_BACKEND=allgather_kv
  run_case layout_pp2_fsdp$((t / 2)) "${problem[@]}" VIT_PP=2 VIT_PP_MICROBATCHES=2
  run_case layout_pp4_fsdp$((t / 4)) "${problem[@]}" VIT_PP=4 VIT_PP_MICROBATCHES=4
  run_case layout_tp2_cp$((t / 2))   "${problem[@]}" VIT_TP=2 VIT_CP=$((t / 2))
  run_case layout_tp2_cp2_fsdp$((t / 4)) "${problem[@]}" VIT_TP=2 VIT_CP=2
}

sweep_model() {
  local flavor
  for flavor in ${MODELS}; do
    run_case "model_${flavor}" VIT_FLAVOR="${flavor}" VIT_VOLUME="${VOLUME}" \
      VIT_PATCH="${PATCH}" ${SWEEP_LAYOUT}
  done
}

sweep_context() {
  local context volume patch
  for context in ${CONTEXTS}; do
    volume=${context%%:*}; patch=${context##*:}
    run_case "context_${volume}_p${patch}" VIT_FLAVOR="${FLAVOR}" VIT_VOLUME="${volume}" \
      VIT_PATCH="${patch}" ${SWEEP_LAYOUT}
  done
}

{
  echo "sweep=${SWEEP} tiles=${TILES} steps=${STEPS} warmup=${WARMUP} data=${VIT_DATA}"
  echo "flavor=${FLAVOR} volume=${VOLUME} patch=${PATCH} layout(model/context)=${SWEEP_LAYOUT}"
  echo "models=${MODELS}"
  echo "contexts=${CONTEXTS}"
} | tee "${OUT}/sweep_info.txt"

case ${SWEEP} in
  layouts) sweep_layouts ;;
  model)   sweep_model ;;
  context) sweep_context ;;
  all)     sweep_layouts; sweep_model; sweep_context ;;
  *) echo "usage: $0 [layouts|model|context|all]" >&2; exit 1 ;;
esac

echo
echo "results: ${CSV}"
python3 "${HERE}/tools/summarize_run.py" --table "${CSV}"
