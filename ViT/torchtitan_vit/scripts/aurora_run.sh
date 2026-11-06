#!/bin/bash
#PBS -A <project>
#PBS -q debug
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare
#PBS -l select=1
#PBS -j oe
#
# Run the 3D ViT benchmark on Aurora: one rank per GPU tile, on every node of
# the job. The same script works both ways:
#
#   On a compute node (qsub -I), from the torchtitan_vit directory:
#     VIT_VENV=/path/venv scripts/aurora_run.sh <tiles per node> <config>
#     VIT_VENV=/path/venv scripts/aurora_run.sh 1 vit3d_debug
#     VIT_VENV=/path/venv VIT_DATA=synthetic VIT_CP=12 scripts/aurora_run.sh 12 vit3d_scaling
#
#   As a batch job, submitted from the torchtitan_vit directory (set the
#   #PBS -A line above first; arguments go in -v as TILES and CONFIG):
#     qsub -v VIT_VENV=/path/venv,TILES=1,CONFIG=vit3d_debug scripts/aurora_run.sh
#     qsub -l select=2 -q debug-scaling \
#       -v VIT_VENV=/path/venv,VIT_DATA=synthetic,VIT_CP=24 scripts/aurora_run.sh
#
# Defaults: 12 tiles per node, config vit3d_scaling. Configs are the functions
# in vit3d/recipes.py; VIT_* variables (problem size, parallelism, data) reach
# every rank, see vit3d_scaling. VIT_VENV must hold a PyTorch XPU nightly and
# TorchTitan at the pinned commit (README.md). Output goes to
# outputs/<config>_<tiles>tiles_<time>/ unless VIT_OUT is set.
set -eo pipefail

TILES=${1:-${TILES:-12}}
CONFIG=${2:-${CONFIG:-vit3d_scaling}}
if (( TILES < 1 || TILES > 12 )); then
  echo "tiles per node must be 1-12 (an Aurora node has 12 tiles)" >&2; exit 1
fi
if [[ -z ${VIT_VENV} ]]; then
  echo "set VIT_VENV to the venv with the PyTorch XPU nightly and TorchTitan" >&2; exit 1
fi
# PBS runs a copy of this script, so locate the code from the submit directory.
if [[ ${PBS_ENVIRONMENT:-} == PBS_BATCH ]]; then
  HERE=${PBS_O_WORKDIR}
else
  HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
if [[ ! -f ${HERE}/vit3d/recipes.py ]]; then
  echo "cannot find vit3d/ in ${HERE}; run or submit from the torchtitan_vit directory" >&2; exit 1
fi
cd "${HERE}"
OUT=${VIT_OUT:-${HERE}/outputs/${CONFIG}_${TILES}tiles_$(date +%Y%m%d_%H%M%S)}

# Environment. The frameworks module only provides Python 3.12; its packages
# are kept out (unset PYTHONPATH, venv without system site-packages), and the
# venv's Intel runtime (libsycl, libur_loader) must come before the system
# oneAPI on LD_LIBRARY_PATH, or torch fails to import.
type module >/dev/null 2>&1 || source /etc/profile
module load frameworks
unset PYTHONPATH
source "${VIT_VENV}/bin/activate"
export LD_LIBRARY_PATH="${VIT_VENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${HERE}"

# Rendezvous for torch.distributed.
if [[ -n ${PBS_NODEFILE:-} ]]; then
  NNODES=$(sort -u "${PBS_NODEFILE}" | wc -l)
  export MASTER_ADDR=$(head -n 1 "${PBS_NODEFILE}")
else
  NNODES=1
  export MASTER_ADDR=$(hostname)
fi
export MASTER_PORT=${MASTER_PORT:-29500}
export WORLD_SIZE=$(( NNODES * TILES ))
export VIT_CONFIG=${CONFIG} VIT_OUT_DIR=${OUT}

# One CPU range per tile, on the socket local to that tile (tiles 0-5 on
# socket 0, 6-11 on socket 1); each rank's loader workers share its range.
RANGES=(1-8 9-16 17-24 25-32 33-40 41-48 53-60 61-68 69-76 77-84 85-92 93-100)
BIND="list:$(IFS=:; echo "${RANGES[*]:0:${TILES}}")"

mkdir -p "${OUT}"
{
  echo "config=${CONFIG} nodes=${NNODES} tiles/node=${TILES} ranks=${WORLD_SIZE}"
  echo "output=${OUT}"
  echo "torch=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
  env | grep '^VIT_' | sort || true
} | tee "${OUT}/run_info.txt"

# PALS gives each rank its global and local rank; LOCAL_RANK picks the tile.
mpiexec -n "${WORLD_SIZE}" --ppn "${TILES}" --cpu-bind="${BIND}" \
  bash -c 'export RANK=${PALS_RANKID} LOCAL_RANK=${PALS_LOCAL_RANKID};
           exec python -m torchtitan.train --module vit3d.recipes \
             --config "${VIT_CONFIG}" --output-dir "${VIT_OUT_DIR}"' \
  2>&1 | tee "${OUT}/train.log"
