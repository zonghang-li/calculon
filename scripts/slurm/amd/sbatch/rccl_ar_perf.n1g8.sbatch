#!/bin/bash
#SBATCH -J rccl-ar-perf-n1g8
#SBATCH -p faculty
#SBATCH -q zhqos
#SBATCH -A faculty-acc
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:8
#SBATCH --mem=16G
#SBATCH -t 00:05:00
#SBATCH -o running.out
#SBATCH -e running.out
# Avoid nodes with broken ROCm:
#SBATCH -x auh7-1b-gpu-212,auh7-1b-gpu-255,auh7-1b-gpu-256

set -euo pipefail

# ---- paths & outputs ----
RCCL_HOME="$HOME/calculon/3rdparty/rccl-tests"
RCCL_AR_PERF="$RCCL_HOME/build/all_reduce_perf"
OUTDIR="../rccl_tests"
mkdir -p "$OUTDIR"
JSON="$OUTDIR/rccl_ar_perf.n1g8.json"

srun --mpi=pmix -N 1 --ntasks-per-node=8 \
     --gres=gpu:8 --gpus-per-task=1 \
     --gpu-bind=closest --cpu-bind=none \
     --export=HOME \
  /bin/bash -lc '
    export PATH=/opt/rocm/bin:/usr/bin:/bin
    export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/hip/lib:/opt/rocm/hsa/lib:/opt/rocm/rccl/lib:/vast/apps/ompi-5.0.8/lib:/vast/apps/ucx/rocm633/1.18.1/lib
    export OMPI_MCA_pml=ucx
    export OMPI_MCA_osc=ucx
    export OMPI_MCA_btl="^openib,tcp"
    export UCX_TLS=rc,sm,self
    export NCCL_DEBUG=WARN
    unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES HSA_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
    exec "'"$RCCL_AR_PERF"'" -b 8 -e 2G -f 2 -g 1 -Z json -x "'"$JSON"'"
  '

echo "[$(date)] Done. JSON saved to: $JSON"
