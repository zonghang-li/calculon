#!/bin/bash
#SBATCH -J nccl-pp-perf-n2g1        # Job name
#SBATCH -p cscc-gpu-p               # GPU partition
#SBATCH -q cscc-gpu-qos             # QoS
#SBATCH -N 2                        # Number of nodes
#SBATCH --ntasks-per-node=1         # 1 rank per node
#SBATCH --cpus-per-task=1           # CPU cores per task
#SBATCH --gres=gpu:1                # 1 GPU per node (2 GPUs total)
#SBATCH --mem=16G                   # Memory per node
#SBATCH -t 00:05:00                 # Wall time
#SBATCH -o running.out              # Stdout file
#SBATCH -e running.out              # Stderr file

set -euo pipefail
set -x

NCCL_HOME="$HOME/calculon/3rdparty/nccl-tests"
NCCL_PP_PERF="$NCCL_HOME/build/sendrecv_perf"
OUTDIR="../nccl_tests"
mkdir -p "$OUTDIR"
JSON="$OUTDIR/nccl_pp_perf.n2g1.json"

echo "[DBG] starting 2-node / 1-GPU-per-node inter-node sendrecv test (IB)"

########################################
# 2 ranks, 2 nodes, 1 GPU each
########################################
srun --mpi=pmi2 -N 2 --ntasks-per-node=1 \
     --gres=gpu:1 --cpu-bind=cores \
  /bin/bash -lc '
    set -euo pipefail

    module load nvidia/cuda/12.0

    # Point NCCL to the conda-installed libnccl.so
    export NCCL_HOME=$HOME/.conda/envs/megatron-lm/lib/python3.12/site-packages/nvidia/nccl
    export LD_LIBRARY_PATH=$NCCL_HOME/lib:/apps/local/anaconda3/lib:${LD_LIBRARY_PATH:-}

    # Inter-node test: allow IB (do not disable it)
    unset NCCL_IB_DISABLE
    unset NCCL_NET

    # 2 ranks total, each rank controls 1 GPU (-g 1)
    exec '"$NCCL_PP_PERF"' -b 8 -e 64M -f 2 -g 1 -J '"$JSON"'
  '

echo "[$(date)] Done. JSON (per-rank) saved under: ${JSON}.*"
