#!/bin/bash
#SBATCH -J nccl-ar-perf-n2g1        # Job name
#SBATCH -p cscc-gpu-p               # GPU partition
#SBATCH -q cscc-gpu-qos             # QoS
#SBATCH -N 2                        # Number of nodes
#SBATCH --ntasks-per-node=1         # Tasks per node (1 ranks per node)
#SBATCH --cpus-per-task=1           # CPU cores per task
#SBATCH --gres=gpu:1                # GPUs per node (1 GPUs per node -> 2 GPUs total)
#SBATCH --mem=16G                   # Memory per node
#SBATCH -t 00:05:00                 # Wall time
#SBATCH -o running.out              # Stdout file
#SBATCH -e running.out              # Stderr file

set -euo pipefail
set -x

NCCL_HOME="$HOME/calculon/3rdparty/nccl-tests"
NCCL_AR_PERF="$NCCL_HOME/build/all_reduce_perf"

# Use a clean, predictable output directory
OUTDIR="../nccl_tests"
mkdir -p "$OUTDIR"
JSON="$OUTDIR/nccl_ar_perf.n2g1.json"

srun --mpi=pmi2 -N 2 --ntasks-per-node=1 \
     --gres=gpu:1 --cpu-bind=cores \
  /bin/bash -lc '
    set -euo pipefail

    module load nvidia/cuda/12.0

    export NCCL_HOME=$HOME/.conda/envs/megatron-lm/lib/python3.12/site-packages/nvidia/nccl
    export LD_LIBRARY_PATH=$NCCL_HOME/lib:/apps/local/anaconda3/lib:${LD_LIBRARY_PATH:-}

    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT,NET

    echo "RANK=${SLURM_PROCID} LOCAL_RANK=${SLURM_LOCALID} CVD=${CUDA_VISIBLE_DEVICES}"

    exec '"$NCCL_AR_PERF"' -b 8 -e 2G -f 2 -g 1 -J '"$JSON"'
  '

echo "[$(date)] Done. JSON saved under: $JSON"
