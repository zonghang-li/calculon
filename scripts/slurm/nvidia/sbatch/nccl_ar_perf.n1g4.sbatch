#!/bin/bash
#SBATCH -J nccl-ar-perf-n1g4
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem=16G
#SBATCH -t 00:05:00
#SBATCH -o running.out
#SBATCH -e running.out

set -euo pipefail
set -x

NCCL_HOME="$HOME/calculon/3rdparty/nccl-tests"
NCCL_AR_PERF="$NCCL_HOME/build/all_reduce_perf"
OUTDIR="../nccl_tests"
mkdir -p "$OUTDIR"
JSON="$OUTDIR/nccl_ar_perf.n1g4.json"

srun --mpi=pmi2 -N 1 --ntasks-per-node=4 \
     --gres=gpu:4 --cpu-bind=cores \
  /bin/bash -lc '
    set -euo pipefail

    module load nvidia/cuda/12.0

    export NCCL_HOME=$HOME/.conda/envs/megatron-lm/lib/python3.12/site-packages/nvidia/nccl
    export LD_LIBRARY_PATH=$NCCL_HOME/lib:/apps/local/anaconda3/lib:${LD_LIBRARY_PATH:-}
    export NCCL_DEBUG=WARN

    echo "RANK=${SLURM_PROCID} LOCAL_RANK=${SLURM_LOCALID} CVD=${CUDA_VISIBLE_DEVICES}"

    exec '"$NCCL_AR_PERF"' -b 8 -e 2G -f 2 -g 1 -J '"$JSON"'
  '

echo "[$(date)] Done. JSON saved under: $JSON"
