#!/bin/bash
#SBATCH -J nccl-pp-perf-n1g2        # Job name
#SBATCH -p cscc-gpu-p               # GPU partition
#SBATCH -q cscc-gpu-qos             # QoS
#SBATCH -N 1                        # Number of nodes
#SBATCH --ntasks-per-node=1         # Single task (one rank)
#SBATCH --cpus-per-task=1           # CPU cores per task
#SBATCH --gres=gpu:2                # 2 GPUs on this node
#SBATCH --mem=16G                   # Memory per node
#SBATCH -t 00:05:00                 # Wall time
#SBATCH -o running.out              # Stdout file
#SBATCH -e running.out              # Stderr file

set -euo pipefail
set -x

########################################
# nccl-tests binary & output
########################################
NCCL_HOME="$HOME/calculon/3rdparty/nccl-tests"
NCCL_PP_PERF="$NCCL_HOME/build/sendrecv_perf"
OUTDIR="../nccl_tests"
mkdir -p "$OUTDIR"
JSON="$OUTDIR/nccl_pp_perf.n1g2.json"

echo "[DBG] starting 1-node / 2-GPU intra-node sendrecv test (NVLink/PCIe)"

########################################
# Launch a single rank using 2 GPUs (intra-node only)
########################################
srun -N1 -n1 --gres=gpu:2 --cpu-bind=none \
  /bin/bash -lc '
    set -euo pipefail

    module load nvidia/cuda/12.0

    export NCCL_HOME=$HOME/.conda/envs/megatron-lm/lib/python3.12/site-packages/nvidia/nccl
    export LD_LIBRARY_PATH=$NCCL_HOME/lib:/apps/local/anaconda3/lib:${LD_LIBRARY_PATH:-}

    echo "[INFO] RANK=${SLURM_PROCID} LOCAL_RANK=${SLURM_LOCALID} CVD=${CUDA_VISIBLE_DEVICES}"

    # Force intra-node path: disable IB so we measure NVLink/PCIe only
    export NCCL_IB_DISABLE=1
    unset NCCL_NET

    exec '"$NCCL_PP_PERF"' -b 8 -e 64M -f 2 -g 2 -J '"$JSON"'
  '

echo "[$(date)] Done. JSON saved under: $JSON"
