#!/bin/bash
#SBATCH -J megatron-lm              # Job name
#SBATCH -N 1                        # Number of nodes
#SBATCH --gres=gpu:4                # GPUs per node
#SBATCH --cpus-per-task=8           # CPU cores per task
#SBATCH --ntasks-per-node=1         # One task per node
#SBATCH --mem=128G                  # Memory per node
#SBATCH -t 00:30:00                 # Wall time
#SBATCH --output=slurm.out          # Stdout file
#SBATCH --error=slurm.out           # Stderr file
#SBATCH -p cscc-gpu-p               # Partition
#SBATCH --qos=cscc-gpu-qos          # QoS

set -euo pipefail
set -x

########################################
# Default configuration (can be overridden by CLI flags)
########################################
TP=4
PP=1
GLOBAL_BS=8
MICRO_BS=1

########################################
# Parse command-line overrides
########################################
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tp)
      TP="$2"
      shift 2
      ;;
    --pp)
      PP="$2"
      shift 2
      ;;
    --gbs)
      GLOBAL_BS="$2"
      shift 2
      ;;
    --mbs)
      MICRO_BS="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: sbatch [sbatch options] -- $0 [--tp N] [--pp N] [--gbs N] [--mbs N]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage."
      exit 1
      ;;
  esac
done

echo "Using config: TP=${TP}, PP=${PP}, GLOBAL_BS=${GLOBAL_BS}, MICRO_BS=${MICRO_BS}"

########################################
# Basic environment and logging setup
########################################

# Filter extremely noisy ibverbs warnings from stderr
exec 2> >(grep -v 'libibverbs: Warning:' >&2)

# Pick the first node in the allocation as the rendezvous (master) node
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

# Load shell configuration so that 'conda' is available
if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

# Change to the Megatron-LM project directory
cd /home/zonghang.li/Megatron-LM

# Common environment variables for PyTorch jobs
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONWARNINGS=ignore
export TORCH_CPP_LOG_LEVEL=ERROR
export TMPDIR=/tmp

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

########################################
# Launch torchrun on all nodes via srun
########################################
# One Slurm task per node; srun will run the bash block on all nodes
srun --ntasks-per-node=1 --nodes="$SLURM_NNODES" \
  bash -lc '
    set -euo pipefail

    NODE_RANK=${SLURM_PROCID}

    # Ensure we are in the Megatron-LM directory on each node
    cd /home/zonghang.li/Megatron-LM

    conda activate megatron-lm

    module load nvidia/cuda/12.0

    torchrun \
      --nnodes='"$SLURM_NNODES"' \
      --nproc_per_node=4 \
      --rdzv_backend=c10d \
      --rdzv_endpoint='"$MASTER_ADDR:$MASTER_PORT"' \
      --node_rank=${NODE_RANK} \
      pretrain_gpt.py \
        --tensor-model-parallel-size '"$TP"' \
        --pipeline-model-parallel-size '"$PP"' \
        --num-layers 32 \
        --hidden-size 3200 \
        --num-attention-heads 32 \
        --ffn-hidden-size 8640 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --micro-batch-size '"$MICRO_BS"' \
        --global-batch-size '"$GLOBAL_BS"' \
        --train-iters 10 \
        --lr 6.0e-4 \
        --min-lr 6.0e-5 \
        --lr-decay-style cosine \
        --lr-decay-iters 10 \
        --lr-warmup-fraction 0.1 \
        --weight-decay 1.0e-2 \
        --clip-grad 1.0 \
        --log-interval 1 \
        --timing-log-level 2 \
        --eval-interval 1000000 \
        --data-path "$HOME/data/wiki-gpt2_text_document_text_document" \
        --vocab-file "$HOME/data/gpt2-vocab.json" \
        --merge-file "$HOME/data/gpt2-merges.txt" \
        --split 949,50,1 \
        --no-masked-softmax-fusion \
        --no-bias-gelu-fusion \
        --no-bias-dropout-fusion \
        --no-gradient-accumulation-fusion \
        --transformer-impl local \
        --no-persist-layer-norm
  '
