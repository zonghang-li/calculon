#!/bin/bash
#SBATCH -J megatron-lm
#SBATCH -N 2
#SBATCH -A faculty-acc
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH -t 00:30:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.out
#SBATCH -p faculty
#SBATCH --qos zhqos
# Avoid nodes with broken ROCm:
#SBATCH -x auh7-1b-gpu-212,auh7-1b-gpu-255,auh7-1b-gpu-256

set -euo pipefail
set -x

TP=8
PP=2
GLOBAL_BS=64
MICRO_BS=1

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

# Filter extremely noisy driver warnings from stderr
exec 2> >(grep -v 'libibverbs: Warning:' >&2)

# Pick master as the first hostname in the job
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

export MASTER_ADDR MASTER_PORT

# One Slurm task per node; srun will start this block on ALL 3 nodes
srun --ntasks-per-node=1 --nodes="$SLURM_NNODES" \
  apptainer exec \
    --rocm \
    --env HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    --env CUDA_DEVICE_MAX_CONNECTIONS=1 \
    --env TMPDIR=/tmp \
    --env XDG_CACHE_HOME=/tmp/.cache \
    --env PYTHONWARNINGS=ignore \
    --env TORCH_CPP_LOG_LEVEL=ERROR \
    --bind /tmp:/tmp \
    --bind "$HOME/data:/data" \
    --bind "$HOME/Stanford-Megatron-LM:/root/Stanford-Megatron-LM" \
    --bind "$HOME:/home/$USER" \
    --pwd /root/Stanford-Megatron-LM \
    "$HOME/images/stanford-megatron-lm.sif" \
    bash -lc '
      NODE_RANK=${SLURM_PROCID}

      torchrun \
        --nnodes='"$SLURM_NNODES"' \
        --nproc_per_node=8 \
        --rdzv_backend=c10d \
        --rdzv_endpoint='"$MASTER_ADDR:$MASTER_PORT"' \
        --node_rank=${NODE_RANK} \
        pretrain_gpt.py \
          --tensor-model-parallel-size '"$TP"' \
          --pipeline-model-parallel-size '"$PP"' \
          --num-layers 24 \
          --hidden-size 4096 \
          --num-attention-heads 32 \
          --ffn-hidden-size 11008 \
          --seq-length 4096 \
          --max-position-embeddings 4096 \
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
          --data-path /data/wiki-gpt2_text_document_text_document \
          --vocab-file /data/gpt2-vocab.json \
          --merge-file /data/gpt2-merges.txt \
          --split 949,50,1 \
          --no-masked-softmax-fusion \
          --no-bias-gelu-fusion \
          --no-bias-dropout-fusion \
          --no-gradient-accumulation-fusion
    '
