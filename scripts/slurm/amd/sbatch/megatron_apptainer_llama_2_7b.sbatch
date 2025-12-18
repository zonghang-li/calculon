#!/bin/bash
#SBATCH -J megatron-lm
#SBATCH -N 2
#SBATCH -A faculty-acc
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=8          # one task per GPU
#SBATCH --mem=128G
#SBATCH -t 00:30:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.out
#SBATCH -p faculty
#SBATCH --qos zhqos
# Avoid nodes with broken ROCm:
#SBATCH -x auh7-1b-gpu-212,auh7-1b-gpu-214,auh7-1b-gpu-215,auh7-1b-gpu-255,auh7-1b-gpu-256,auh7-1b-gpu-196,auh7-1b-gpu-197,auh7-1b-gpu-303,auh7-1b-gpu-304,auh7-1b-gpu-200,auh7-1b-gpu-201,auh7-1b-gpu-297,auh7-1b-gpu-298

set -euo pipefail
set -x

TP=8
PP=2
GLOBAL_BS=64
MICRO_BS=1
ACT_RECOMP=none   # none | full | attn_only
FP=""             # if set to "float16", we add --fp16

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
    --act-recomp)
      ACT_RECOMP="$2"
      shift 2
      ;;
    --fp16)
      FP="float16"
      shift 1
      ;;
    --help|-h)
      echo "Usage: sbatch [sbatch options] -- $0 \\"
      echo "  [--tp N] [--pp N] [--gbs N] [--mbs N] \\"
      echo "  [--act-recomp none|full|attn_only] [--fp16]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage."
      exit 1
      ;;
  esac
done

case "$ACT_RECOMP" in
  none)
    RECOMP_FLAGS=""
    ;;
  full)
    RECOMP_FLAGS="--recompute-granularity full --recompute-method uniform --recompute-num-layers 1"
    ;;
  attn_only)
    RECOMP_FLAGS="--recompute-activations"
    ;;
  *)
    echo "Invalid --act-recomp value: $ACT_RECOMP (expected none|full|attn_only)"
    exit 1
    ;;
esac

FP_ARGS=""
if [[ "${FP:-}" == "float16" ]]; then
  FP_ARGS="--fp16"
fi

echo "Using config: TP=${TP}, PP=${PP}, GLOBAL_BS=${GLOBAL_BS}, MICRO_BS=${MICRO_BS}, ACT_RECOMP=${ACT_RECOMP}, FP=${FP:-none}"
echo "SLURM_NNODES=${SLURM_NNODES}, SLURM_NTASKS=${SLURM_NTASKS}, NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-unset}"

# Filter extremely noisy driver warnings from stderr
exec 2> >(grep -v 'libibverbs: Warning:' >&2)

# Master address/port for env:// init
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

# One Python process per Slurm task (per GPU)
srun \
  --kill-on-bad-exit=1 \
  apptainer exec \
    --rocm \
    --env MASTER_ADDR="${MASTER_ADDR}" \
    --env MASTER_PORT="${MASTER_PORT}" \
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
      set -euo pipefail
      set -x

      # Map Slurm to torch.distributed env://
      export RANK=${SLURM_PROCID}
      export LOCAL_RANK=${SLURM_LOCALID}
      export WORLD_SIZE=${SLURM_NTASKS}

      if rocm-smi --showpidgpus | grep -qE "^PID[[:space:]][0-9]"; then
        echo "ERROR: Detected processes using GPUs."
        rocm-smi --showpidgpus || true
        exit 1
      fi

      python pretrain_gpt.py \
        --tensor-model-parallel-size '"${TP}"' \
        --pipeline-model-parallel-size '"${PP}"' \
        --num-layers 24 \
        --hidden-size 4096 \
        --num-attention-heads 32 \
        --ffn-hidden-size 11008 \
        --seq-length 4096 \
        --max-position-embeddings 4096 \
        --micro-batch-size '"${MICRO_BS}"' \
        --global-batch-size '"${GLOBAL_BS}"' \
        --train-iters 5 \
        --lr 6.0e-4 \
        --min-lr 6.0e-5 \
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
        --no-gradient-accumulation-fusion \
        '"${RECOMP_FLAGS}"' \
        '"${FP_ARGS}"'
    '
