#!/bin/bash
#SBATCH -J rccl-pp-perf-n1g2-intra-numa
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

# ---- toolchain roots (adjust if different) ----
MPI_HOME="${MPI_HOME:-/vast/apps/ompi-5.0.8}"
UCX_HOME="${UCX_HOME:-/vast/apps/ucx/rocm633/1.18.1}"
ROCM_HOME="${ROCM_HOME:-/opt/rocm}"

# ---- rccl-tests binary & output ----
RCCL_SR_PERF="${RCCL_SR_PERF:-$HOME/calculon/3rdparty/rccl-tests/build/sendrecv_perf}"
OUTDIR="../rccl_tests"
mkdir -p "$OUTDIR"
JSON="$OUTDIR/rccl_pp_perf.n1g2.intra.json"

echo "[DBG] starting 1-node / 2-GPU (same NUMA) intra-node test"

# ---------- choose two GPUs on the same NUMA node ----------
choose_two_gpus_same_numa() {
  if [ -n "${GPU_PAIR_OVERRIDE:-}" ]; then echo "$GPU_PAIR_OVERRIDE"; return 0; fi
  mapfile -t gpu_numa < <(
    for d in /sys/class/drm/card[0-9]*; do
      [ -e "$d/device/numa_node" ] || continue
      idx="${d##*card}"
      numa=$(cat "$d/device/numa_node" 2>/dev/null || echo -1)
      [ "$numa" -ge 0 ] && echo "${idx}:${numa}"
    done | sort -t: -k2,2n -k1,1n
  )
  [ "${#gpu_numa[@]}" -gt 0 ] || { echo "[ERR] cannot read GPU/NUMA map"; return 1; }

  allowed="${SLURM_JOB_GPUS:-}"
  declare -A allowed_set=()
  if [ -n "$allowed" ]; then IFS=',' read -r -a arr <<< "$allowed"; for a in "${arr[@]}"; do allowed_set["$a"]=1; done; fi

  declare -A by_numa=()
  for pair in "${gpu_numa[@]}"; do
    g="${pair%%:*}"; n="${pair##*:}"
    [ -n "$allowed" ] && [ -z "${allowed_set[$g]:-}" ] && continue
    by_numa["$n"]+="${by_numa[$n]:+ }$g"
  done

  for n in "${!by_numa[@]}"; do
    read -r -a glist <<< "${by_numa[$n]}"
    if [ "${#glist[@]}" -ge 2 ]; then echo "${glist[0]},${glist[1]}"; return 0; fi
  done
  echo "[ERR] no NUMA node has >=2 GPUs among allowed set {${allowed:-all}}"; return 2
}

GPU_PAIR="$(choose_two_gpus_same_numa)"
echo "[SEL] GPU pair (same NUMA) = ${GPU_PAIR}"

srun -N1 -n1 --gres=gpu:8 --gpus-per-task=8 --cpu-bind=none bash -lc '
  set -e
  unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  export HIP_VISIBLE_DEVICES="0,1"            # use the same-NUMA pair you picked
  export NCCL_IB_DISABLE=1; unset NCCL_NET
  export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader  # shm path; avoids UCX mismatch

  RCCL_SR_PERF=${RCCL_SR_PERF:-$HOME/calculon/3rdparty/rccl-tests/build/sendrecv_perf}
  OUTDIR=./output; mkdir -p "$OUTDIR"
  CSV="$OUTDIR/rccl_pp_perf.n1g2.intra.single_rank.csv"

  # quick visibility probe
  which rocminfo >/dev/null 2>&1 && rocminfo | sed -n "1,20p" || true

  exec "'"$RCCL_SR_PERF"'" -b 8 -e 64M -f 2 -g 2 -Z json -x "'"$JSON"'"
'

echo "[$(date)] Done. JSON saved to: $JSON"
