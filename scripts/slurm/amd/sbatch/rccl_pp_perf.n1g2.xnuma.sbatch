#!/bin/bash
#SBATCH -J rccl-pp-perf-n1g2-xnuma
#SBATCH -p faculty
#SBATCH -q zhqos
#SBATCH -A faculty-acc
#SBATCH -N 1
#SBATCH --ntasks=1
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
JSON="$OUTDIR/rccl_pp_perf.n1g2.xnuma.json"

echo "[DBG] starting 1-node / 2-GPU (cross NUMA) intra-node test (alloc 8 GPUs)"

# ---------- helpers ----------
expand_allowed_list() {
  # Expand SLURM_JOB_GPUS tokens like "0,1,2-5,7"
  local raw="$1" out=() tok a b
  IFS=',' read -r -a toks <<< "$raw"
  for tok in "${toks[@]}"; do
    if [[ "$tok" =~ ^[0-9]+-[0-9]+$ ]]; then
      a="${tok%-*}"; b="${tok#*-}"
      for i in $(seq "$a" "$b"); do out+=("$i"); done
    else
      out+=("$tok")
    fi
  done
  printf "%s " "${out[@]}"
}

# ---- enumerate allowed GPUs -> NUMA + PCI BDF ----
declare -A GPU_NUMA GPU_BDF
allowed_raw="${SLURM_JOB_GPUS:-}"

# Fallback: if SLURM_JOB_GPUS is empty, assume all visible cards
if [ -z "$allowed_raw" ]; then
  allowed_raw="$(ls /sys/class/drm/card[0-9]* 2>/dev/null | sed 's/.*card//' | tr '\n' ',' | sed 's/,$//')"
fi

read -r -a ALLOWED_ARR <<< "$(expand_allowed_list "$allowed_raw")"

if [ "${#ALLOWED_ARR[@]}" -lt 2 ]; then
  echo "[ERR] Need at least 2 GPUs in allocation; got: {$allowed_raw}"
  exit 1
fi

echo "[DBG] Allowed GPU ordinals from Slurm: {$(IFS=,; echo "${ALLOWED_ARR[*]}")}"
for g in "${ALLOWED_ARR[@]}"; do
  nfile="/sys/class/drm/card${g}/device/numa_node"
  dpath="$(readlink -f "/sys/class/drm/card${g}/device" 2>/dev/null || true)"
  bdf="$(basename "$dpath" 2>/dev/null || echo '?')"
  numa="$(cat "$nfile" 2>/dev/null || echo -1)"
  GPU_NUMA["$g"]="$numa"
  GPU_BDF["$g"]="$bdf"
done

echo "[DBG] Allowed GPU topology (ordinal -> NUMA,PCI):"
for g in "${ALLOWED_ARR[@]}"; do
  echo "  - GPU $g : NUMA=${GPU_NUMA[$g]} PCI=${GPU_BDF[$g]}"
done

# ---- select a cross-NUMA pair from allowed set ----
if [ -n "${GPU_PAIR_OVERRIDE:-}" ]; then
  GPU_PAIR="$GPU_PAIR_OVERRIDE"
  echo "[SEL] GPU_PAIR_OVERRIDE in effect: $GPU_PAIR"
else
  GPU_PAIR=""
  for ((i=0; i<${#ALLOWED_ARR[@]}; ++i)); do
    gi="${ALLOWED_ARR[$i]}"
    for ((j=i+1; j<${#ALLOWED_ARR[@]}; ++j)); do
      gj="${ALLOWED_ARR[$j]}"
      if [ "${GPU_NUMA[$gi]}" -ge 0 ] && [ "${GPU_NUMA[$gj]}" -ge 0 ] && [ "${GPU_NUMA[$gi]}" != "${GPU_NUMA[$gj]}" ]; then
        GPU_PAIR="${gi},${gj}"
        break 2
      fi
    done
  done
  if [ -z "$GPU_PAIR" ]; then
    echo "[ERR] No cross-NUMA pair inside allowed set {$(IFS=,; echo "${ALLOWED_ARR[*]}")}"
    echo "[HINT] Try a different node or broader allocation; or set GPU_PAIR_OVERRIDE=\"a,b\" if you know a cross-NUMA pair."
    exit 2
  fi
fi

IFS=, read -r G0 G1 <<< "$GPU_PAIR"
echo "[SEL] Cross-NUMA pair selected: $GPU_PAIR"
echo "     GPU $G0 : NUMA=${GPU_NUMA[$G0]} PCI=${GPU_BDF[$G0]}"
echo "     GPU $G1 : NUMA=${GPU_NUMA[$G1]} PCI=${GPU_BDF[$G1]}"

# ---- single-rank / 2-GPU run, cgroup-bound to the chosen pair ----
# Use --gpu-bind=map_gpu:$GPU_PAIR so the step exposes only those two GPUs.
BIND_PAIR="$GPU_PAIR" \
srun -N1 -n1 --gres=gpu:8 --gpus-per-task=2 \
     --gpu-bind=map_gpu:${GPU_PAIR} --cpu-bind=none --label /bin/bash -lc '
  set -e
  # Keep visibility simple: rely on gpu-bind + cgroups; optionally pin HIP to the pair too.
  unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  export HIP_VISIBLE_DEVICES="'"$GPU_PAIR"'"

  # Intra-node; avoid IB path
  export NCCL_IB_DISABLE=1
  unset NCCL_NET

  # OMPI transport (single rank; shm or UCX shm)
  if ompi_info --components pml 2>/dev/null | grep -q "mca:pml:ucx"; then
    export OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx UCX_TLS=sm,self
    unset OMPI_MCA_btl
  else
    export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader
    unset OMPI_MCA_osc UCX_TLS
  fi

  export PATH="'"$ROCM_HOME"'/bin:'"$MPI_HOME"'/bin:/usr/bin:/bin"
  export LD_LIBRARY_PATH="'"$MPI_HOME"'/lib:'"$UCX_HOME"'/lib:'"$ROCM_HOME"'/lib:'"$ROCM_HOME"'/lib64:'"$ROCM_HOME"'/hip/lib:'"$ROCM_HOME"'/hsa/lib:'"$ROCM_HOME"'/rccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  echo "[DBG] HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
  (which rocminfo >/dev/null 2>&1 && rocminfo | sed -n "1,20p") || true

  exec "'"$RCCL_SR_PERF"'" -b 8 -e 64M -f 2 -g 2 -Z json -x "'"$JSON"'"
'

echo "[$(date)] Done. JSON saved to: $JSON"
