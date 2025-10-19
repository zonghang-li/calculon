#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# main.sh — CUDA micro-bench orchestration & system.json generator
#
# WHAT THIS SCRIPT DOES
# ---------------------
# Orchestrates build + execution of 5 CUDA micro-benchmarks and aggregates their
# results into a single machine-readable file: ./system.json. It also preserves
# prior GEMM benchmark results (./gemm_bench.json) and provides clean, stepwise
# logging plus per-tool logs in ./builds/logs/.
#
# Included tools (auto-built from *.cu in this directory):
#   - gemm_bench.cu   : FP32 GEMM sweep with micro-batch dimension; prints
#                       progress (stderr) and writes a JSON summary file:
#                       gemm_bench.json (no ASCII tables)
#   - eltwise_fma.cu  : Vector FMA throughput; prints TFLOPS & efficiency table
#   - gpu_info.cu     : Basic GPU/device info (name, CC, FP32 peak, mem theo)
#   - mem_bw.cu       : Device & host memory bandwidths (mem1/mem2)
#   - net_bw.cu       : NCCL collectives + P2P bandwidth/latency (optional)
#
# MAIN FEATURES
# -------------
# • Safe, incremental builds: hashes each .cu; recompiles only on changes
# • Auto-detects CUDA install & GPU compute capability for nvcc flags
# • Consumes gemm_bench’s native JSON (no fragile text parsing)
# • Live GEMM progress/log streaming while the benchmark runs
# • Preserves any existing ./gemm_bench.json (never deletes/overwrites it);
#   fresh GEMM runs execute in a temp dir and are parsed from there
# • Forwards GEMM tuning env-knobs (FAST, DIM_STRIDE, MB_DIMS, etc.) to gemm_bench
# • Robust parsers for eltwise/mem_bw/net_bw outputs
# • Emits ./system.json; pretty-prints via jq when available (compact otherwise)
# • Stores build/runtime logs in ./builds/logs for post-mortem debugging
#
# RUNTIME OPTIONS (set as environment variables)
# ----------------------------------------------
# General orchestration:
#   FORCE_REBUILD=0|1
#       Force recompilation of all .cu files when set to 1. Default: 0.
#
#   RERUN_GEMM_BENCH=true|false|0|1
#       If false/0 AND ./gemm_bench.json exists, reuse it without re-running
#       gemm_bench. If true/1, gemm_bench is executed in a temp dir (existing
#       ./gemm_bench.json is kept intact). Default: false.
#
#   RERUN_NET_BENCH=true|false|0|1
#       If false/0 AND ./net_bw.json exists, reuse it without re-running
#       net_bw. If true/1, net_bw is executed in a temp dir (existing
#       ./net_bw.json is kept intact). Default: false.
#
# GEMM grid & memory knobs (forwarded to gemm_bench.cu):
#   FAST=0|1
#       Coarsen the dimension grid (FAST=1 implies DIM_STRIDE=2 by default and,
#       if MB_DIMS is unset, tests B in {1,2,4,8,16,32} only).
#
#   DIM_STRIDE=<int>
#       Subsample factor for M/N/K candidate lists (e.g., 2 keeps every 2nd dim).
#
#   MB_DIM_STRIDE=<int>
#       Subsample factor for MB_DIMS when provided (default: DIM_STRIDE).
#
#   MB_DIMS="b1,b2,..."
#       Micro-batch sizes to sweep. If unset, defaults to
#       {1,2,4,8,16,32,64,128} (FAST=1 -> {1,2,4,8,16}).
#
#   MEM_CAP_GB=<float>
#       Per-shape memory cap for B*A/B/C combined (approx). If not set, auto-
#       derived as ~80% of free device memory.
#
#   REPS_AUTO_TARGET_MS=<float>
#       Target milliseconds for per-shape timing (auto-tunes repetitions).
#       Default: 35 ms.
#
#   COMMON_DIMS="a,b,c,..."
#       Comma-separated dimension set used for M,N,K simultaneously.
#
#   M_DIMS / N_DIMS / K_DIMS="a,b,c,..."
#       Per-axis overrides for candidate dims (win over COMMON_DIMS).
#
#   TOPK_PRINT=<int>
#       Kept for compatibility; gemm_bench no longer prints ASCII tables.
#
# HOW TO RUN
# ----------
# Basic:
#   ./main.sh
#
# Rebuild everything:
#   FORCE_REBUILD=1 ./main.sh
#
# Don't reuse prior GEMM benchmarks (./gemm_bench.json):
#   RERUN_GEMM_BENCH=1 ./main.sh
#
# Run GEMM faster with a thinned grid:
#   FAST=1 ./main.sh
#
# Custom dims & memory cap:
#   COMMON_DIMS="128,256,512,1024,2048,4096,8192" MEM_CAP_GB=16 ./main.sh
#
# WHAT IT OUTPUTS
#
# System configuration file:
#    ./system.json
#    {
#      "processing_mode": "no_overlap",
#      "matrix": {
#        "float32": {
#          "tflops": <number>,
#          "gflops_efficiency": <gemm_bench_json_array>
#        }
#      },
#      "vector": {
#        "float32": {
#          "tflops": <number>,
#          "gflops_efficiency": [[16,e],[4,e],[1,e],[0,e]]
#        }
#      },
#      "mem1": { "GiB": <num>, "GBps": <num>, "MB_efficiency": [[MB,e],...] },
#      "mem2": { "GiB": <num>, "GBps": <num>, "MB_efficiency": [[MB,e],...] },
#      "networks": [ ... ]
#    }

set -euo pipefail

# --------------------------- UI / Logging helpers ----------------------------
if [ -t 1 ]; then
  C_BOLD="\033[1m"; C_DIM="\033[2m"; C_RED="\033[31m"; C_GRN="\033[32m"
  C_YLW="\033[33m"; C_BLU="\033[34m"; C_CYN="\033[36m"; C_RST="\033[0m"
else
  C_BOLD=""; C_DIM=""; C_RED=""; C_GRN=""; C_YLW=""; C_BLU=""; C_CYN=""; C_RST=""
fi
STEP_NUM=0
log_step()   { STEP_NUM=$((STEP_NUM+1)); echo -e "${C_BOLD}${C_BLU}==> [$STEP_NUM] $*${C_RST}"; }
log_info()   { echo -e "    ${C_CYN}•${C_RST} $*"; }
log_warn()   { echo -e "    ${C_YLW}!${C_RST} $*"; }
log_ok()     { echo -e "    ${C_GRN}✓${C_RST} $*"; }
log_fail()   { echo -e "    ${C_RED}✗${C_RST} $*"; }

# ------------------------------ Paths & inputs -------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUILD_DIR="$SCRIPT_DIR/builds"
LOG_DIR="$BUILD_DIR/logs"
mkdir -p "$BUILD_DIR" "$LOG_DIR"

# Sources:
SOURCES=(gemm_bench.cu eltwise_fma.cu gpu_info.cu mem_bw.cu net_bw.cu)

FORCE_REBUILD=${FORCE_REBUILD:-0}            # set to 1 to force rebuilds
RERUN_GEMM_BENCH=${RERUN_GEMM_BENCH:-false}  # default: false (do not re-run by default)

# -------------------------- Env knobs (forwarding) ---------------------------
# Added MB_DIMS and MB_DIM_STRIDE; kept TOPK_PRINT for compatibility.
GEMM_ENV_KEYS=(FAST DIM_STRIDE MB_DIM_STRIDE MEM_CAP_GB REPS_AUTO_TARGET_MS COMMON_DIMS M_DIMS N_DIMS K_DIMS MB_DIMS TOPK_PRINT)
for k in "${GEMM_ENV_KEYS[@]}"; do
  if [[ -n "${!k-}" ]]; then export "$k"; fi
done

# ------------------------------- Utilities ----------------------------------
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    log_fail "Need sha256sum or shasum (install coreutils)."; exit 2
  fi
}

detect_cuda() {
  if ! command -v nvcc >/dev/null 2>&1; then
    log_fail "nvcc not found on PATH. Please install CUDA Toolkit."; exit 2
  fi
  CUDA_HOME="$(cd "$(dirname "$(dirname "$(command -v nvcc)")")" && pwd)"
  echo "$CUDA_HOME"
}

detect_cc() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local line
    line="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed -n '1p' || true)"
    line="${line//./}"
    if [[ -n "${line:-}" ]]; then echo "$line"; return; fi
  fi
  echo "70"  # safe default
}

extract_build_cmd_from_header() {
  local src="$1"
  head -n 100 "$src" | awk '
    BEGIN { cmd="" }
    /(^\/\/|^#).*(nvcc[[:space:]].*)/ {
      line=$0; sub(/^\/\/[[:space:]]*/,"",line); sub(/^#[[:space:]]*/,"",line)
      if (match(line,/nvcc[^\r\n]*/)) { cmd=substr(line,RSTART,RLENGTH); print cmd; exit }
    }'
}

compose_nvcc_cmd() {
  local src="$1" out="$2" cc="$3" cuda_home="$4"
  local libs=""
  grep -Eq '#include[[:space:]]*<cublas' "$src" && libs="$libs -lcublas"
  grep -Eq '#include[[:space:]]*<nccl\.h>' "$src" && libs="$libs -lnccl"
  echo "nvcc -O3 -std=c++17 -I\"$cuda_home/include\" -L\"$cuda_home/lib64\" -gencode arch=compute_${cc},code=sm_${cc} \"$src\" -o \"$out\" $libs"
}

normalize_nvcc_cmd() {
  local raw="$1" src="$2" out="$3" cc="$4" cuda_home="$5"
  local cmd="$raw"
  cmd="$(echo "$cmd" | sed -E "s@(^|[[:space:]])([[:alnum:]_./-]*${src##*/})([[:space:]]|$)@ \"$src\" @")"
  if echo "$cmd" | grep -qE -- '(^|[[:space:]])-o[[:space:]]'; then
    cmd="$(echo "$cmd" | sed -E "s@-o[[:space:]]+[^[:space:]]+@-o \"$out\"@")"
  else
    cmd="$cmd -o \"$out\""
  fi
  echo "$cmd" | grep -q -- "-I$cuda_home/include" || cmd="$cmd -I\"$cuda_home/include\""
  echo "$cmd" | grep -q -- "-L$cuda_home/lib64"   || cmd="$cmd -L\"$cuda_home/lib64\""
  if ! echo "$cmd" | grep -q -- '-gencode' && ! echo "$cmd" | grep -q -- '-arch'; then
    cmd="$cmd -gencode arch=compute_${cc},code=sm_${cc}"
  fi
  echo "$cmd"
}

build_one() {
  local src="$1"
  if [[ ! -f "$src" ]]; then log_warn "Missing source: $src (skipping)"; return 0; fi

  local stem="${src##*/}"; stem="${stem%.cu}"
  local bin="$BUILD_DIR/$stem"
  local hashnew; hashnew="$(sha256_file "$src")"
  local hashfile="$BUILD_DIR/$stem.sha256"
  local bldlog="$LOG_DIR/$stem.build.log"

  local need=0
  if [[ $FORCE_REBUILD -eq 1 || ! -x "$bin" || ! -f "$hashfile" || "$(cat "$hashfile")" != "$hashnew" ]]; then need=1; fi
  if [[ $need -eq 0 ]]; then log_ok "Up to date: $src → $bin"; return 0; fi

  log_info "Compiling $src → $bin"
  local cuda_home cc nvcc_cmd raw
  cuda_home="$(detect_cuda)"; cc="$(detect_cc)"
  raw="$(extract_build_cmd_from_header "$src" || true)"
  if [[ -n "$raw" ]]; then nvcc_cmd="$(normalize_nvcc_cmd "$raw" "$src" "$bin" "$cc" "$cuda_home")"
  else nvcc_cmd="$(compose_nvcc_cmd "$src" "$bin" "$cc" "$cuda_home")"; fi

  log_info "nvcc cmd: ${C_DIM}$nvcc_cmd${C_RST}"
  set +e; eval "$nvcc_cmd" >"$bldlog" 2>&1; rc=${PIPESTATUS[0]:-0}; set -e
  if [[ $rc -ne 0 ]]; then
    log_fail "Build failed for $src (see $bldlog)"
    [[ "$stem" == "net_bw" ]] && { log_warn "NCCL missing? Skipping net bench."; return 0; }
    exit 3
  fi
  echo "$hashnew" >"$hashfile"
  chmod +x "$bin"
  log_ok "Built $bin"
}

# ------------------------------- Parse helpers -------------------------------
parse_gpu_info() {
  local txt="$1"
  GPU_NAME="$(echo "$txt" | awk -F= '/^GPU_NAME=/{print $2; exit}')"
  GPU_CC="$(echo "$txt" | awk -F= '/^CC=/{print $2; exit}')"
  GPU_FP32_TFLOPS="$(echo "$txt" | awk -F= '/^FP32_TFLOPS_PEAK=/{print $2; exit}')"
  GPU_MEM1_GBPS_PEAK_THEO="$(echo "$txt" | awk -F= '/^MEM1_GBPS_PEAK_THEO=/{print $2; exit}')"
}

parse_mem_bw() {
  local txt="$1"
  local l1 l2
  l1="$(echo "$txt" | grep -m1 'MEM1_SUMMARY' || true)"
  l2="$(echo "$txt" | grep -m1 'MEM2_SUMMARY' || true)"

  MEM1_GiB="$(echo "$l1" | sed -E 's/.*GiB=([0-9.]+).*/\1/')"
  MEM1_GBps="$(echo "$l1" | sed -E 's/.*GBps=([0-9.]+).*/\1/')"
  local a1; a1="$(echo "$l1" | sed -n -E 's/.*MB_eff=\[(.*)\].*/\1/p' | sed 's/(/[/g;s/)/]/g')"
  MEM1_MB_EFF="[$a1]"

  MEM2_GiB="$(echo "$l2" | sed -E 's/.*GiB=([0-9.]+).*/\1/')"
  MEM2_GBps="$(echo "$l2" | sed -E 's/.*GBps=([0-9.]+).*/\1/')"
  local a2; a2="$(echo "$l2" | sed -n -E 's/.*MB_eff=\[(.*)\].*/\1/p' | sed 's/(/[/g;s/)/]/g')"
  MEM2_MB_EFF="[$a2]"
}

# -------- FIXED: robustly capture the full [[...]] vector; never drop it -----
parse_eltwise() {
  local txt="$1"

  VEC_TFLOPS="$(printf '%s\n' "$txt" \
    | awk -F= '/^[[:space:]]*tflops=/{gsub(/[ \t\r]/,"",$2); print $2; exit}')"
  [[ -z "${VEC_TFLOPS:-}" ]] && VEC_TFLOPS="0"

  local first
  first="$(printf '%s\n' "$txt" \
    | awk -F= '/^[[:space:]]*gflops_efficiency=/{print substr($0,index($0,"=")+1); exit}')"

  if [[ -n "${first:-}" && "$first" != *"]]"* ]]; then
    local rest
    rest="$(printf '%s\n' "$txt" | awk '
      /^[[:space:]]*gflops_efficiency=/ {grab=1; next}
      grab && !done {print; if ($0 ~ /\]\]/) {done=1; exit}}
    ')"
    first="${first}${rest}"
  fi

  first="$(printf '%s' "$first" | tr -d '\r' | sed -E 's/^(.*\]\]).*$/\1/')"
  if [[ -n "${first:-}" && "$first" != *"]]" ]]; then
    first="${first}]]"
  fi
  VEC_EFF="$first"
}

parse_gemm_json() {
  local json_path="$1"
  if [[ -f "$json_path" ]]; then
    if command -v jq >/dev/null 2>&1; then
      MAT_TFLOPS="$(jq -r '.float32.tflops // 0' "$json_path" 2>/dev/null || echo 0)"
      MAT_EFF="$(jq -c '.float32.gflops_efficiency // []' "$json_path" 2>/dev/null || echo '[]')"
      return 0
    elif command -v python3 >/dev/null 2>&1; then
      read -r MAT_TFLOPS MAT_EFF < <(python3 - "$json_path" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
tf=d.get("float32",{}).get("tflops",0)
eff=d.get("float32",{}).get("gflops_efficiency",[])
print(tf, json.dumps(eff,separators=(",",":")))
PY
)
      return 0
    else
      log_warn "Neither jq nor python3 found to parse $json_path; leaving matrix empty."
      return 1
    fi
  else
    log_warn "gemm JSON not found at $json_path"
    return 1
  fi
}

parse_net() {
  local txt="$1"
  if [[ -z "${txt:-}" ]]; then NET_ARR="[]"; return; fi
  if command -v jq >/dev/null 21>&1; then
    NET_ARR="$(echo "$txt" | jq -c '.networks // []' 2>/dev/null || echo "[]")"
  elif command -v python3 >/dev/null 2>&1; then
    NET_ARR="$(python3 - <<'PY'
import json,sys
s=sys.stdin.read()
try:
  d=json.loads(s)
  print(json.dumps(d.get("networks",[]),separators=(",",":")))
except Exception:
  print("[]")
PY
<<<"$txt")"
  else
    local one; one="$(echo "$txt" | tr -d '\n')"
    local arr; arr="$(echo "$one" | sed -E 's/.*"networks"[[:space:]]*:[[:space:]]*(\[[[:print:]]*\]).*/\1/' || true)"
    [[ -z "${arr:-}" ]] && arr="[]"
    NET_ARR="$arr"
  fi
}

# ----------------------------- 1) Toolchain check ----------------------------
log_step "Environment check"
CUDA_HOME="$(detect_cuda)"
log_ok "CUDA_HOME: $CUDA_HOME"

if command -v nvidia-smi >/dev/null 2>&1; then
  NSUM="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | sed -n '1p' || true)"
  log_ok "nvidia-smi: ${NSUM:-unknown}"
else
  log_warn "nvidia-smi not found; continuing without it"
fi

if command -v jq >/dev/null 2>&1; then
  HAVE_JQ=1; log_ok "jq found (will pretty-print JSON)"
else
  HAVE_JQ=0; log_warn "jq not found (JSON will be compact). Install jq for pretty output."
fi

# ----------------------------- 2) Build binaries -----------------------------
log_step "Build check & compilation (into ./builds)"
for src in "${SOURCES[@]}"; do
  build_one "$src"
done

# ----------------------------- 3) Run benchmarks -----------------------------
# gpu_info
if [[ -x "$BUILD_DIR/gpu_info" ]]; then
  log_step "Running gpu_info"
  set +e
  "$BUILD_DIR/gpu_info" 2>&1 | tee "$LOG_DIR/gpu_info.out" >/dev/null
  rc=${PIPESTATUS[0]:-0}
  set -e
  if [[ $rc -ne 0 ]]; then
    log_warn "gpu_info failed (rc=$rc)"
  fi
  parse_gpu_info "$(cat "$LOG_DIR/gpu_info.out" 2>/dev/null || true)"
  log_ok "GPU: ${GPU_NAME:-unknown}, CC=${GPU_CC:-?}, FP32 TFLOPS peak=${GPU_FP32_TFLOPS:-?}, MEM1 theo GB/s=${GPU_MEM1_GBPS_PEAK_THEO:-?}"
else
  log_warn "gpu_info binary missing; proceeding without it"
fi

# mem_bw
if [[ -x "$BUILD_DIR/mem_bw" ]]; then
  log_step "Running mem_bw"
  set +e
  "$BUILD_DIR/mem_bw" 2>&1 | tee "$LOG_DIR/mem_bw.out" >/dev/null
  rc=${PIPESTATUS[0]:-0}
  set -e
  if [[ $rc -ne 0 ]]; then
    log_warn "mem_bw failed (rc=$rc); memory sections will be empty"
  else
    parse_mem_bw "$(cat "$LOG_DIR/mem_bw.out")"
    log_ok "mem1: ${MEM1_GiB:-?} GiB @ ${MEM1_GBps:-?} GB/s; mem2: ${MEM2_GiB:-?} GiB @ ${MEM2_GBps:-?} GB/s"
  fi
else
  log_warn "mem_bw binary missing; mem sections will be empty"
fi

# eltwise_fma (vector)
if [[ -x "$BUILD_DIR/eltwise_fma" ]]; then
  log_step "Running eltwise_fma"
  set +e
  "$BUILD_DIR/eltwise_fma" 2>&1 | tee "$LOG_DIR/eltwise_fma.out"
  rc=${PIPESTATUS[0]:-0}
  set -e
  if [[ $rc -ne 0 ]]; then
    log_warn "eltwise_fma failed (rc=$rc); vector section may be empty"
    VEC_TFLOPS="${VEC_TFLOPS:-0}"; VEC_EFF="${VEC_EFF:-[]}"
  else
    parse_eltwise "$(cat "$LOG_DIR/eltwise_fma.out")"
    log_ok "vector.fp32 tflops=${VEC_TFLOPS:-?}"
  fi
else
  log_warn "eltwise_fma binary missing; vector section will be empty"
fi

# gemm_bench (matrix) — reuse JSON by default
GEMM_JSON="$SCRIPT_DIR/gemm_bench.json"
reuse=false
case "${RERUN_GEMM_BENCH,,}" in
  0|false|no|off) reuse=true ;;
esac

if [[ -x "$BUILD_DIR/gemm_bench" ]]; then
  if [[ "$reuse" == true && -f "$GEMM_JSON" ]]; then
    log_step "Using existing gemm_bench.json (RERUN_GEMM_BENCH=false)"
    if parse_gemm_json "$GEMM_JSON"; then
      log_ok "matrix.fp32 tflops=${MAT_TFLOPS:-?} (from existing ./gemm_bench.json)"
    else
      log_warn "Existing gemm_bench.json unreadable; will run gemm_bench in a temp dir."
      reuse=false
    fi
  elif [[ "$reuse" == true && ! -f "$GEMM_JSON" ]]; then
    log_warn "gemm_bench.json not found; will run gemm_bench in a temp dir."
    reuse=false
  fi

  if [[ "$reuse" == false ]]; then
    log_step "Running gemm_bench in a temp directory"
    env_summary=(); for k in "${GEMM_ENV_KEYS[@]}"; do [[ -n "${!k-}" ]] && env_summary+=("$k=${!k}"); done
    [[ ${#env_summary[@]} -gt 0 ]] && log_info "GEMM env: ${env_summary[*]}"

    # Create a unique temp run dir and execute gemm_bench there
    RUN_DIR="$BUILD_DIR/gemm_run_$(date +%Y%m%d-%H%M%S)_$$"
    mkdir -p "$RUN_DIR"

    if command -v stdbuf >/dev/null 2>&1; then STDBUF="stdbuf -oL -eL"; else STDBUF=""; fi

    set +e
    ( cd "$RUN_DIR" && $STDBUF "$BUILD_DIR/gemm_bench" ) 2>&1 | tee "$LOG_DIR/gemm_bench.out"
    rc=${PIPESTATUS[0]:-0}
    set -e

    if [[ $rc -ne 0 ]]; then
      log_warn "gemm_bench failed (rc=$rc); matrix section will be empty"
      MAT_TFLOPS="${MAT_TFLOPS:-0}"; MAT_EFF="[]"
    else
      # Parse the freshly produced JSON inside RUN_DIR (do NOT touch ./gemm_bench.json)
      if parse_gemm_json "$RUN_DIR/gemm_bench.json"; then
        log_ok "matrix.fp32 tflops=${MAT_TFLOPS:-?} (from $RUN_DIR/gemm_bench.json)"
      else
        log_warn "Temp gemm_bench.json missing/unreadable; leaving matrix empty"
        MAT_TFLOPS="${MAT_TFLOPS:-0}"; MAT_EFF="[]"
      fi
    fi
  fi
else
  log_warn "gemm_bench binary missing; matrix section will be empty"
  MAT_TFLOPS="${MAT_TFLOPS:-0}"; MAT_EFF="[]"
fi

# net_bw (optional) — reuse JSON by default
NET_JSON="$SCRIPT_DIR/net_bw.json"
net_reuse=false
case "${RERUN_NET_BENCH,,}" in
  0|false|no|off) net_reuse=true ;;
esac

if [[ $SKIP_NET -eq 0 && -x "$BUILD_DIR/net_bw" ]]; then
  if [[ "$net_reuse" == true && -f "$NET_JSON" ]]; then
    log_step "Using existing net_bw.json (RERUN_NET_BENCH=false)"
    if [[ -s "$NET_JSON" ]]; then
      if command -v jq >/dev/null 2>&1; then
        NET_ARR="$(jq -c '.networks // []' "$NET_JSON" 2>/dev/null || echo "[]")"
      else
        parse_net "$(cat "$NET_JSON" 2>/dev/null || true)"
      fi
      log_ok "networks parsed (from existing ./net_bw.json)"
    else
      log_warn "Existing net_bw.json is empty; will run net_bw in a temp dir."
      net_reuse=false
    fi
  elif [[ "$net_reuse" == true && ! -f "$NET_JSON" ]]; then
    log_warn "net_bw.json not found; will run net_bw in a temp dir."
    net_reuse=false
  fi

  if [[ "$net_reuse" == false ]]; then
    log_step "Running net_bw in a temp directory"
    RUN_DIR="$BUILD_DIR/net_run_$(date +%Y%m%d-%H%M%S)_$$"
    mkdir -p "$RUN_DIR"

    if command -v stdbuf >/dev/null 2>&1; then STDBUF="stdbuf -oL -eL"; else STDBUF=""; fi

    set +e
    ( cd "$RUN_DIR" && $STDBUF "$BUILD_DIR/net_bw" ) 2>&1 | tee "$LOG_DIR/net_bw.out"
    rc=${PIPESTATUS[0]:-0}
    set -e

    if [[ $rc -ne 0 ]]; then
      log_warn "net_bw failed (rc=$rc); networks will be []"
      NET_ARR="[]"
    else
      if [[ -f "$RUN_DIR/net_bw.json" ]]; then
        if command -v jq >/dev/null 2>&1; then
          NET_ARR="$(jq -c '.networks // []' "$RUN_DIR/net_bw.json" 2>/dev/null || echo "[]")"
        else
          parse_net "$(cat "$RUN_DIR/net_bw.json" 2>/dev/null || true)"
        fi
        log_ok "networks parsed (from $RUN_DIR/net_bw.json)"
      else
        log_warn "Expected $RUN_DIR/net_bw.json not found; networks will be []"
        NET_ARR="[]"
      fi
    fi
  fi
else
  NET_ARR="[]"
  if [[ $SKIP_NET -eq 1 ]]; then
    log_warn "Skipping net_bw by request (SKIP_NET=1)"
  else
    log_warn "net_bw binary missing; networks will be []"
  fi
fi

# --------------------------- 4) JSON defaults/validation ---------------------
json_is_array() { [[ "${1:-}" =~ ^[[:space:]]*\[.*\][[:space:]]*$ ]]; }
json_is_obj_or_arr() { [[ "${1:-}" =~ ^[[:space:]]*[\[\{].*[\]\}][[:space:]]*$ ]]; }
num_sanitize() { printf '%s' "${1:-0}" | tr -cd '0-9.+-'; }

MAT_TFLOPS="$(num_sanitize "${MAT_TFLOPS:-0}")"
VEC_TFLOPS="$(num_sanitize "${VEC_TFLOPS:-0}")"
MEM1_GiB="$(num_sanitize "${MEM1_GiB:-0}")"
MEM1_GBps="$(num_sanitize "${MEM1_GBps:-0}")"
MEM2_GiB="$(num_sanitize "${MEM2_GiB:-0}")"
MEM2_GBps="$(num_sanitize "${MEM2_GBps:-0}")"

json_is_array "${VEC_EFF:-}"      || VEC_EFF="[]"
json_is_array "${MEM1_MB_EFF:-}"  || MEM1_MB_EFF="[]"
json_is_array "${MEM2_MB_EFF:-}"  || MEM2_MB_EFF="[]"
json_is_array "${NET_ARR:-}"      || NET_ARR="[]"
json_is_obj_or_arr "${MAT_EFF:-}" || MAT_EFF="{}"

# ----------------------------- 5) Emit system.json ---------------------------
log_step "Generating ./system.json"

TMP_JSON="$BUILD_DIR/system.json.tmp"
cat > "$TMP_JSON" <<JSON
{
  "processing_mode": "no_overlap",
  "matrix": {
    "float32": {
      "tflops": ${MAT_TFLOPS},
      "gflops_efficiency": ${MAT_EFF}
    }
  },
  "vector": {
    "float32": {
      "tflops": ${VEC_TFLOPS},
      "gflops_efficiency": ${VEC_EFF}
    }
  },
  "mem1": {
    "GiB": ${MEM1_GiB},
    "GBps": ${MEM1_GBps},
    "MB_efficiency": ${MEM1_MB_EFF}
  },
  "mem2": {
    "GiB": ${MEM2_GiB},
    "GBps": ${MEM2_GBps},
    "MB_efficiency": ${MEM2_MB_EFF}
  },
  "networks": ${NET_ARR}
}
JSON

if command -v jq >/dev/null 2>&1; then
  if ! jq -M . "$TMP_JSON" > "$SCRIPT_DIR/system.json" 2> "$BUILD_DIR/system.json.err"; then
    log_fail "jq failed to pretty-print JSON. Raw file with line numbers:"
    nl -ba "$TMP_JSON" 1>&2 || true
    log_warn "Leaving compact JSON at $SCRIPT_DIR/system.json (without jq formatting)"
    cp -f "$TMP_JSON" "$SCRIPT_DIR/system.json"
  else
    log_ok "Wrote: $SCRIPT_DIR/system.json"
  fi
else
  mv "$TMP_JSON" "$SCRIPT_DIR/system.json"
  log_ok "Wrote: $SCRIPT_DIR/system.json (without jq formatting)"
fi

echo
log_step "Done"
echo -e "   Inspect logs in ${C_DIM}$LOG_DIR${C_RST}"
echo -e "   Re-run with ${C_DIM}FORCE_REBUILD=1${C_RST} to force recompilation"
echo -e "   Example: ${C_DIM}FAST=1 ./main.sh${C_RST} to thin the GEMM grid"
echo -e "   Reuse previous GEMM results with ${C_DIM}RERUN_GEMM_BENCH=false ./main.sh${C_RST}"
