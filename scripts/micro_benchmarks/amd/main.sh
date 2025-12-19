#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# main.sh — ROCm/HIP micro-bench orchestration & system.json generator (AMD)
#
# WHAT THIS SCRIPT DOES
# ---------------------
# Orchestrates build + execution of 5 HIP/ROCm micro-benchmarks and aggregates
# their results into a single machine-readable file: ./system.json. It preserves
# prior results when available and logs every step to ./builds/logs/.
#
# Included tools (auto-built from *.cpp in this directory):
#   - gemm_bench.cpp  : FP32 GEMM sweep (micro-batch aware). Writes JSON:
#                       ./gemm_bench.json. Progress to stderr.
#   - eltwise_fma.cpp : Vector FMA throughput; prints TFLOPS & efficiency table.
#   - gpu_info.cpp    : Basic GPU/device info (name, gfx arch, FP32 peak, mem).
#   - mem_bw.cpp      : Device & host memory bandwidths (mem1/mem2).
#   - net_bw.cpp      : P2P & RCCL collectives; writes JSON: ./net_bw.json.
#                       When run directly, it prints JSON to stdout by default;
#                       from this script it is run with DISPLAY_OUTPUT=0 to keep
#                       stdout clean.
#
# KEY BEHAVIOR
# ------------
# • Incremental builds: each source is hashed; recompiles only when changed.
# • Auto-detects ROCm install & offload archs (via hipcc/rocminfo).
# • Reuse-first for JSONs:
#     - GEMM: If ./gemm_bench.json exists and RERUN_GEMM_BENCH=0 (default),
#       reuse it; otherwise run ./builds/gemm_bench in a temp dir and parse its
#       fresh JSON.
#     - NETWORKS (REQUIRED): If ./net_bw.json exists and is readable and
#       RERUN_NET_BENCH=0 (default), reuse it; otherwise run ./builds/net_bw in
#       a temp dir and parse its fresh JSON. The “networks” field is [] only if
#       neither reuse nor a fresh run yields a valid JSON.
# • Pretty-prints ./system.json via jq when available (compact otherwise).
# • Stores build/runtime logs in ./builds/logs for post-mortem debugging.
#
# RUNTIME OPTIONS (environment variables)
# ---------------------------------------
# General orchestration:
#   FORCE_REBUILD=0|1        Force rebuild of all sources (default: 0).
#   RERUN_GEMM_BENCH=0|1     0 (default) reuses ./gemm_bench.json if present.
#   RERUN_NET_BENCH=0|1      0 (default) reuses ./net_bw.json if present;
#                            1 forces a fresh net_bw run.
#
# GEMM grid & memory knobs (forwarded to gemm_bench.cpp):
#   FAST=0|1                 Default here: 0.
#   DIM_STRIDE=<int>
#   MB_DIM_STRIDE=<int>
#   MB_DIMS="1,2,4,8,16,32,64,128"   (FAST=1 ⇒ {1,2,4,8,16})
#   MEM_CAP_GB=<float>
#   REPS_AUTO_TARGET_MS=<float>      (typ. 35 ms if supported by the tool)
#   COMMON_DIMS / M_DIMS / N_DIMS / K_DIMS
#   TOPK_PRINT                        (kept for compatibility)
#
# net_bw.cpp knobs (consumed by the binary; not all are set by this script):
#   DISPLAY_OUTPUT=0|1       Default in this script: 0 (suppresses JSON to stdout).
#                            When you run ./builds/net_bw directly, it defaults to 1.
#   PP/AR controls, topology hints, etc. are respected by the binary if set
#   (e.g., RUN_SIZES, LLM_PAYLOAD_MB, XGMI_PERDIR_GBPS, PCIE_GEN/PCIE_WIDTH, …).
#
# HOW TO RUN
# ----------
#   ./main.sh                         # build if needed; reuse JSONs when present
#   FORCE_REBUILD=1 ./main.sh         # force recompilation of all tools
#   RERUN_GEMM_BENCH=1 ./main.sh      # ignore existing gemm_bench.json
#   RERUN_NET_BENCH=1 ./main.sh       # ignore existing net_bw.json (required step)
#   FAST=1 ./main.sh                  # thinner GEMM grid forwarded to gemm_bench
#
# OUTPUTS
# -------
#   ./system.json                     # pretty if jq is installed; compact otherwise
#   ./builds/logs/*.{build.log,out}   # per-tool build & runtime logs
#
# NOTES
# -----
# • If ./net_bw.json exists but is unreadable (permissions/ACL), the script will
#   fall back to running ./builds/net_bw.
# • Offload arch auto-detection falls back to gfx90a if rocminfo is unavailable.
# • This script does not modify ./gemm_bench.json or ./net_bw.json unless it has
#   to run the corresponding tool; fresh runs are executed in temp dirs.
# -----------------------------------------------------------------------------

set -euo pipefail
set -E
trap 'log_fail "Error on or near line $LINENO (exit $?)"; exit 1' ERR

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

: "${ROCM_PATH:=/opt/rocm}"
BUILD_DIR="$SCRIPT_DIR/builds"
LOG_DIR="$BUILD_DIR/logs"
mkdir -p "$BUILD_DIR" "$LOG_DIR"

# Sources expected in this directory:
SOURCES=(gemm_bench.cpp eltwise_fma.cpp gpu_info.cpp mem_bw.cpp net_bw.cpp)

FORCE_REBUILD=${FORCE_REBUILD:-0}
RERUN_GEMM_BENCH=${RERUN_GEMM_BENCH:-0}   # default: reuse if JSON exists
RERUN_NET_BENCH=${RERUN_NET_BENCH:-0}     # default: reuse if JSON exists

# -------------------------- Env knobs (forwarding) ---------------------------
# Set FAST default to 0 explicitly and forward recognized keys.
FAST="${FAST:-0}"
export FAST
GEMM_ENV_KEYS=(FAST DIM_STRIDE MB_DIM_STRIDE MEM_CAP_GB REPS_AUTO_TARGET_MS COMMON_DIMS M_DIMS N_DIMS K_DIMS MB_DIMS TOPK_PRINT)
for k in "${GEMM_ENV_KEYS[@]}"; do
  if [[ -v $k ]]; then export "$k"; fi
done

# ------------------------------- Utilities ----------------------------------
need() { command -v "$1" >/dev/null 2>&1 || { log_fail "Missing command: $1"; exit 2; }; }

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    log_fail "Need sha256sum or shasum (install coreutils)."; exit 2
  fi
}

detect_rocm() {
  # Prefer hipcc path if present; fallback to ROCM_PATH
  if command -v hipcc >/dev/null 2>&1; then
    local hipcc_path; hipcc_path="$(command -v hipcc)"
    local hip_root; hip_root="$(cd "$(dirname "$(dirname "$hipcc_path")")" && pwd)"
    echo "$hip_root"
  else
    echo "$ROCM_PATH"
  fi
}

detect_offload_archs() {
  if [[ -n "${OFFLOAD_ARCHS:-}" ]]; then
    echo "${OFFLOAD_ARCHS}"
    return
  fi
  if command -v rocminfo >/dev/null 2>&1; then
    local archs
    archs=$(rocminfo 2>/dev/null | awk '/Name:[[:space:]]*gfx/{print $2}' | sed 's/Name:[[:space:]]*//;s/:.*$//' | sort -u | tr '\n' ',')
    archs="${archs%,}"
    if [[ -n "$archs" ]]; then
      echo "$archs"
      return
    fi
  fi
  echo "gfx90a"  # MI250 default
}

extract_build_cmd_from_header() {
  local src="$1"
  head -n 100 "$src" | awk '
    BEGIN { cmd="" }
    /(^\/\/|^#).*(hipcc[[:space:]].*)/ {
      line=$0; sub(/^\/\/[[:space:]]*/,"",line); sub(/^#[[:space:]]*/,"",line)
      if (match(line,/hipcc[^\r\n]*/)) { cmd=substr(line,RSTART,RLENGTH); print cmd; exit }
    }'
}

compose_hipcc_cmd() {
  local src="$1" out="$2" rocm_home="$3" offarchs="$4"
  local libs=""
  grep -Eq '#include[[:space:]]*<rocblas' "$src" && libs="$libs -lrocblas"
  grep -Eq '#include[[:space:]]*<rccl\.h>' "$src" && libs="$libs -lrccl"
  grep -Eq '#include[[:space:]]*<hipblaslt' "$src" && libs="$libs -lhipblaslt"
  echo "hipcc -O3 -std=c++17 --offload-arch=${offarchs} -I\"$rocm_home/include\" -L\"$rocm_home/lib\" \"$src\" -o \"$out\" $libs"
}


normalize_hipcc_cmd() {
  local raw="$1" src="$2" out="$3" rocm_home="$4" offarchs="$5"
  local cmd="$raw"

  # Ensure quoted source is used
  cmd="$(echo "$cmd" | sed -E "s@(^|[[:space:]])([[:alnum:]_./-]*${src##*/})([[:space:]]|$)@ \"$src\" @")"
  # Ensure -o <out>
  if echo "$cmd" | grep -qE -- '(^|[[:space:]])-o[[:space:]]'; then
    cmd="$(echo "$cmd" | sed -E "s@-o[[:space:]]+[^[:space:]]+@-o \"$out\"@")"
  else
    cmd="$cmd -o \"$out\""
  fi
  # Ensure includes/libs and offload-arch present
  echo "$cmd" | grep -q -- "-I$rocm_home/include" || cmd="$cmd -I\"$rocm_home/include\""
  echo "$cmd" | grep -q -- "-L$rocm_home/lib"     || cmd="$cmd -L\"$rocm_home/lib\""
  echo "$cmd" | grep -q -- "--offload-arch"       || cmd="$cmd --offload-arch=${offarchs}"
  echo "$cmd" | grep -q -- "-std="                 || cmd="$cmd -std=c++17"
  echo "$cmd" | grep -q -- "-O[0-3s]"              || cmd="$cmd -O3"

  # Link common libs based on includes if missing
  if grep -Eq '#include[[:space:]]*<rocblas' "$src"; then echo "$cmd" | grep -q -- "-lrocblas" || cmd="$cmd -lrocblas"; fi
  if grep -Eq '#include[[:space:]]*<rccl\.h>' "$src"; then echo "$cmd" | grep -q -- "-lrccl"   || cmd="$cmd -lrccl";   fi
  if grep -Eq '#include[[:space:]]*<hipblaslt' "$src"; then echo "$cmd" | grep -q -- "-lhipblaslt" || cmd="$cmd -lhipblaslt"; fi

  echo "$cmd"
}

build_one() {
  local src="$1"
  if [[ ! -f "$src" ]]; then log_warn "Missing source: $src (skipping)"; return 0; fi

  local stem="${src##*/}"; stem="${stem%.cpp}"
  local bin="$BUILD_DIR/$stem"
  local hashnew; hashnew="$(sha256_file "$src")"
  local hashfile="$BUILD_DIR/$stem.sha256"
  local bldlog="$LOG_DIR/$stem.build.log"

  local need=0
  if [[ $FORCE_REBUILD -eq 1 || ! -x "$bin" || ! -f "$hashfile" || "$(cat "$hashfile")" != "$hashnew" ]]; then need=1; fi
  if [[ $need -eq 0 ]]; then log_ok "Up to date: $src → $bin"; return 0; fi

  log_info "Compiling $src → $bin"
  need hipcc
  local rocm_home offarch raw hipcc_cmd
  rocm_home="$(detect_rocm)"
  offarch="$(detect_offload_archs)"

  raw="$(extract_build_cmd_from_header "$src" || true)"
  if [[ -n "$raw" ]]; then hipcc_cmd="$(normalize_hipcc_cmd "$raw" "$src" "$bin" "$rocm_home" "$offarch")"
  else hipcc_cmd="$(compose_hipcc_cmd "$src" "$bin" "$rocm_home" "$offarch")"; fi

  log_info "hipcc cmd: ${C_DIM}$hipcc_cmd${C_RST}"
  set +e; eval "$hipcc_cmd" >"$bldlog" 2>&1; rc=${PIPESTATUS[0]:-0}; set -e
  if [[ $rc -ne 0 ]]; then
    log_fail "Build failed for $src (see $bldlog)"
    [[ "$stem" == "net_bw" ]] && { log_warn "RCCL missing? Skipping net bench."; return 0; }
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
  GPU_GFX="$(echo "$txt" | awk -F= '/^GFX=/{print $2; exit}')"
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

parse_eltwise() {
  local txt="$1"

  # Prefer Python for robust parsing (handles multiline efficiency arrays)
  if command -v python3 >/dev/null 2>&1; then
    VEC_ALL="$(
      python3 -c 'import json,sys,re
s=sys.stdin.read()
out={}
for dt in ("float32","float16","float8"):
    m=re.search(r"Vector table \(%s\):.*?tflops\s*=\s*([0-9.+-Ee]+).*?gflops_efficiency\s*=\s*(\[\[.*?\]\])" % dt, s, re.S)
    if m:
        try: tf=float(m.group(1))
        except: tf=0.0
        try: eff=json.loads(m.group(2))
        except: eff=[]
        out[dt]={"tflops":tf,"gflops_efficiency":eff}
print(json.dumps(out, separators=(",",":")))' <<<"$txt"
    )"
  else
    # Fallback: awk (single-quoted program so Bash doesn't expand $1/$2)
    VEC_ALL="$(
      awk '
        function flush() {
          if (dtype != "") {
            t=(tflops=="" ? "0" : tflops)
            e=(eff=="" ? "[]" : eff)
            if (printed>0) printf(",")
            printf("\"%s\":{\"tflops\":%s,\"gflops_efficiency\":%s}", dtype, t, e)
            printed++
          }
          dtype=""; tflops=""; eff=""; grab=0
        }
        BEGIN { printed=0; dtype=""; tflops=""; eff=""; grab=0 }
        /Vector table \(float(32|16|8)\):/ {
          match($0,/Vector table \(float(32|16|8)\):/,m)
          flush(); dtype="float" m[1]; next
        }
        /tflops[[:space:]]*=/ {
          line=$0; sub(/.*=/,"",line); gsub(/[ \t\r]/,"",line); tflops=line; next
        }
        /gflops_efficiency[[:space:]]*=/ {
          grab=1; part=$0; sub(/.*=/,"",part); gsub(/\r/,"",part); eff=part
          if (index(part,"]]")==0) next; else { grab=0; next }
        }
        grab { eff=eff $0; if (index($0,"]]")) grab=0; next }
        END { flush() }
      ' <<<"$txt"
    )"
    VEC_ALL="{${VEC_ALL}}"
  fi

  # Ensure object shape even if nothing matched
  [[ -n "${VEC_ALL//[[:space:]]/}" ]] || VEC_ALL="{}"

  # Convenience vars used elsewhere (keep old names working)
  if command -v jq >/dev/null 2>&1; then
    VEC_TFLOPS="$(printf '%s' "$VEC_ALL" | jq -r '.float32.tflops // 0' 2>/dev/null || echo 0)"
    VEC_EFF="$(printf '%s' "$VEC_ALL" | jq -c '.float32.gflops_efficiency // []' 2>/dev/null || echo '[]')"
    VEC_TFLOPS_F16="$(printf '%s' "$VEC_ALL" | jq -r '.float16.tflops // 0' 2>/dev/null || echo 0)"
    VEC_EFF_F16="$(printf '%s' "$VEC_ALL" | jq -c '.float16.gflops_efficiency // []' 2>/dev/null || echo '[]')"
    VEC_TFLOPS_F8="$(printf '%s' "$VEC_ALL" | jq -r '.float8.tflops // 0' 2>/dev/null || echo 0)"
    VEC_EFF_F8="$(printf '%s' "$VEC_ALL" | jq -c '.float8.gflops_efficiency // []' 2>/dev/null || echo '[]')"
  else
    # Minimal sed fallback for fp32 only
    VEC_TFLOPS="$(printf '%s\n' "$txt" \
      | sed -n '/Vector table (float32):/,$p' \
      | sed -n 's/.*tflops[[:space:]]*=[[:space:]]*\([0-9.eE+-]\+\).*/\1/p' \
      | head -n1)"
    [[ -n "${VEC_TFLOPS:-}" ]] || VEC_TFLOPS=0
    VEC_EFF="$(printf '%s\n' "$txt" \
      | sed -n '/Vector table (float32):/,$p' \
      | sed -n 's/.*gflops_efficiency[[:space:]]*=[[:space:]]*\(\[\[.*\]\]\).*/\1/p' \
      | head -n1)"
    [[ -n "${VEC_EFF:-}" ]] || VEC_EFF="[]"
  fi

  return 0
}

parse_gemm_json() {
  local json_path="$1"
  if [[ -f "$json_path" ]]; then
    if command -v jq >/dev/null 2>&1; then
      # Float32 values for logging/back-compat:
      MAT_TFLOPS="$(
        jq -r '
          .float32.tflops
          // .fp32.tflops
          // .matrix.float32.tflops
          // .tflops
          // 0
        ' "$json_path" 2>/dev/null || echo 0
      )"
      MAT_EFF="$(
        jq -c '
          .float32.gflops_efficiency
          // .fp32.gflops_efficiency
          // .matrix.float32.gflops_efficiency
          // .efficiency
          // {}
        ' "$json_path" 2>/dev/null || echo '{}'
      )"
      # Build a compact object including float16/float8 when present.
      MAT_ALL="$(
        jq -c '
          def emit(k):
            if .|has(k) then
              { (k): { tflops: (.[k].tflops // 0),
                       gflops_efficiency: (.[k].gflops_efficiency // {}) } }
            else {} end;
          emit("float32") + emit("float16") + emit("float8")
        ' "$json_path" 2>/dev/null || echo '{}'
      )"
      return 0
    elif command -v python3 >/dev/null 2>&1; then
      read -r MAT_TFLOPS MAT_EFF MAT_ALL < <(python3 - "$json_path" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
def pick(*paths, default=None):
    for p in paths:
        cur=d
        ok=True
        for k in p.split('.'):
            if isinstance(cur, dict) and k in cur: cur=cur[k]
            else:
                ok=False; break
        if ok: return cur
    return default
tf = pick('float32.tflops','fp32.tflops','matrix.float32.tflops','tflops', default=0)
eff = pick('float32.gflops_efficiency','fp32.gflops_efficiency','matrix.float32.gflops_efficiency','efficiency', default={})
out={}
for k in ('float32','float16','float8'):
    if isinstance(d,dict) and k in d and isinstance(d[k],dict):
        out[k]={'tflops': d[k].get('tflops',0),
                'gflops_efficiency': d[k].get('gflops_efficiency',{})}
import json as J
print(tf, J.dumps(eff, separators=(",",":")), J.dumps(out, separators=(",",":")))
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
  if command -v jq >/dev/null 2>&1; then
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

num_sanitize() { printf '%s' "${1:-0}" | tr -cd '0-9.+-'; }
json_is_array() { [[ "${1:-}" =~ ^[[:space:]]*\[.*\][[:space:]]*$ ]]; }
json_is_obj_or_arr() { [[ "${1:-}" =~ ^[[:space:]]*[\[\{].*[\]\}][[:space:]]*$ ]]; }

# ----------------------------- 1) Environment check --------------------------
log_step "Environment check"
need bash
need awk
need sed
need tr
need tee
need sort

if command -v hipcc >/dev/null 2>&1; then
  HIPCC="$(command -v hipcc)"
  log_ok "hipcc: $HIPCC"
else
  log_fail "hipcc not found on PATH. Please install ROCm (HIP) toolchain."
  exit 2
fi

ROCM_HOME="$(detect_rocm)"
log_ok "ROCm home: $ROCM_HOME"

if command -v rocm-smi >/dev/null 2>&1; then
  local_sum="$(rocm-smi --showproductname --showdriverversion 2>/dev/null | sed -n '1,4p' || true)"
  [[ -n "$local_sum" ]] && log_ok "rocm-smi: $(echo "$local_sum" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')"
else
  log_warn "rocm-smi not found; continuing without it"
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
  else
    parse_gpu_info "$(cat "$LOG_DIR/gpu_info.out" 2>/dev/null || true)"
    log_ok "GPU: ${GPU_NAME:-unknown}, GFX=${GPU_GFX:-?}, FP32 TFLOPS peak=${GPU_FP32_TFLOPS:-?}, MEM1 theo GB/s=${GPU_MEM1_GBPS_PEAK_THEO:-?}"
  fi
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
  # Write all output to file first (avoid pipefail), then show it.
  "$BUILD_DIR/eltwise_fma" >"$LOG_DIR/eltwise_fma.out" 2>&1
  rc=$?
  set -e
  if [[ "${VERBOSE:-0}" == "1" ]]; then cat "$LOG_DIR/eltwise_fma.out" || true; fi

  # If the tool returns a non-zero (e.g., after printing “FP8…skipping”),
  # keep going and parse whatever it printed.
  if [[ $rc -ne 0 ]]; then
    log_warn "eltwise_fma returned rc=$rc (continuing; parsing output anyway)"
  fi
  parse_eltwise "$(cat "$LOG_DIR/eltwise_fma.out" 2>/dev/null || true)"
  VEC_TFLOPS="${VEC_TFLOPS:-0}"
  [[ -n "${VEC_EFF:-}" ]] || VEC_EFF="[]"
  log_ok "vector.fp32 tflops=${VEC_TFLOPS}"
else
  log_warn "eltwise_fma binary missing; vector section will be empty"
fi

# gemm_bench (matrix) — reuse JSON by default
GEMM_JSON="$SCRIPT_DIR/gemm_bench.json"
reuse_gemm=true
case "$(printf '%s' "${RERUN_GEMM_BENCH:-0}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on) reuse_gemm=false ;;
esac

if [[ -x "$BUILD_DIR/gemm_bench" ]]; then
  if [[ "$reuse_gemm" == true && -f "$GEMM_JSON" ]]; then
    log_step "Using existing gemm_bench.json (RERUN_GEMM_BENCH=0)"
    if parse_gemm_json "$GEMM_JSON"; then
      log_ok "matrix.fp32 tflops=${MAT_TFLOPS:-?} (from existing ./gemm_bench.json)"
    else
      log_warn "Existing gemm_bench.json unreadable; will run gemm_bench in a temp dir."
      reuse_gemm=false
    fi
  elif [[ "$reuse_gemm" == true && ! -f "$GEMM_JSON" ]]; then
    log_warn "gemm_bench.json not found; will run gemm_bench in a temp dir."
    reuse_gemm=false
  fi

  if [[ "$reuse_gemm" == false ]]; then
    log_step "Running gemm_bench in a temp directory"
    env_summary=()
    for k in "${GEMM_ENV_KEYS[@]}"; do
      if [[ -v $k ]]; then env_summary+=("$k=${!k}"); fi
    done
    (( ${#env_summary[@]} > 0 )) && log_info "GEMM env: ${env_summary[*]}"


    RUN_DIR="$BUILD_DIR/gemm_run_$(date +%Y%m%d-%H%M%S)_$$"
    mkdir -p "$RUN_DIR"

    if command -v stdbuf >/dev/null 2>&1; then STDBUF="stdbuf -oL -eL"; else STDBUF=""; fi

    set +e
    ( cd "$RUN_DIR" && $STDBUF "$BUILD_DIR/gemm_bench" ) >"$LOG_DIR/gemm_bench.out" 2>&1
    rc=$?
    set -e
    cat "$LOG_DIR/gemm_bench.out" || true

    if [[ $rc -ne 0 ]]; then
      log_warn "gemm_bench failed (rc=$rc); matrix section will be empty"
      MAT_TFLOPS="${MAT_TFLOPS:-0}"; MAT_EFF="[]"
    else
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

# net_bw
NET_JSON="$SCRIPT_DIR/net_bw.json"

# Force rerun if requested
force_rerun=false
case "$(printf '%s' "${RERUN_NET_BENCH:-0}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on) force_rerun=true ;;
esac

# Decide whether we need to run the binary
need_run=true
if [[ "$force_rerun" == false && -s "$NET_JSON" ]]; then
  need_run=false
fi

parse_networks_json() {
  local path="$1"
  if [[ ! -r "$path" ]]; then
    log_warn "Cannot read $path (permissions/ACL?); will rerun net_bw."
    return 1
  fi
  if command -v jq >/dev/null 2>&1; then
    NET_ARR="$(jq -c '.networks // []' "$path" 2>/dev/null)" || return 1
  elif command -v python3 >/dev/null 2>&1; then
    NET_ARR="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d.get("networks",[]), separators=(",",":")))' "$path")" || return 1
  else
    # crude fallback
    one="$(tr -d '\n' < "$path" 2>/dev/null)" || return 1
    NET_ARR="$(printf '%s' "$one" | sed -E 's/.*"networks"[[:space:]]*:[[:space:]]*(\[[[:print:]]*\]).*/\1/')" || true
    [[ -z "${NET_ARR:-}" ]] && return 1
  fi
  return 0
}

if [[ "$need_run" == false ]]; then
  log_step "Using existing net_bw.json (RERUN_NET_BENCH=0)"
  parse_networks_json "$NET_JSON"
  log_ok "networks parsed from ./net_bw.json"
else
  # Must run the benchmark
  if [[ ! -x "$BUILD_DIR/net_bw" ]]; then
    log_fail "net_bw binary missing and no reusable ./net_bw.json; networks will be []."
    NET_ARR="[]"
  else
    log_step "Running net_bw (DISPLAY_OUTPUT=0)"
    RUN_DIR="$BUILD_DIR/net_run_$(date +%Y%m%d-%H%M%S)_$$"
    mkdir -p "$RUN_DIR"
    if command -v stdbuf >/dev/null 2>&1; then STDBUF="stdbuf -oL -eL"; else STDBUF=""; fi

    set +e
    ( cd "$RUN_DIR" && DISPLAY_OUTPUT=0 $STDBUF "$BUILD_DIR/net_bw" ) >"$LOG_DIR/net_bw.out" 2>&1
    rc=$?
    set -e
    cat "$LOG_DIR/net_bw.out" || true

    if [[ $rc -ne 0 ]]; then
      log_fail "net_bw run failed (rc=$rc); networks will be []."
      NET_ARR="[]"
    elif [[ -f "$RUN_DIR/net_bw.json" ]]; then
      parse_networks_json "$RUN_DIR/net_bw.json"
      log_ok "networks parsed from $RUN_DIR/net_bw.json"
      # Optional: if no existing JSON, persist a copy for future reuse
      if [[ ! -f "$NET_JSON" ]]; then cp -f "$RUN_DIR/net_bw.json" "$NET_JSON" || true; fi
    else
      log_warn "Expected $RUN_DIR/net_bw.json not found; networks will be []."
      NET_ARR="[]"
    fi
  fi
fi

# --------------------------- 4) JSON defaults/validation ---------------------
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
json_is_obj_or_arr "${MAT_ALL:-}" || MAT_ALL="{}"
json_is_obj_or_arr "${VEC_ALL:-}" || VEC_ALL="{}"   # ← add this

# Fallback if vector multi-dtype parsing was unavailable
if [[ -z "${VEC_ALL}" || "${VEC_ALL}" == "{}" ]]; then
  VEC_ALL="{\"float32\":{\"tflops\":${VEC_TFLOPS:-0},\"gflops_efficiency\":${VEC_EFF:-[]}}}"
fi
# Fallback if matrix multi-dtype was unavailable (already present in your script)
if [[ -z "${MAT_ALL}" || "${MAT_ALL}" == "{}" ]]; then
  MAT_ALL="{\"float32\":{\"tflops\":${MAT_TFLOPS:-0},\"gflops_efficiency\":${MAT_EFF:-{}}}}"
fi

# ----------------------------- 5) Emit system.json ---------------------------
log_step "Generating ./system.json"

TMP_JSON="$BUILD_DIR/system.json.tmp"
cat > "$TMP_JSON" <<JSON
{
  "processing_mode": "no_overlap",
  "matrix": ${MAT_ALL},
  "vector": ${VEC_ALL},
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
echo -e "   Reuse previous results with ${C_DIM}RERUN_GEMM_BENCH=0 RERUN_NET_BENCH=0 ./main.sh${C_RST}"
