#!/usr/bin/env bash
# node_net_bw.n1g8.sh  (tier-1 / cross-NUMA)
# Build the tier-1 network tier entry for Calculon from rccl-tests outputs.
# Produces a single JSON object with fields in the exact order required by the config:
#   {
#     "bandwidth": <GB/s>,
#     "pp_efficiency": [[MiB, eff], ...],
#     "ar_efficiency": [[MiB, eff], ...],
#     "size": <int>,
#     "latency": <seconds>,
#     "ops": { "p2p":[1.0,null], "reduce_scatter":[1.0,-1], "all_gather":[1.0,-1], "all_reduce":[2.0,-1] },
#     "must_be_filled": <bool>,
#     "processor_usage": <float>
#   }
#
# WHAT IT DOES
# - Reads:
#     * AR_JSON  = rccl_ar_perf.n1g8.json         (AllReduce, 1 node × 8 GPUs)
#     * PP_JSON  = rccl_pp_perf.n1g2.xnuma.json   (SendRecv, 1 rank × 2 GPUs across NUMA)
# - Computes per-MiB efficiency (bin by MiB, take max busBw per bin, snap to canonical MiB, eff = busBw/BANDWIDTH_GBPS).
# - Estimates latency from PP_JSON via linear fit on small sizes (≤ 4 KiB).
#
# UNITS
# - rccl-tests busBw is GB/s; BANDWIDTH_GBPS is also GB/s.
#
# REQUIREMENTS
# - Python 3.x on PATH. No jq required.
#
# ENV (override as needed)
#   RCCL_OUTDIR                   [default: ./rccl_tests]
#   RCCL_AR_PERF_N1G8_JSON        [default: $RCCL_OUTDIR/rccl_ar_perf.n1g8.json]
#   RCCL_PP_PERF_N1G2_XNUMA_JSON  [default: $RCCL_OUTDIR/rccl_pp_perf.n1g2.xnuma.json]
#   BANDWIDTH_GBPS                [default: 32.0]        # theoretical cross-NUMA GB/s (e.g., PCIe Gen4 x16)
#   CANONICAL_SIZES               [default: 128,96,64,32,16,8,4,2,1,0]
#   MIN_MIB                       [default: 0]
#   FABRIC_SIZE                   [default: 8]           # GPUs in this tier
#   PROCESSOR_USAGE               [default: 0.04]
#   MUST_BE_FILLED                [default: false]
#
# USAGE
#   bash node_net_bw.n1g8.sh > tier1_network.json
#   BANDWIDTH_GBPS=32 CANONICAL_SIZES=128,64,32,16,8,4,2,1,0 bash node_net_bw.n1g8.sh > out.json

set -euo pipefail

# --- defaults (override via env) ---
RCCL_OUTDIR="${RCCL_OUTDIR:-./rccl_tests}"

# Tier-1 inputs
AR_JSON="${RCCL_AR_PERF_N1G8_JSON:-$RCCL_OUTDIR/rccl_ar_perf.n1g8.json}"
PP_JSON="${RCCL_PP_PERF_N1G2_XNUMA_JSON:-$RCCL_OUTDIR/rccl_pp_perf.n1g2.xnuma.json}"

# Theoretical hardware bandwidth in GB/s (busBw in JSON is also GB/s)
BANDWIDTH_GBPS="${BANDWIDTH_GBPS:-32.0}"

# One canonical list for both AR and PP (MiB)
CANONICAL_SIZES="${CANONICAL_SIZES:-128,96,64,32,16,8,4,2,1,0}"

# Keep sub-1MiB bucket so "0" snaps properly
MIN_MIB="${MIN_MIB:-0}"

# Fixed tier fields (override if needed)
FABRIC_SIZE="${FABRIC_SIZE:-8}"
PROCESSOR_USAGE="${PROCESSOR_USAGE:-0.04}"
MUST_BE_FILLED="${MUST_BE_FILLED:-false}"

python3 - "$AR_JSON" "$PP_JSON" "$BANDWIDTH_GBPS" "$CANONICAL_SIZES" "$MIN_MIB" "$FABRIC_SIZE" "$PROCESSOR_USAGE" "$MUST_BE_FILLED" << 'PY'
import sys, json, math, collections

if len(sys.argv) < 8:
    sys.exit("usage: script AR_JSON PP_JSON BANDWIDTH_GBPS CANONICAL_SIZES MIN_MIB FABRIC_SIZE PROCESSOR_USAGE MUST_BE_FILLED")

ar_path, pp_path, bw_s, canon_s, min_mib_s, fabric_size_s, proc_usage_s, must_fill_s = sys.argv[1:9]

bandwidth = float(bw_s)                    # GB/s
canon = [int(x) for x in canon_s.split(',') if x.strip()]
min_mib = int(min_mib_s)
fabric_size = int(fabric_size_s)
processor_usage = float(proc_usage_s)
must_be_filled = (must_fill_s.strip().lower() == "true")

def load_list(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except FileNotFoundError:
        return []

def fit_latency_from_pp(rows, max_bytes=4096):
    """Return latency (seconds) via linear fit on small sizes."""
    pts = []
    for r in rows:
        if r.get("wrong") not in ("0","N/A"):
            continue
        s = int(r.get("size", 0))
        t = float(r.get("time", 0.0))  # microseconds
        if s > 0 and t > 0:
            pts.append((s, t))
    # keep tiny messages (latency-dominated)
    pts = [p for p in pts if p[0] <= max_bytes] or sorted(pts, key=lambda x:x[0])[:6]
    if len(pts) < 2:
        L_us = min((t for _, t in pts), default=0.0)
        return max(0.0, L_us) * 1e-6

    n  = len(pts)
    sx = sum(s for s, _ in pts)
    sy = sum(t for _, t in pts)
    sxx = sum(s*s for s, _ in pts)
    sxy = sum(s*t for s, t in pts)
    den = n*sxx - sx*sx
    if den <= 0:
        L_us = min(t for _, t in pts)
    else:
        slope = (n*sxy - sx*sy) / den
        L_us  = (sy - slope*sx) / n
    L_us = max(0.0, L_us)
    return L_us * 1e-6  # seconds

def best_map(rows):
    """Return {MiB: best busBw(GB/s)} for valid rows, MiB >= MIN_MIB."""
    m = {}
    for r in rows:
        if r.get('wrong') not in ('0', 'N/A'):
            continue
        mib = int(r.get('size', 0)) // 1048576
        if mib < min_mib:
            continue
        bw = float(r.get('busBw', 0.0))  # GB/s from rccl-tests
        if mib not in m or bw > m[mib]:
            m[mib] = bw
    return m

def snap_to_eff(bmap):
    """For each canonical S, pick best MiB <= S (fallback to smallest). Convert to efficiency."""
    if not bmap:
        return [[S, 0.0] for S in canon]
    avail = sorted(bmap.keys())
    out = []
    for S in canon:
        cand = [a for a in avail if a <= S]
        key = cand[-1] if cand else avail[0]
        eff = (bmap.get(key, 0.0) / bandwidth) if bandwidth > 0 else 0.0
        eff = 0.0 if eff < 0 else (1.0 if eff > 1.0 else eff)
        eff = math.floor(eff * 10000) / 10000.0  # round down to 4 decimals
        out.append([S, eff])
    return out

ar_rows = load_list(ar_path)
pp_rows = load_list(pp_path)

latency = fit_latency_from_pp(pp_rows)

ar_eff = snap_to_eff(best_map(ar_rows))
pp_eff = snap_to_eff(best_map(pp_rows))

# Build output in the exact field order requested
out = collections.OrderedDict()
out["bandwidth"] = bandwidth
out["pp_efficiency"] = pp_eff
out["ar_efficiency"] = ar_eff
out["size"] = fabric_size
out["latency"] = latency
out["ops"] = {
    "p2p": [1.0, None],
    "reduce_scatter": [1.0, -1],
    "all_gather": [1.0, -1],
    "all_reduce": [2.0, -1]
}
out["must_be_filled"] = must_be_filled
out["processor_usage"] = processor_usage

print(json.dumps(out, indent=2))
PY
