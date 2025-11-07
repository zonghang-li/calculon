#!/usr/bin/env bash
# node_net_bw.n3g1.sh  (tier-2 / across-node)
# Build the tier-2 network tier entry for Calculon from rccl-tests outputs.
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
#     * AR_JSON  = rccl_ar_perf.n3g1.json       (AllReduce, 3 ranks × 1 GPU each, across nodes)
#     * PP_JSON  = rccl_pp_perf.n2g1.json       (SendRecv, 2 nodes × 1 GPU)
# - Bins by MiB, takes max busBw per bin, snaps to canonical MiB, eff = busBw / BANDWIDTH_GBPS.
# - Estimates latency from PP_JSON by linear fit on tiny sizes (≤ 4 KiB).
#
# UNITS
# - rccl-tests busBw is GB/s (decimal 1e9). BANDWIDTH_GBPS is also GB/s.
#
# REQUIREMENTS
# - Python 3.x on PATH. No jq required.
#
# ENV (override as needed)
#   RCCL_OUTDIR                 [default: ./rccl_tests]
#   RCCL_AR_PERF_N3G1_JSON      [default: $RCCL_OUTDIR/rccl_ar_perf.n3g1.json]
#   RCCL_PP_PERF_N2G1_JSON      [default: $RCCL_OUTDIR/rccl_pp_perf.n2g1.json]
#   BANDWIDTH_GBPS              [default: 12.5]   # cross-node link GB/s (100 Gbps NIC). 200 Gbps → 25, 400 → 50, etc.
#   CANONICAL_SIZES             [default: 128,96,64,32,16,8,4,2,1,0]
#   MIN_MIB                     [default: 0]
#   FABRIC_SIZE                 [default: 2048]      # participants in this tier; override to your typical cross-node group
#   PROCESSOR_USAGE             [default: 0.04]
#   MUST_BE_FILLED              [default: false]
#
# USAGE
#   bash node_net_bw.n3g1.sh > tier2_network.json
#   BANDWIDTH_GBPS=25 FABRIC_SIZE=16 bash node_net_bw.n3g1.sh > out.json

set -euo pipefail

# --- defaults (override via env) ---
RCCL_OUTDIR="${RCCL_OUTDIR:-./rccl_tests}"

# Tier-2 inputs
AR_JSON="${RCCL_AR_PERF_N3G1_JSON:-$RCCL_OUTDIR/rccl_ar_perf.n3g1.json}"
PP_JSON="${RCCL_PP_PERF_N2G1_JSON:-$RCCL_OUTDIR/rccl_pp_perf.n2g1.json}"

# Cross-node theoretical bandwidth in GB/s (busBw in JSON is also GB/s)
BANDWIDTH_GBPS="${BANDWIDTH_GBPS:-12.5}"

# One canonical list for both AR and PP (MiB)
CANONICAL_SIZES="${CANONICAL_SIZES:-128,96,64,32,16,8,4,2,1,0}"

# Keep sub-1MiB bucket so "0" snaps properly
MIN_MIB="${MIN_MIB:-0}"

# Fixed tier fields (override if needed)
FABRIC_SIZE="${FABRIC_SIZE:-2048}"
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
