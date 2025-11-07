#!/usr/bin/env bash
# node_net_bw.n1g4.sh  (tier-0 / intra-node)
# Build the tier-0 network tier entry for Calculon from rccl-tests outputs.
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
# - Reads two rccl-tests JSON files:
#     * AR_JSON  = rccl_ar_perf.n1g4.json          (AllReduce, 1 node × 4 GPUs)
#     * PP_JSON  = rccl_pp_perf.n1g2.intra.json    (SendRecv, 1 rank × 2 GPUs on same NUMA)
# - Computes per-MiB efficiency tables for AR and PP:
#     1) Bin rows by MiB = floor(size_bytes / 2^20).
#     2) For each MiB bin, take the maximum observed busBw (GB/s) across valid rows (wrong ∈ {"0","N/A"}).
#     3) “Snap” each canonical size S (MiB) to the best bin ≤ S (fallback to the smallest available bin).
#     4) Efficiency = busBw / BANDWIDTH_GBPS, clamped to [0,1], rounded down to 4 decimals.
# - Estimates link latency from PP_JSON automatically:
#     * Fits time_us(size_bytes) ≈ L_us + size_bytes * slope on small messages (≤ 4 KiB).
#     * Outputs latency = L_us * 1e-6 (seconds). This replaces any hard-coded LATENCY.
#
# UNITS & ASSUMPTIONS
# - rccl-tests busBw is in GB/s (decimal 1e9). BANDWIDTH_GBPS is also GB/s.
# - message_size_units for the efficiency tables is MiB.
# - “Latency” here is a per-message one-way estimate derived from the small-size intercept.
#
# REQUIREMENTS
# - Python 3.x available on PATH. No jq required.
# - rccl-tests JSONs must exist at the paths below (or override via env).
#
# ENVIRONMENT VARIABLES (all optional)
#   RCCL_OUTDIR                 [default: ./rccl_tests]
#   RCCL_AR_PERF_N1G4_JSON      [default: $RCCL_OUTDIR/rccl_ar_perf.n1g4.json]
#   RCCL_PP_PERF_N1G2_INTRA_JSON[default: $RCCL_OUTDIR/rccl_pp_perf.n1g2.intra.json]
#   BANDWIDTH_GBPS              [default: 100.0]      # theoretical intra-node link GB/s
#   CANONICAL_SIZES             [default: 128,96,64,32,16,8,4,2,1,0]  # MiB breakpoints for both AR/PP
#   MIN_MIB                     [default: 0]          # drop bins < MIN_MIB; keep 0 to populate the 0 bucket
#   FABRIC_SIZE                 [default: 4]          # number of GPUs in the tier (NUMA group)
#   PROCESSOR_USAGE             [default: 0.04]       # GPU time fraction to drive comms
#   MUST_BE_FILLED              [default: true]       # pipeline saturation assumption
#
# USAGE
#   bash node_net_bw.n1g4.sh > tier0_network.json
#   BANDWIDTH_GBPS=100 CANONICAL_SIZES=128,64,32,16,8,4,2,1,0 bash node_net_bw.n1g4.sh > out.json
#
# NOTES
# - Identical efficiency values at multiple large sizes are expected: “snap to best ≤ S” reuses the same
#   best bin when the maximum stays flat near peak.
# - If you want the 0-MiB bucket to reflect sub-1MiB behavior, keep MIN_MIB=0; set MIN_MIB=1 to mirror 1-MiB.

set -euo pipefail

# --- defaults (override via env) ---
RCCL_OUTDIR="${RCCL_OUTDIR:-./rccl_tests}"

# Tier-0 inputs
AR_JSON="${RCCL_AR_PERF_N1G4_JSON:-$RCCL_OUTDIR/rccl_ar_perf.n1g4.json}"
PP_JSON="${RCCL_PP_PERF_N1G2_INTRA_JSON:-$RCCL_OUTDIR/rccl_pp_perf.n1g2.intra.json}"

# Theoretical hardware bandwidth in GB/s (busBw in JSON is also GB/s)
BANDWIDTH_GBPS="${BANDWIDTH_GBPS:-100.0}"

# One canonical list for both AR and PP (MiB). Includes 96 and 0 as you showed.
CANONICAL_SIZES="${CANONICAL_SIZES:-128,96,64,32,16,8,4,2,1,0}"

# Keep sub-1MiB bucket so "0" snaps properly
MIN_MIB="${MIN_MIB:-0}"

# Fixed tier fields (override if needed)
FABRIC_SIZE="${FABRIC_SIZE:-4}"
PROCESSOR_USAGE="${PROCESSOR_USAGE:-0.04}"
MUST_BE_FILLED="${MUST_BE_FILLED:-true}"

python3 - "$AR_JSON" "$PP_JSON" "$BANDWIDTH_GBPS" "$CANONICAL_SIZES" "$MIN_MIB" "$FABRIC_SIZE" "$PROCESSOR_USAGE" "$MUST_BE_FILLED" << 'PY'
import sys, json, math, collections

if len(sys.argv) < 9:
    sys.exit("usage: script AR_JSON PP_JSON BANDWIDTH_GBPS CANONICAL_SIZES MIN_MIB FABRIC_SIZE PROCESSOR_USAGE MUST_BE_FILLED")

ar_path, pp_path, bw_s, canon_s, min_mib_s, fabric_size_s, proc_usage_s, must_fill_s = sys.argv[1:10]

bandwidth = float(bw_s)                    # GB/s
canon = [int(x) for x in canon_s.split(',') if x.strip()]
min_mib = int(min_mib_s)
fabric_size = int(fabric_size_s)
processor_usage = float(proc_usage_s)
must_be_filled = (must_fill_s.strip().lower() == "true")

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
        # fallback: min observed time
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

def load_list(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except FileNotFoundError:
        return []

def best_map(rows):
    """Return {MiB: best busBw(GB/s)} for valid rows, MiB >= MIN_MIB."""
    m = {}
    for r in rows:
        if r.get('wrong') not in ('0', 'N/A'):
            continue
        mib = int(r.get('size', 0)) // 1048576
        if mib < min_mib:
            continue
        bw = float(r.get('busBw', 0.0))  # GB/s from nccl-tests
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
