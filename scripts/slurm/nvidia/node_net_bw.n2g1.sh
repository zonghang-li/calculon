#!/usr/bin/env bash
# node_net_bw.n2g1.sh  (inter-node / 2 nodes × 1 GPU)
# Build the inter-node network tier entry from nccl-tests outputs.
# Produces a single JSON object with fields:
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
# INPUTS (nccl-tests JSON; non-aggregated format)
#   - AR_JSON = nccl_ar_perf.n2g1.json   (all_reduce_perf, 2 nodes × 1 GPU)
#   - PP_JSON = nccl_pp_perf.n2g1.json   (sendrecv_perf, 2 nodes × 1 GPU)
#
# ENVIRONMENT VARIABLES (optional)
#   RCCL_OUTDIR                 [default: ./nccl_tests]
#   NCCL_AR_PERF_N2G1_JSON      [default: $RCCL_OUTDIR/nccl_ar_perf.n2g1.json]
#   NCCL_PP_PERF_N2G1_JSON      [default: $RCCL_OUTDIR/nccl_pp_perf.n2g1.json]
#   BANDWIDTH_GBPS              [default: 10.0]      # theoretical inter-node GB/s
#   CANONICAL_SIZES             [default: 128,96,64,32,16,8,4,2,1,0]
#   MIN_MIB                     [default: 0]
#   FABRIC_SIZE                 [default: 2048]         # 2048 nodes
#   PROCESSOR_USAGE             [default: 0.03]
#   MUST_BE_FILLED              [default: false]
#
# USAGE
#   bash node_net_bw.n2g1.sh > tier1_network.json
#   BANDWIDTH_GBPS=10 bash node_net_bw.n2g1.sh > out.json

set -euo pipefail

RCCL_OUTDIR="${RCCL_OUTDIR:-./nccl_tests}"

AR_JSON="${NCCL_AR_PERF_N2G1_JSON:-$RCCL_OUTDIR/nccl_ar_perf.n2g1.json}"
PP_JSON="${NCCL_PP_PERF_N2G1_JSON:-$RCCL_OUTDIR/nccl_pp_perf.n2g1.json}"

BANDWIDTH_GBPS="${BANDWIDTH_GBPS:-10.0}"
CANONICAL_SIZES="${CANONICAL_SIZES:-128,96,64,32,16,8,4,2,1,0}"
MIN_MIB="${MIN_MIB:-0}"
FABRIC_SIZE="${FABRIC_SIZE:-2048}"
PROCESSOR_USAGE="${PROCESSOR_USAGE:-0.03}"
MUST_BE_FILLED="${MUST_BE_FILLED:-false}"

python3 - "$AR_JSON" "$PP_JSON" "$BANDWIDTH_GBPS" "$CANONICAL_SIZES" "$MIN_MIB" "$FABRIC_SIZE" "$PROCESSOR_USAGE" "$MUST_BE_FILLED" << 'PY'
import sys, json, math, collections

if len(sys.argv) < 9:
    sys.exit("usage: script AR_JSON PP_JSON BANDWIDTH_GBPS CANONICAL_SIZES MIN_MIB FABRIC_SIZE PROCESSOR_USAGE MUST_BE_FILLED")

ar_path, pp_path, bw_s, canon_s, min_mib_s, fabric_size_s, proc_usage_s, must_fill_s = sys.argv[1:9]

bandwidth = float(bw_s)                    # GB/s
canon = [int(x) for x in canon_s.split(',') if x.strip()]
min_mib = int(min_mib_s)
fabric_size = int(fabric_size_s)
processor_usage = float(proc_usage_s)
must_be_filled = (must_fill_s.strip().lower() == "true")


def load_nccl_rows(path):
    """Convert nccl-tests JSON into a flat list of rows with size, time, busBw, wrong."""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        return []

    res = data.get("results", [])
    rows = []
    for r in res:
        try:
            size = int(r.get("size", 0))
        except (TypeError, ValueError):
            continue
        if size <= 0:
            continue

        outp = r.get("out_of_place") or {}
        inp  = r.get("in_place") or {}

        def extract(rec):
            try:
                bw = float(rec.get("bus_bw", 0.0))
            except (TypeError, ValueError):
                bw = 0.0
            try:
                t = float(rec.get("time", 0.0))  # microseconds
            except (TypeError, ValueError):
                t = 0.0
            raw_wrong = rec.get("nwrong")
            # Treat None / 0 / 0.0 / "0" as good
            if raw_wrong in (None, 0, 0.0, "0", "0.0"):
                wrong = "0"
            else:
                wrong = str(raw_wrong)
            return bw, t, wrong

        bw_o, t_o, w_o = extract(outp)
        bw_i, t_i, w_i = extract(inp)

        if bw_i > bw_o:
            bw, t, wrong = bw_i, t_i, w_i
        else:
            bw, t, wrong = bw_o, t_o, w_o

        rows.append({"size": size, "time": t, "busBw": bw, "wrong": wrong})
    return rows


def fit_latency_from_pp(rows, max_bytes=4096):
    """Return latency (seconds) via linear fit on small sizes."""
    pts = []
    for r in rows:
        if r.get("wrong") not in ("0", "N/A"):
            continue
        try:
            s = int(r.get("size", 0))
            t = float(r.get("time", 0.0))  # microseconds
        except (TypeError, ValueError):
            continue
        if s > 0 and t > 0:
            pts.append((s, t))

    # Prefer latency-dominated region
    pts = [p for p in pts if p[0] <= max_bytes] or sorted(pts, key=lambda x: x[0])[:6]
    if len(pts) < 2:
        L_us = min((t for _, t in pts), default=0.0)
        return max(0.0, L_us) * 1e-6

    n = len(pts)
    sx = sum(s for s, _ in pts)
    sy = sum(t for _, t in pts)
    sxx = sum(s * s for s, _ in pts)
    sxy = sum(s * t for s, t in pts)
    den = n * sxx - sx * sx
    if den <= 0:
        L_us = min(t for _, t in pts)
    else:
        slope = (n * sxy - sx * sy) / den
        L_us = (sy - slope * sx) / n
    L_us = max(0.0, L_us)
    return L_us * 1e-6  # seconds


def best_map(rows):
    """Return {MiB: best busBw(GB/s)} for valid rows, MiB >= MIN_MIB."""
    m = {}
    for r in rows:
        if r.get("wrong") not in ("0", "N/A"):
            continue
        try:
            size_bytes = int(r.get("size", 0))
        except (TypeError, ValueError):
            continue
        mib = size_bytes // 1048576
        if mib < min_mib:
            continue
        try:
            bw = float(r.get("busBw", 0.0))
        except (TypeError, ValueError):
            bw = 0.0
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
        eff = math.floor(eff * 10000) / 10000.0  # floor to 4 decimals
        out.append([S, eff])
    return out


ar_rows = load_nccl_rows(ar_path)
pp_rows = load_nccl_rows(pp_path)

latency = fit_latency_from_pp(pp_rows)

ar_eff = snap_to_eff(best_map(ar_rows))
pp_eff = snap_to_eff(best_map(pp_rows))

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
    "all_reduce": [2.0, -1],
}
out["must_be_filled"] = must_be_filled
out["processor_usage"] = processor_usage

print(json.dumps(out, indent=2))
PY
