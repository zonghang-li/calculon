#!/usr/bin/env bash
# main.sh
# Compose the full Calculon "network" array from the three tier scripts.
# Order: tier-0 (n1g4), tier-1 (n1g8), tier-2 (n3g1).
#
# Usage:
#   bash main.sh > system_network.json
#
# Env (optional):
#   TIER0_SH, TIER1_SH, TIER2_SH  # paths to the three scripts
#   # Any other env vars are passed through to the child scripts (e.g., BANDWIDTH_GBPS, CANONICAL_SIZES, etc.)

set -euo pipefail

TIER_SH_DIR="${TIER_SH_DIR:-./}"
TIER0_SH="${TIER0_SH:-$TIER_SH_DIR/node_net_bw.n1g4.sh}"  # tier-0 / intra-node
TIER1_SH="${TIER1_SH:-$TIER_SH_DIR/node_net_bw.n1g8.sh}"  # tier-1 / cross-NUMA
TIER2_SH="${TIER2_SH:-$TIER_SH_DIR/node_net_bw.n3g1.sh}"  # tier-2 / across-node

# Ensure the scripts exist
for s in "$TIER0_SH" "$TIER1_SH" "$TIER2_SH"; do
  if [[ ! -f "$s" ]]; then
    echo "[ERR] missing script: $s" >&2
    exit 1
  fi
done

# Run each script and capture its JSON (stderr passes through for visibility)
T0_JSON="$("$TIER0_SH")"
T1_JSON="$("$TIER1_SH")"
T2_JSON="$("$TIER2_SH")"

indent_json() { sed 's/^/    /'; }

# Emit a single JSON with only one top-level key: "network"
# We avoid jq and just concatenate the three JSON objects into an array.
printf '{\n  "network": [\n'
echo "$T0_JSON," | indent_json;
echo "$T1_JSON," | indent_json;
echo "$T2_JSON" | indent_json; printf '  ]\n}\n'