#!/usr/bin/env bash
# main.sh
# Compose the full Calculon "network" array from the two A100 tier scripts.
# Order: tier-0 (n1g4, intra-node NVLink), tier-1 (n2g1, inter-node InfiniBand).
#
# Usage:
#   bash main.sh > system_network.json
#
# Env (optional):
#   TIER_SH_DIR            # base directory of the tier scripts (default: ./)
#   TIER0_SH, TIER1_SH     # explicit paths to the two scripts
#   # Any other env vars are passed through to the child scripts
#   # (e.g., BANDWIDTH_GBPS, CANONICAL_SIZES, FABRIC_SIZE, etc.)

set -euo pipefail

TIER_SH_DIR="${TIER_SH_DIR:-./}"
TIER0_SH="${TIER0_SH:-$TIER_SH_DIR/node_net_bw.n1g4.sh}"  # tier-0 / intra-node (NVLink clique)
TIER1_SH="${TIER1_SH:-$TIER_SH_DIR/node_net_bw.n2g1.sh}"  # tier-1 / inter-node (InfiniBand link)

# Ensure the scripts exist
for s in "$TIER0_SH" "$TIER1_SH"; do
  if [[ ! -f "$s" ]]; then
    echo "[ERR] missing script: $s" >&2
    exit 1
  fi
done

# Run each script and capture its JSON (stderr passes through for visibility)
T0_JSON="$("$TIER0_SH")"
T1_JSON="$("$TIER1_SH")"

indent_json() { sed 's/^/    /'; }

# Emit a single JSON with only one top-level key: "network"
# We avoid jq and just concatenate the two JSON objects into an array.
printf '{\n  "networks": [\n'
echo "$T0_JSON," | indent_json
echo "$T1_JSON"   | indent_json
printf '  ]\n}\n'
