#!/bin/bash
#SBATCH -J rccl-pp-perf-n2g1
#SBATCH -p faculty
#SBATCH -q zhqos
#SBATCH -A faculty-acc
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
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
JSON="$OUTDIR/rccl_pp_perf.n2g1.json"

# ---- actual run with verbose RDMA/GID discovery ----
srun --mpi=pmix -N 2 -n 2 --ntasks-per-node=1 \
     --gpus-per-task=1 --gpu-bind=closest --cpu-bind=none \
  /bin/bash -lc '
    set -e
    echo "[DBG][$(hostname)] PATH/LD_LIBRARY_PATH setup"
    export PATH="'"$ROCM_HOME"'/bin:'"$MPI_HOME"'/bin:/usr/bin:/bin"
    export LD_LIBRARY_PATH="'"$MPI_HOME"'/lib:'"$UCX_HOME"'/lib:'"$ROCM_HOME"'/lib:'"$ROCM_HOME"'/lib64:'"$ROCM_HOME"'/hip/lib:'"$ROCM_HOME"'/hsa/lib:'"$ROCM_HOME"'/rccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    # OMPI->UCX and clean BTL set
    export OMPI_MCA_pml=ucx
    export OMPI_MCA_osc=ucx
    export OMPI_MCA_btl="^openib,tcp,vader"
    export UCX_TLS=rc,sm,self
    export UCX_WARN_UNUSED_ENV_VARS=n

    # RCCL over RoCE (not sockets)
    export NCCL_NET=IB
    export NCCL_IB_DISABLE=0
    export NCCL_NET_GDR_LEVEL=${GDR_LEVEL_OVERRIDE:-0}
    export NCCL_IB_MTU=${MTU_OVERRIDE:-4096}
    export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
    export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET,IB}

    # OOB: default route iface unless overridden
    if [ -n "${IFACE_OVERRIDE:-}" ]; then
      export NCCL_SOCKET_IFNAME="${IFACE_OVERRIDE}"
      echo "[DBG][$(hostname)] OOB IFACE override: NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
    else
      IFACE_DEF=$(ip -o -4 route show default 2>/dev/null | awk "{print \$5}" | head -n1)
      [ -n "${IFACE_DEF:-}" ] && export NCCL_SOCKET_IFNAME="${IFACE_DEF}"
      echo "[DBG][$(hostname)] OOB IFACE default route: NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<unset>}"
    fi

    # -------- RDMA/ROCE discovery (very verbose) --------
    pick_roce() {
      local want_hca="${HCA_OVERRIDE:-}"
      local want_port="${PORT_OVERRIDE:-}"
      local want_gid="${GID_INDEX_OVERRIDE:-}"

      # If full override supplied, use it directly
      if [ -n "$want_hca" ] && [ -n "$want_port" ] && [ -n "$want_gid" ]; then
        export NCCL_IB_HCA="${want_hca}:${want_port}"
        export NCCL_IB_GID_INDEX="${want_gid}"
        echo "[DBG][$(hostname)] Using explicit overrides: NCCL_IB_HCA=$NCCL_IB_HCA NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX"
        return 0
      fi

      local sel_hca="" sel_port="" sel_ndev="" sel_ip="" sel_gid=""
      local h

      for h in $(ls /sys/class/infiniband 2>/dev/null | sort); do
        if [ -n "$want_hca" ] && [ "$h" != "$want_hca" ]; then
          echo "[DBG][$(hostname)]   skip HCA=$h (override=$want_hca)"; continue
        fi

        local ports_dir="/sys/class/infiniband/$h/ports"
        [ -d "$ports_dir" ] || continue
        local p
        for p in $(ls "$ports_dir" | sort -n); do
          if [ -n "$want_port" ] && [ "$p" != "$want_port" ]; then
            echo "[DBG][$(hostname)]     skip port=$p (override=$want_port)"; continue
          fi
          local ll="?"; [ -f "$ports_dir/$p/link_layer" ] && ll=$(tr -d "\n" < "$ports_dir/$p/link_layer")
          echo "[DBG][$(hostname)]   HCA=$h port=$p link_layer=$ll"
          [ "$ll" = "Ethernet" ] || { echo "[DBG][$(hostname)]     not Ethernet -> skip"; continue; }

          # Gather netdevs for this port
          declare -a ndevs=()
          if [ -d "$ports_dir/$p/gid_attrs/ndevs" ]; then
            for e in "$ports_dir/$p/gid_attrs/ndevs/"*; do
              [ -e "$e" ] || continue
              if [ -f "$e" ]; then
                # Some indices error on read (EINVAL) — guard & skip
                if val=$(cat "$e" 2>/dev/null); then
                  [ -n "$val" ] && ndevs+=( "$val" )
                else
                  continue
                fi
              else
                ndevs+=( "$(basename "$e")" )
              fi
            done
          fi
          # Dedup
          ndevs=( $(printf "%s\n" "${ndevs[@]}" | awk "{print \$1}" | sort -u) )
          echo "[DBG][$(hostname)]     netdev candidates: ${ndevs[*]:-<none>}"

          local nd
          for nd in "${ndevs[@]}"; do
            [ -n "$nd" ] || continue
            if [ ! -d "/sys/class/net/$nd" ]; then
              echo "[DBG][$(hostname)]       skip missing netdev $nd"; continue
            fi
            local state=$(ip -br link show dev "$nd" 2>/dev/null | awk "{print \$2}")
            local ip4=$(ip -o -4 addr show dev "$nd" 2>/dev/null | awk "{print \$4}" | cut -d/ -f1 | head -n1)
            echo "[DBG][$(hostname)]       netdev=$nd state=${state:-?} ipv4=${ip4:-<none>}"

            # Dump GIDs/types for this port
            local gid_dir="$ports_dir/$p/gids"
            local type_dir="$ports_dir/$p/gid_attrs/types"
            if [ -d "$gid_dir" ]; then
              for g in "$gid_dir"/*; do
                [ -f "$g" ] || continue
                local i=$(basename "$g")
                local gv=$(cat "$g" 2>/dev/null)
                local tv=$(cat "$type_dir/$i" 2>/dev/null || echo "?")
              done
            fi

            # If we have IPv4, try to find IPv4-mapped GID index first
            local idx=""
            if [ -n "$ip4" ]; then
              for g in "$gid_dir"/*; do
                [ -f "$g" ] || continue
                local i=$(basename "$g")
                local gv=$(cat "$g" 2>/dev/null)
                local tv=$(cat "$type_dir/$i" 2>/dev/null || echo "?")
                if echo "$gv" | grep -qi "::ffff:$ip4$"; then
                  idx="$i"; echo "[DBG][$(hostname)]         MATCH IPv4-mapped GID idx=$i"
                  break
                fi
              done
            fi

            # Fallback: first non-link-local RoCE v2
            if [ -z "$idx" ]; then
              for g in "$gid_dir"/*; do
                [ -f "$g" ] || continue
                local i=$(basename "$g")
                local gv=$(cat "$g" 2>/dev/null)
                local tv=$(cat "$type_dir/$i" 2>/dev/null || echo "?")
                if [ "$tv" = "RoCE v2" ] && ! echo "$gv" | grep -qi "^fe80:"; then
                  idx="$i"; echo "[DBG][$(hostname)]         PICK RoCE v2 non-link-local idx=$i"
                  break
                fi
              done
            fi

            # Last resort: any RoCE v2 (even link-local)
            if [ -z "$idx" ]; then
              for g in "$gid_dir"/*; do
                [ -f "$g" ] || continue
                local i=$(basename "$g")
                local tv=$(cat "$type_dir/$i" 2>/dev/null || echo "?")
                if [ "$tv" = "RoCE v2" ]; then
                  idx="$i"; echo "[DBG][$(hostname)]         FALLBACK any RoCE v2 idx=$i"
                  break
                fi
              done
            fi

            if [ -n "$idx" ]; then
              sel_hca="$h"; sel_port="$p"; sel_ndev="$nd"; sel_ip="$ip4"; sel_gid="$idx"
              break 3
            fi
          done
        done
      done

      if [ -z "$sel_hca" ]; then
        echo "[ERR][$(hostname)] No RoCE HCA/port/netdev with usable GID found"
        return 1
      fi

      export NCCL_IB_HCA="${sel_hca}:${sel_port}"
      export NCCL_IB_GID_INDEX="${sel_gid}"
      # Use discovered netdev for OOB if IFACE_OVERRIDE not set
      if [ -z "${IFACE_OVERRIDE:-}" ] && [ -n "$sel_ndev" ]; then
        export NCCL_SOCKET_IFNAME="$sel_ndev"
      fi

      echo "[SEL][$(hostname)] RDMA selected: HCA=${NCCL_IB_HCA} IFACE=${NCCL_SOCKET_IFNAME:-<unset>} IP=${sel_ip:-<none>} GID_INDEX=${NCCL_IB_GID_INDEX} IB_MTU=${NCCL_IB_MTU}"
      return 0
    }

    if ! pick_roce; then
      echo "[ERR][$(hostname)] RDMA selection failed"; exit 3
    fi

    # Mirror GID selection to UCX as well (helps OMPI/UCX traffic be consistent)
    export UCX_IB_GID_INDEX="${NCCL_IB_GID_INDEX}"

    echo "[DBG][$(hostname)] ldd $(basename "'"$RCCL_SR_PERF"'") (mpi/ucx/rocm lines):"
    ldd "'"$RCCL_SR_PERF"'" | egrep -i "mpi|ucx|rccl|hip|hsa" || true

    echo "[DBG][$(hostname)] OMPI/UCX versions:"
    (ompi_info -V 2>/dev/null | head -n1 || true)
    (ucx_info -v 2>/dev/null | head -n3 || true)

    # ---- run the test ----
    exec "'"$RCCL_SR_PERF"'" -b 8 -e 64M -f 2 -g 1 -Z json -x "'"$JSON"'"
  '

echo "[$(date)] Done. JSON saved to: $JSON"
