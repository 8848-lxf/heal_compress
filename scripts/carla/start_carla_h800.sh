#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="${1:-2}"
CARLA_PORT="${2:-27896}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARLA_ROOT="${CARLA_ROOT:-${SCRIPT_DIR}/../../../Carla/carla}"

if [[ ! -x "${CARLA_ROOT}/CarlaUE4.sh" ]]; then
  echo "CARLA launcher not found: ${CARLA_ROOT}/CarlaUE4.sh" >&2
  exit 2
fi
if ss -H -ltn "sport = :${CARLA_PORT}" | awk 'NR == 1 { found = 1 } END { exit !found }'; then
  echo "Port ${CARLA_PORT} is already in use; refusing to replace an existing server." >&2
  exit 3
fi

cd "${CARLA_ROOT}"
exec env CUDA_VISIBLE_DEVICES="${GPU_INDEX}" ./CarlaUE4.sh \
  -prefer-nvidia -opengl -RenderOffScreen -graphicsadapter="${GPU_INDEX}" \
  -world-port="${CARLA_PORT}" -nosound -quality-level=Low
