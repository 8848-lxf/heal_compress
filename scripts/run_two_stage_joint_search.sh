#!/usr/bin/env bash
set -euo pipefail

ROOT="."
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"

source "${CONDA_SH}"
conda activate modelopt
cd "${ROOT}"

python -m search.cli \
  --config search/configs/two_stage_joint_search.yaml \
  --checkpoint ${MODEL_ROOT}/lidar_pyramid/net_epoch_bestval_at17.pth \
  --output-root tests/outputs/two_stage_joint_search \
  --gpu-id auto \
  "$@"
