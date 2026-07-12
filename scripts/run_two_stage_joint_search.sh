#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lixingfeng/UniAD_examine/heal_compress"
CONDA_SH="/home/lixingfeng/anaconda3/etc/profile.d/conda.sh"

source "${CONDA_SH}"
conda activate modelopt
cd "${ROOT}"

python -m search.cli \
  --config search/configs/two_stage_joint_search.yaml \
  --checkpoint /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth \
  --output-root tests/outputs/two_stage_joint_search \
  --gpu-id auto \
  "$@"
