#!/usr/bin/env bash
set -euo pipefail

cd /home/lixingfeng/UniAD_examine/heal_compress

python tests/run_l1_grouped_conv_ablation.py \
  --checkpoint /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth \
  --gpu-id auto \
  --output-root tests/outputs \
  --experiment-name l1_grouped_conv_ablation \
  --ratios 0.25 0.50 0.75 \
  --modes shared_local_mean independent_group_topk \
  --run-prune true \
  --run-eval true \
  --run-latency true \
  --max-frames -1 \
  --warmup-frames 50 \
  --rounds 3
