#!/usr/bin/env bash
set -euo pipefail

cd .

python tests/run_l1_grouped_conv_ablation.py \
  --checkpoint ${MODEL_ROOT}/lidar_pyramid/net_epoch_bestval_at17.pth \
  --gpu-id auto \
  --output-root tests/outputs \
  --experiment-name l1_grouped_conv_ablation_debug \
  --ratios 0.25 0.50 0.75 \
  --modes shared_local_mean independent_group_topk \
  --run-prune true \
  --run-eval true \
  --run-latency true \
  --max-frames 50 \
  --warmup-frames 20 \
  --rounds 1
