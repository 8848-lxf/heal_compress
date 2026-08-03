#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="outputs/latency_lut/v11_mixed_precision_lut_dataset_trt_full"
LOG_DIR="$RUN_DIR/dual_gpu_worker_logs"
TRT_ROOT="${TENSORRT_ROOT}"
PLUGIN="./tests/quant_deploy/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so"

mkdir -p "$LOG_DIR"

export LD_LIBRARY_PATH="$TRT_ROOT/lib:$TRT_ROOT/targets/x86_64-linux-gnu/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="../..:.:../../HEAL:${PYTHONPATH:-}"

${CONDA_PREFIX}/bin/python tools/latency_lut/run_v11_dual_gpu_profile_workers.py \
  --source-dir "$RUN_DIR" \
  --max-subnets 50 \
  --precision-profiles-per-subnet 4 \
  --gpus 6,7 \
  --max-workers 2 \
  --profile-sampling-policy stratified \
  --profile-seed 20260708 \
  --build-engines true \
  --eval-engines true \
  --smoke-frames 5 \
  --warmup-frames 100 \
  --eval-frames 1000 \
  --ap-thresholds 0.03,0.30,0.50,0.70 \
  --resume \
  --overwrite-invalid-profiles \
  --quarantine-invalid-profile \
  --max-consecutive-failures 5 \
  --max-total-gate-failures 10 \
  --trt-build-timeout-seconds 600 \
  --trt-root "$TRT_ROOT" \
  --plugin "$PLUGIN" \
  --log-dir "$LOG_DIR"
