#!/usr/bin/env bash
set -euo pipefail

ROOT=.
PYTHON=${CONDA_PREFIX}/bin/python
HEAL_ROOT=../../HEAL
TRT_ROOT=${TENSORRT_ROOT}
PLUGIN=$ROOT/quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so

OUT=$ROOT/outputs/latency_lut/v12_combined_lut_dataset_300frames
LOGDIR=$OUT/logs

export PYTHONPATH=../..:$ROOT:$HEAL_ROOT
export CUDA_HOME=${CONDA_PREFIX}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$TRT_ROOT/lib:${LD_LIBRARY_PATH:-}

cd "$ROOT"
mkdir -p "$LOGDIR"

prepare() {
  local ts
  ts=$(date +%Y%m%d_%H%M%S)
  local log="$LOGDIR/prepare_v12_${ts}.log"

  echo "[INFO] Starting v12 dataset prepare..."
  echo "[INFO] Log: $log"

  nohup "$PYTHON" tools/latency_lut/prepare_v12_combined_lut_dataset.py \
    --historical-v11-dir outputs/latency_lut/v11_random_deployment_aware_subnets_dryrun_v2 \
    --output-dir "$OUT" \
    --new-v12-subnets 32 \
    --subnets-per-bin 8 \
    --overwrite \
    > "$log" 2>&1 &

  echo "[INFO] Prepare PID: $!"
  echo "[INFO] Follow log with:"
  echo "tail -f \"$log\""
}

run_workers() {
  local ts
  ts=$(date +%Y%m%d_%H%M%S)
  local log="$LOGDIR/v12_4gpu_300frame_workers_${ts}.log"

  echo "[INFO] Starting v12 4-GPU engine/eval workers..."
  echo "[INFO] GPUs: 1,2,6,7"
  echo "[INFO] Eval frames: 300"
  echo "[INFO] Log: $log"

  nohup "$PYTHON" tools/latency_lut/run_v12_lut_engine_eval_workers.py \
    --dataset-dir "$OUT" \
    --gpus 1,2,6,7 \
    --workers-per-gpu 1 \
    --eval-frame-count 300 \
    --resume \
    --skip-existing-success \
    --build-engine \
    --run-structure-check \
    --run-precision-check \
    --run-smoke \
    --run-eval \
    --plugin-path "$PLUGIN" \
    --trt-root "$TRT_ROOT" \
    --require-engine-structure-check \
    --require-precision-realization-check \
    --require-validation-dataloader \
    --forbid-synthetic-eval \
    --max-retries 1 \
    > "$log" 2>&1 &

  echo "[INFO] Worker PID: $!"
  echo "[INFO] Follow log with:"
  echo "tail -f \"$log\""
}

status() {
  echo "[INFO] Active v12 workers / trtexec:"
  pgrep -af "[r]un_v12_lut_engine_eval_workers.py|[t]rtexec" || true

  echo
  echo "[INFO] Current label count:"
  find "$OUT/subnets" -name lut_sample_label.json 2>/dev/null | wc -l || true
}

tail_log() {
  echo "[INFO] Latest logs:"
  ls -t "$LOGDIR"/*.log 2>/dev/null | head -n 5 || true
  echo
  local latest
  latest=$(ls -t "$LOGDIR"/*.log 2>/dev/null | head -n 1 || true)
  if [[ -n "${latest:-}" ]]; then
    echo "[INFO] Tailing: $latest"
    tail -f "$latest"
  else
    echo "[WARN] No log file found."
  fi
}

check_gate() {
  local report="$OUT/prepare_v12_combined_lut_dataset_report.md"
  if [[ -f "$report" ]]; then
    echo "[INFO] Prepare report:"
    cat "$report"
  else
    echo "[WARN] Prepare report not found: $report"
  fi
}

count_labels() {
  "$PYTHON" - <<PY
import json, pathlib, collections
root = pathlib.Path("$OUT")
labels = []
for p in root.glob("subnets/subnet_*/profile_*/lut_sample_label.json"):
    try:
        labels.append(json.loads(p.read_text()))
    except Exception:
        pass

cnt = collections.Counter()
for x in labels:
    cnt["total"] += 1
    cnt["available"] += bool(x.get("label_available"))
    cnt["eval_300"] += int(x.get("evaluated_frames", 0)) == 300
    cnt["historical_v11"] += x.get("source_subset") == "historical_v11"
    cnt["new_v12_deblock_protected"] += x.get("source_subset") == "new_v12_deblock_protected"
    cnt["deblock_output_pruned"] += bool(x.get("deblock_output_pruned"))

print(dict(cnt))
PY
}

collect() {
  if [[ ! -f tools/latency_lut/collect_v12_lut_samples.py ]]; then
    echo "[WARN] collect_v12_lut_samples.py not found."
    echo "[WARN] Skipping collection."
    exit 0
  fi

  "$PYTHON" tools/latency_lut/collect_v12_lut_samples.py \
    --dataset-dir "$OUT" \
    --output-json "$OUT/lut_samples.json" \
    --output-csv "$OUT/lut_samples.csv"

  echo "[INFO] Wrote:"
  echo "$OUT/lut_samples.json"
  echo "$OUT/lut_samples.csv"
}

stop() {
  echo "[INFO] Stopping v12 workers and trtexec..."
  pkill -TERM -f "run_v12_lut_engine_eval_workers.py|trtexec" || true
  sleep 2
  pgrep -af "[r]un_v12_lut_engine_eval_workers.py|[t]rtexec" || true
}

help_msg() {
  cat <<USAGE
Usage:
  bash scripts/v12_lut_300frames.sh prepare      # 后台准备 combined dataset + gate，不构建 engine
  bash scripts/v12_lut_300frames.sh check-gate   # 查看 prepare gate 报告
  bash scripts/v12_lut_300frames.sh run          # 后台启动 GPU 1,2,6,7 构建 engine + 300-frame eval
  bash scripts/v12_lut_300frames.sh status       # 查看进程和 label 数量
  bash scripts/v12_lut_300frames.sh tail         # tail 最新日志
  bash scripts/v12_lut_300frames.sh count        # 统计 lut_sample_label.json
  bash scripts/v12_lut_300frames.sh collect      # 汇总 lut_samples.json/csv
  bash scripts/v12_lut_300frames.sh stop         # 停止 worker/trtexec
USAGE
}

case "${1:-help}" in
  prepare) prepare ;;
  run) run_workers ;;
  status) status ;;
  tail) tail_log ;;
  check-gate) check_gate ;;
  count) count_labels ;;
  collect) collect ;;
  stop) stop ;;
  help|-h|--help) help_msg ;;
  *) echo "[ERROR] Unknown command: $1"; help_msg; exit 1 ;;
esac
