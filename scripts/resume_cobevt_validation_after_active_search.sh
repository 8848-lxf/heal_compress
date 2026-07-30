#!/usr/bin/env bash
set -euo pipefail

physical_gpu="7"
active_search_pid="508877"
legacy_followon_pid="1568532"
search_root="/data/lxf/heal_data/outputs/h800_cobevt_jaq0_sixbudget_gen5_v3_20260729_111500"
python_bin="/home/lixingfeng/miniconda3/envs/univ2x-opt/bin/python"
plugin="/home/lixingfeng/UniAD_examine/heal_compress/quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so"
trt_root="/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"
followon="/home/lixingfeng/UniAD_examine/heal_compress_h800_cobevt_jaq0_repeat3_pq/scripts/run_cobevt_jaq0_gen5_repeat3_pq_followon.sh"

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "$physical_gpu" ]]; then
  echo "cobevt_validation_resume_gpu_binding_mismatch:${CUDA_VISIBLE_DEVICES:-}:$physical_gpu" >&2
  exit 2
fi

wait_for_exact_process() {
  local pid="$1"
  local required_text="$2"
  while [[ -d "/proc/$pid" ]]; do
    local cmdline
    cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    if [[ "$cmdline" != *"$required_text"* ]]; then
      echo "cobevt_validation_resume_pid_identity_changed:$pid:$cmdline" >&2
      exit 1
    fi
    sleep 30
  done
}

# Read-only waits only.  The active campaign and the already-queued legacy
# follow-on receive no signal and retain exclusive ownership of their paths.
wait_for_exact_process "$active_search_pid" "run_cobevt_formal_ga_gen5.py"
if [[ -d "/proc/$legacy_followon_pid" ]]; then
  wait_for_exact_process "$legacy_followon_pid" \
    "run_cobevt_jaq0_gen5_repeat3_pq_followon.sh"
fi

"$python_bin" scripts/run_cobevt_formal_ga_gen5.py \
  --output-root "$search_root" \
  --physical-gpu "$physical_gpu" \
  --generations 5 \
  --seed 0 \
  --activation-taylor-fitness-weight 0 \
  --plugin "$plugin" \
  --tensorrt-root "$trt_root" \
  --resume

jq -e '.results | length == 6' "$search_root/reports/formal_ga_results.json" >/dev/null
jq -e '.failures | length == 0' "$search_root/reports/formal_ga_results.json" >/dev/null

timestamp="$(date +%Y%m%d_%H%M%S)"
bash "$followon" "$physical_gpu" 0 "$timestamp"

printf '%s\n' "cobevt_fixed_validation_and_repeat3_pq_complete:$timestamp"
