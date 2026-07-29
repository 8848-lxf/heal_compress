#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 PHYSICAL_GPU WAIT_PID TIMESTAMP" >&2
  exit 2
fi

physical_gpu="$1"
wait_pid="$2"
timestamp="$3"
search_root="/data/lxf/heal_data/outputs/h800_cobevt_jaq0_sixbudget_gen5_v3_20260729_111500"

if [[ "$physical_gpu" != "7" ]]; then
  echo "cobevt_followon_requires_gpu7:$physical_gpu" >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "$physical_gpu" ]]; then
  echo "cobevt_followon_gpu_binding_mismatch:${CUDA_VISIBLE_DEVICES:-}:$physical_gpu" >&2
  exit 2
fi
if [[ "$wait_pid" != "0" ]]; then
  while [[ -d "/proc/$wait_pid" ]]; do
    cmdline="$(tr '\0' ' ' < "/proc/$wait_pid/cmdline" 2>/dev/null || true)"
    if [[ "$cmdline" != *"run_cobevt_formal_ga_gen5.py"* ]] \
      || [[ "$cmdline" != *"$search_root"* ]]; then
      echo "cobevt_followon_wait_pid_identity_changed:$wait_pid:$cmdline" >&2
      exit 1
    fi
    sleep 30
  done
fi

python_bin="/home/lixingfeng/miniconda3/envs/univ2x-opt/bin/python"
plugin="/home/lixingfeng/UniAD_examine/heal_compress/quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so"
manifest="/home/lixingfeng/UniAD_examine/heal_compress/outputs/H800_explicit_qdq_acceptance_20260714_023005/protocol_manifests_v2/eval_1789_warmup200_reset.json"
config="search/configs/heal_lidar_cobevt_h800_domain_width_joint_ga.yaml"
baseline="/home/lixingfeng/UniAD_examine/heal_compress/outputs/h800_dair_lidar_trt_fp32_all_models_smoke_20260718_2138/lidar_cobevt/strict_fp32.plan"
model_config="/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_cobevt/config.yaml"
build_root="/data/lxf/heal_data/outputs/h800_cobevt_jaq0_gen5_pq_controls_repeat3_${timestamp}"
eval_root="/data/lxf/heal_data/outputs/h800_cobevt_jaq0_gen5_pq_full1789_repeat3_${timestamp}"

[[ -f "$search_root/reports/formal_ga_results.json" ]]
[[ "$(jq -r '.results | length' "$search_root/reports/formal_ga_results.json")" == "6" ]]
[[ "$(jq -r '.failures | length' "$search_root/reports/formal_ga_results.json")" == "0" ]]

"$python_bin" scripts/run_heal_lidar_prune_quant_ablation.py \
  --action prepare --config "$config" --formal-root "$search_root" \
  --budgets 0.30,0.25,0.20,0.15,0.10,0.05 --run-dir "$build_root"
"$python_bin" scripts/run_heal_lidar_prune_quant_ablation.py \
  --action build-all --config "$config" --run-dir "$build_root" \
  --gpu-id "$physical_gpu" --plugin "$plugin"
"$python_bin" scripts/run_heal_lidar_prune_quant_ablation.py \
  --action finalize --config "$config" --run-dir "$build_root" \
  --gpu-id "$physical_gpu"
"$python_bin" scripts/run_heal_lidar_family_split_gpu_fair_evaluation.py \
  --family-id heal_lidar_cobevt --build-root "$build_root" \
  --baseline-engine "$baseline" --model-config "$model_config" \
  --eval-manifest "$manifest" --plugin "$plugin" \
  --single-gpu "$physical_gpu" --repeat-count 3 \
  --num-frames 1789 --warmup-frames 200 --latency-rounds 3 \
  --run-dir "$eval_root"

printf '%s\n' "cobevt_jaq0_gen5_repeat3_pq_complete:$physical_gpu:$timestamp"
