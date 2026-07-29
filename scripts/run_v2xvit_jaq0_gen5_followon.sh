#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 WAIT_PID PHYSICAL_GPU TIMESTAMP" >&2
  exit 2
fi

wait_pid="$1"
physical_gpu="$2"
timestamp="$3"

if [[ "$physical_gpu" != "3" ]]; then
  echo "v2xvit_campaign_requires_gpu3:$physical_gpu" >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "$physical_gpu" ]]; then
  echo "v2xvit_campaign_gpu_binding_mismatch:${CUDA_VISIBLE_DEVICES:-}:$physical_gpu" >&2
  exit 2
fi

root005="/data/lxf/heal_data/outputs/h800_v2xvit_r005_jaq0_gen5_20260729_130500"
while [[ -d "/proc/$wait_pid" ]]; do
  cmdline="$(tr '\0' ' ' < "/proc/$wait_pid/cmdline" 2>/dev/null || true)"
  if [[ "$cmdline" != *"$root005"* ]]; then
    echo "v2xvit_wait_pid_identity_changed:$wait_pid:$cmdline" >&2
    exit 1
  fi
  sleep 30
done

python_bin="/home/lixingfeng/miniconda3/envs/univ2x-opt/bin/python"
trt_root="/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"
plugin="/home/lixingfeng/UniAD_examine/heal_compress/quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so"
fixed500="/data/lxf/heal_data/outputs/h800_transformer_quantization_20260721_100649/evaluation/manifests/lidar_v2xvit/fixed500.json"
full1789="/home/lixingfeng/UniAD_examine/heal_compress/outputs/H800_explicit_qdq_acceptance_20260714_023005/protocol_manifests_v2/eval_1789_warmup200_reset.json"
b0_engine="/data/lxf/heal_data/outputs/h800_v2xvit_av_merge_audit_formal_ga_r010_gen10_seed0_20260726_004452/engines/greedy_exact_winners/B0/candidate.plan"
request_json="/data/lxf/heal_data/outputs/h800_v2xvit_av_merge_audit_formal_ga_r010_gen10_seed0_20260726_004452/evaluation_fixed50/B0/evaluation_request.json"

summary005="$root005/ga/budget_005/seed_0/budget_summary.json"
[[ -f "$summary005" ]]
[[ "$(jq -r '.completed_evolution_generations' "$summary005")" == "5" ]]
[[ "$(jq -r '.results | length' "$root005/reports/formal_ga_results.json")" == "1" ]]

joint005="/data/lxf/heal_data/outputs/h800_v2xvit_r005_jaq0_gen5_joint_full1789_repeat3_${timestamp}"
pq005="/data/lxf/heal_data/outputs/h800_v2xvit_r005_jaq0_gen5_pq_full1789_repeat3_${timestamp}"
"$python_bin" scripts/run_v2xvit_sixbudget_full1789_repeat3.py \
  --main-root "$root005" \
  --frozen005-root "$root005" \
  --output-root "$joint005" \
  --b0-engine "$b0_engine" \
  --request-json "$request_json" \
  --full-manifest "$full1789" \
  --physical-gpu "$physical_gpu" \
  --labels 005 \
  --repetitions 3 \
  --formal-generations 5 \
  --tensorrt-root "$trt_root"
"$python_bin" scripts/run_v2xvit_sixbudget_pq_only_full1789_repeat3.py \
  --main-root "$root005" \
  --frozen005-root "$root005" \
  --output-root "$pq005" \
  --request-json "$request_json" \
  --full-manifest "$full1789" \
  --physical-gpu "$physical_gpu" \
  --labels 005 \
  --repetitions 3 \
  --formal-generations 5 \
  --plugin "$plugin" \
  --tensorrt-root "$trt_root"

main_root="/data/lxf/heal_data/outputs/h800_v2xvit_remaining5_jaq0_gen5_${timestamp}"
frozen_manifest="$main_root/ga/frozen_domain_manifest.json"
targets="0.30,0.25,0.20,0.15,0.10"
labels="030,025,020,015,010"
"$python_bin" scripts/run_v2xvit_six_budget_proxy.py \
  --output-root "$main_root" \
  --physical-gpu "$physical_gpu" \
  --targets "$targets" \
  --activation-taylor-fitness-weight 0
"$python_bin" scripts/run_v2xvit_six_budget_formal_ga_gen5.py \
  --output-root "$main_root" \
  --physical-gpu "$physical_gpu" \
  --seed 0 \
  --generations 5 \
  --targets "$targets" \
  --frozen-domain-manifest "$frozen_manifest" \
  --prepare-frozen-domain-manifest \
  --stage2-manifest "$fixed500" \
  --plugin "$plugin" \
  --tensorrt-root "$trt_root" \
  --activation-taylor-fitness-weight 0
"$python_bin" scripts/run_v2xvit_six_budget_formal_ga_gen5.py \
  --output-root "$main_root" \
  --physical-gpu "$physical_gpu" \
  --seed 0 \
  --generations 5 \
  --targets "$targets" \
  --frozen-domain-manifest "$frozen_manifest" \
  --stage2-manifest "$fixed500" \
  --plugin "$plugin" \
  --tensorrt-root "$trt_root" \
  --activation-taylor-fitness-weight 0

[[ "$(jq -r '.results | length' "$main_root/reports/formal_ga_results.json")" == "5" ]]
joint_main="/data/lxf/heal_data/outputs/h800_v2xvit_remaining5_jaq0_gen5_joint_full1789_repeat3_${timestamp}"
pq_main="/data/lxf/heal_data/outputs/h800_v2xvit_remaining5_jaq0_gen5_pq_full1789_repeat3_${timestamp}"
"$python_bin" scripts/run_v2xvit_sixbudget_full1789_repeat3.py \
  --main-root "$main_root" \
  --frozen005-root "$root005" \
  --output-root "$joint_main" \
  --b0-engine "$b0_engine" \
  --request-json "$request_json" \
  --full-manifest "$full1789" \
  --physical-gpu "$physical_gpu" \
  --labels "$labels" \
  --repetitions 3 \
  --formal-generations 5 \
  --tensorrt-root "$trt_root"
"$python_bin" scripts/run_v2xvit_sixbudget_pq_only_full1789_repeat3.py \
  --main-root "$main_root" \
  --frozen005-root "$root005" \
  --output-root "$pq_main" \
  --request-json "$request_json" \
  --full-manifest "$full1789" \
  --physical-gpu "$physical_gpu" \
  --labels "$labels" \
  --repetitions 3 \
  --formal-generations 5 \
  --plugin "$plugin" \
  --tensorrt-root "$trt_root"

printf '%s\n' "v2xvit_jaq0_gen5_campaign_complete:$timestamp"
