#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 MODEL PHYSICAL_GPU WAIT_PID TIMESTAMP" >&2
  exit 2
fi

model="$1"
physical_gpu="$2"
wait_pid="$3"
timestamp="$4"
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "$physical_gpu" ]]; then
  echo "cnn_campaign_gpu_binding_mismatch:${CUDA_VISIBLE_DEVICES:-}:$physical_gpu" >&2
  exit 2
fi

case "$model" in
  disco)
    expected_gpu=1
    family="heal_lidar_disco"
    search005="/data/lxf/heal_data/outputs/h800_disco_r005_ga_jaq0_gen5_20260729_040409"
    config="search/configs/heal_lidar_disco_h800_domain_width_joint_ga.yaml"
    model_config="/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_disco/config.yaml"
    baseline="/home/lixingfeng/UniAD_examine/heal_compress/outputs/h800_heal_lidar_fixedk29696_fp32_baselines_20260719/lidar_disco/strict_fp32.plan"
    ;;
  fcooper)
    expected_gpu=2
    family="heal_lidar_fcooper"
    search005="/data/lxf/heal_data/outputs/h800_fcooper_r005_ga_jaq0_gen5_20260729_040409"
    config="search/configs/heal_lidar_fcooper_h800_domain_width_joint_ga.yaml"
    model_config="/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_fcooper/config.yaml"
    baseline="/home/lixingfeng/UniAD_examine/heal_compress/outputs/h800_heal_lidar_fixedk29696_fp32_baselines_20260719/lidar_fcooper/strict_fp32.plan"
    ;;
  attfusion)
    expected_gpu=6
    family="heal_lidar_attfusion"
    search005="/data/lxf/heal_data/outputs/h800_attfusion_r005_jaq0_gen5_20260729_124500"
    config="search/configs/heal_lidar_attfusion_h800_domain_width_joint_ga.yaml"
    model_config="/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_attfuse/config.yaml"
    baseline="/home/lixingfeng/UniAD_examine/heal_compress/outputs/h800_dair_lidar_trt_fp32_all_models_smoke_20260718_2138/lidar_attfuse/strict_fp32.plan"
    ;;
  *)
    echo "cnn_campaign_model_unknown:$model" >&2
    exit 2
    ;;
esac
if [[ "$physical_gpu" != "$expected_gpu" ]]; then
  echo "cnn_campaign_model_gpu_mismatch:$model:$physical_gpu:$expected_gpu" >&2
  exit 2
fi

if [[ "$wait_pid" != "0" ]]; then
  while [[ -d "/proc/$wait_pid" ]]; do
    cmdline="$(tr '\0' ' ' < "/proc/$wait_pid/cmdline" 2>/dev/null || true)"
    if [[ "$cmdline" != *"$search005"* ]]; then
      echo "cnn_campaign_wait_pid_identity_changed:$wait_pid:$cmdline" >&2
      exit 1
    fi
    sleep 30
  done
fi

python_bin="/home/lixingfeng/miniconda3/envs/univ2x-opt/bin/python"
plugin="/home/lixingfeng/UniAD_examine/heal_compress/quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so"
trt_root="/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"
manifest="/home/lixingfeng/UniAD_examine/heal_compress/outputs/H800_explicit_qdq_acceptance_20260714_023005/protocol_manifests_v2/eval_1789_warmup200_reset.json"
[[ "$(jq -r '.results | length' "$search005/reports/formal_ga_results.json")" == "1" ]]

joint005="/data/lxf/heal_data/outputs/h800_${model}_r005_jaq0_gen5_joint_full1789_repeat3_${timestamp}"
if [[ -f "$joint005/reports/final_report.json" ]]; then
  [[ "$(jq -r '.passed' "$joint005/reports/final_report.json")" == "true" ]]
elif [[ "$model" == "disco" || "$model" == "fcooper" ]]; then
  legacy="/data/lxf/heal_data/outputs/h800_${model}_r005_jaq0_gen5_joint_full1789_repeat5_20260729_133000"
  "$python_bin" scripts/reaggregate_formal_repeat_prefix.py \
    --kind cnn-joint \
    --source-root "$legacy" \
    --output-root "$joint005" \
    --repeat-count 3
else
  "$python_bin" scripts/run_cnn_formal_joint_full1789_repeat5.py \
    --model "$model" \
    --search-root "$search005" \
    --output-root "$joint005" \
    --physical-gpu "$physical_gpu" \
    --budget-labels 005 \
    --repeat-count 3 \
    --eval-manifest "$manifest" \
    --plugin "$plugin"
fi

build005="/data/lxf/heal_data/outputs/h800_${model}_r005_jaq0_gen5_pq_controls_repeat3_${timestamp}"
eval005="/data/lxf/heal_data/outputs/h800_${model}_r005_jaq0_gen5_pq_full1789_repeat3_${timestamp}"
"$python_bin" scripts/run_heal_lidar_prune_quant_ablation.py \
  --action prepare --config "$config" --formal-root "$search005" \
  --budgets 0.05 --run-dir "$build005"
"$python_bin" scripts/run_heal_lidar_prune_quant_ablation.py \
  --action build-all --config "$config" --run-dir "$build005" \
  --gpu-id "$physical_gpu"
"$python_bin" scripts/run_heal_lidar_prune_quant_ablation.py \
  --action finalize --config "$config" --run-dir "$build005" \
  --gpu-id "$physical_gpu"
"$python_bin" scripts/run_heal_lidar_family_split_gpu_fair_evaluation.py \
  --family-id "$family" --build-root "$build005" \
  --baseline-engine "$baseline" --model-config "$model_config" \
  --eval-manifest "$manifest" --plugin "$plugin" \
  --single-gpu "$physical_gpu" --budgets 0.05 --repeat-count 3 \
  --num-frames 1789 --warmup-frames 200 --latency-rounds 3 \
  --run-dir "$eval005"

search="/data/lxf/heal_data/outputs/h800_${model}_remaining5_jaq0_gen5_${timestamp}"
targets="0.30,0.25,0.20,0.15,0.10"
labels="030,025,020,015,010"
"$python_bin" scripts/run_cnn_formal_ga_gen5.py \
  --model "$model" --output-root "$search" \
  --physical-gpu "$physical_gpu" --generations 5 --seed 0 \
  --activation-taylor-fitness-weight 0 --targets "$targets" \
  --plugin "$plugin" --tensorrt-root "$trt_root"
[[ "$(jq -r '.results | length' "$search/reports/formal_ga_results.json")" == "5" ]]

joint="/data/lxf/heal_data/outputs/h800_${model}_remaining5_jaq0_gen5_joint_full1789_repeat3_${timestamp}"
"$python_bin" scripts/run_cnn_formal_joint_full1789_repeat5.py \
  --model "$model" --search-root "$search" --output-root "$joint" \
  --physical-gpu "$physical_gpu" --budget-labels "$labels" \
  --repeat-count 3 --eval-manifest "$manifest" --plugin "$plugin"

build="/data/lxf/heal_data/outputs/h800_${model}_remaining5_jaq0_gen5_pq_controls_repeat3_${timestamp}"
evaluation="/data/lxf/heal_data/outputs/h800_${model}_remaining5_jaq0_gen5_pq_full1789_repeat3_${timestamp}"
"$python_bin" scripts/run_heal_lidar_prune_quant_ablation.py \
  --action prepare --config "$config" --formal-root "$search" \
  --budgets "$targets" --run-dir "$build"
"$python_bin" scripts/run_heal_lidar_prune_quant_ablation.py \
  --action build-all --config "$config" --run-dir "$build" \
  --gpu-id "$physical_gpu"
"$python_bin" scripts/run_heal_lidar_prune_quant_ablation.py \
  --action finalize --config "$config" --run-dir "$build" \
  --gpu-id "$physical_gpu"
"$python_bin" scripts/run_heal_lidar_family_split_gpu_fair_evaluation.py \
  --family-id "$family" --build-root "$build" \
  --baseline-engine "$baseline" --model-config "$model_config" \
  --eval-manifest "$manifest" --plugin "$plugin" \
  --single-gpu "$physical_gpu" --budgets "$targets" --repeat-count 3 \
  --num-frames 1789 --warmup-frames 200 --latency-rounds 3 \
  --run-dir "$evaluation"

printf '%s\n' "cnn_jaq0_gen5_campaign_complete:$model:$physical_gpu:$timestamp"
