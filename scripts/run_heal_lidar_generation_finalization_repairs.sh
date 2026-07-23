#!/usr/bin/env bash
set -uo pipefail

repo_root="/home/lixingfeng/UniAD_examine/heal_compress"
output_root="${repo_root}/outputs"
python_bin="/home/lixingfeng/miniconda3/envs/univ2x-opt/bin/python"
wait_pid="${HEAL_WAIT_QUEUE_PID:-1000283}"
run_tag="${HEAL_FINALIZATION_REPAIR_TAG:-$(date +%Y%m%d_%H%M%S)}"
status_file="${output_root}/h800_heal_lidar_generation_finalization_repair_${run_tag}.status.jsonl"
pid_file="${output_root}/h800_heal_lidar_generation_finalization_repair_${run_tag}.pid"
overall_status=0

fcooper_run="${output_root}/h800_heal_lidar_fcooper_runtime_graph_joint_ga_20260723_043415"
disco_run="${output_root}/h800_heal_lidar_disco_runtime_graph_joint_ga_20260723_075044"

mkdir -p "${output_root}"
printf '%s\n' "$$" > "${pid_file}"
printf '{"event":"repair_queue_start","timestamp":"%s","pid":%d,"wait_pid":%d,"git_commit":"%s"}\n' \
  "$(date --iso-8601=seconds)" "$$" "${wait_pid}" \
  "$(git -C "${repo_root}" rev-parse HEAD)" >> "${status_file}"

while kill -0 "${wait_pid}" 2>/dev/null; do
  sleep 30
done

printf '{"event":"upstream_queue_finished","timestamp":"%s","wait_pid":%d}\n' \
  "$(date --iso-8601=seconds)" "${wait_pid}" >> "${status_file}"

while true; do
  ready_count="$(
    nvidia-smi \
      --query-gpu=index,memory.free,utilization.gpu \
      --format=csv,noheader,nounits \
    | awk -F',' '
        {
          gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3)
          if (($1 == 5 || $1 == 6 || $1 == 7) && $2 >= 60000 && $3 <= 10) {
            ready += 1
          }
        }
        END { print ready + 0 }
      '
  )"
  if [[ "${ready_count}" == "3" ]]; then
    break
  fi
  sleep 30
done

snapshot_before_resume() {
  local run_dir="$1"
  local snapshot_dir="${run_dir}/resume_snapshots/pre_generation_finalization"
  mkdir -p "${snapshot_dir}"
  for relative in \
    run_manifest.json \
    global_summary.json \
    global_summary.csv \
    search_cost_summary.json \
    resources/resource_summary.json; do
    if [[ -f "${run_dir}/${relative}" ]]; then
      mkdir -p "${snapshot_dir}/$(dirname "${relative}")"
      cp -a "${run_dir}/${relative}" "${snapshot_dir}/${relative}"
    fi
  done
}

run_resume() {
  local label="$1"
  local config="$2"
  local run_dir="$3"
  local returncode
  if [[ -f "${run_dir}/final_full_validation_results.json" ]]; then
    printf '{"event":"skip","label":"%s","timestamp":"%s","reason":"final_results_already_exist","run_dir":"%s"}\n' \
      "${label}" "$(date --iso-8601=seconds)" "${run_dir}" >> "${status_file}"
    return
  fi
  if [[ ! -d "${run_dir}" ]]; then
    printf '{"event":"finish","label":"%s","timestamp":"%s","returncode":2,"reason":"resume_run_missing","run_dir":"%s"}\n' \
      "${label}" "$(date --iso-8601=seconds)" "${run_dir}" >> "${status_file}"
    overall_status=1
    return
  fi
  snapshot_before_resume "${run_dir}"
  printf '{"event":"start","label":"%s","timestamp":"%s","config":"%s","resume":"%s"}\n' \
    "${label}" "$(date --iso-8601=seconds)" "${config}" "${run_dir}" >> "${status_file}"
  "${python_bin}" -m search.cli \
    --config "${repo_root}/${config}" \
    --output-root "${output_root}" \
    --resume "${run_dir}"
  returncode=$?
  if [[ ${returncode} -eq 0 && ! -f "${run_dir}/final_full_validation_results.json" ]]; then
    returncode=3
  fi
  printf '{"event":"finish","label":"%s","timestamp":"%s","returncode":%d,"run_dir":"%s"}\n' \
    "${label}" "$(date --iso-8601=seconds)" "${returncode}" "${run_dir}" >> "${status_file}"
  if [[ ${returncode} -ne 0 ]]; then
    overall_status=1
  fi
}

run_resume \
  "fcooper_ga_generation_winner_finalization" \
  "search/configs/heal_lidar_fcooper_h800_domain_width_joint_ga.yaml" \
  "${fcooper_run}"
run_resume \
  "disco_ga_generation_winner_finalization" \
  "search/configs/heal_lidar_disco_h800_domain_width_joint_ga.yaml" \
  "${disco_run}"

printf '{"event":"repair_queue_finish","timestamp":"%s","returncode":%d}\n' \
  "$(date --iso-8601=seconds)" "${overall_status}" >> "${status_file}"
exit "${overall_status}"
