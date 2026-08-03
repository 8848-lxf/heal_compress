#!/usr/bin/env bash
set -uo pipefail

repo_root="."
output_root="${repo_root}/outputs"
python_bin="${CONDA_BASE}/envs/univ2x-opt/bin/python"
run_tag="${HEAL_SEARCH_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
status_file="${output_root}/h800_heal_lidar_runtime_graph_formal_rerun_${run_tag}.status.jsonl"
pid_file="${output_root}/h800_heal_lidar_runtime_graph_formal_rerun_${run_tag}.pid"
overall_status=0

mkdir -p "${output_root}"
printf '%s\n' "$$" > "${pid_file}"
printf '{"event":"queue_start","timestamp":"%s","pid":%d,"git_commit":"%s","run_tag":"%s"}\n' \
  "$(date --iso-8601=seconds)" "$$" "$(git -C "${repo_root}" rev-parse HEAD)" "${run_tag}" \
  >> "${status_file}"

run_search() {
  local label="$1"
  local config="$2"
  local started
  local finished
  local returncode
  started="$(date --iso-8601=seconds)"
  printf '{"event":"start","label":"%s","timestamp":"%s","config":"%s"}\n' \
    "${label}" "${started}" "${config}" >> "${status_file}"
  "${python_bin}" -m search.cli \
    --config "${repo_root}/${config}" \
    --output-root "${output_root}"
  returncode=$?
  finished="$(date --iso-8601=seconds)"
  printf '{"event":"finish","label":"%s","timestamp":"%s","returncode":%d}\n' \
    "${label}" "${finished}" "${returncode}" >> "${status_file}"
  if [[ ${returncode} -ne 0 ]]; then
    overall_status=1
  fi
}

# Each formal search owns GPUs 5/6/7, so jobs must remain serial.
run_search \
  "disco_greedy_runtime_merge_rerun" \
  "search/configs/heal_lidar_disco_h800_domain_width_joint_greedy.yaml"
run_search \
  "fcooper_ga_per_generation_stage2_rerun" \
  "search/configs/heal_lidar_fcooper_h800_domain_width_joint_ga.yaml"
run_search \
  "disco_ga_runtime_merge_per_generation_stage2_rerun" \
  "search/configs/heal_lidar_disco_h800_domain_width_joint_ga.yaml"

printf '{"event":"queue_finish","timestamp":"%s","returncode":%d}\n' \
  "$(date --iso-8601=seconds)" "${overall_status}" >> "${status_file}"
exit "${overall_status}"
