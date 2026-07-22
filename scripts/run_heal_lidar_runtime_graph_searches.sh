#!/usr/bin/env bash
set -uo pipefail

repo_root="/home/lixingfeng/UniAD_examine/heal_compress"
output_root="${repo_root}/outputs"
python_bin="/home/lixingfeng/miniconda3/envs/univ2x-opt/bin/python"
status_file="${output_root}/h800_heal_lidar_runtime_graph_search_queue.status.jsonl"
overall_status=0

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

mkdir -p "${output_root}"

# Each search owns the same three-GPU pool, so the four searches are queued.
run_search "fcooper_greedy" "search/configs/heal_lidar_fcooper_h800_domain_width_joint_greedy.yaml"
run_search "disco_greedy" "search/configs/heal_lidar_disco_h800_domain_width_joint_greedy.yaml"
run_search "fcooper_ga" "search/configs/heal_lidar_fcooper_h800_domain_width_joint_ga.yaml"
run_search "disco_ga" "search/configs/heal_lidar_disco_h800_domain_width_joint_ga.yaml"

exit "${overall_status}"
