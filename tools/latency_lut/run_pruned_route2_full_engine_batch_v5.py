from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.latency_lut.v5_pruned_mixed_common import (
    audit_width_changed_onnx,
    append_jsonl,
    has_all_three,
    has_int8,
    is_mixed_precision,
    load_json,
    read_jsonl,
    write_json,
)


def _run_one(args: argparse.Namespace, candidate_path: Path, result_path: Path) -> dict[str, Any]:
    conda_sh = Path(args.conda_sh)
    prefix = f"source {conda_sh} && conda activate {args.conda_env} && " if conda_sh.is_file() else ""
    cmd = [
        "bash",
        "-lc",
        (
            f"{prefix}"
            f"export LD_LIBRARY_PATH={args.trt_lib}:${{LD_LIBRARY_PATH:-}} && "
            f"python tools/latency_lut/run_full_engine_candidate_benchmark.py "
            f"--candidate {candidate_path} --output {result_path} "
            f"--val-subset-size {int(args.val_subset_size)} --deploy-mode single_engine_maxK --fixed-k 29696 "
            f"--trtexec {args.trtexec} --device {int(args.device)} --timeout {int(args.timeout)} --rebuild"
        ),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False, timeout=int(args.timeout) + 300)
    data: dict[str, Any] = {}
    if result_path.is_file():
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data.setdefault("success", False)
    data.setdefault("status", "runner_failed" if proc.returncode else "unknown")
    data["batch_returncode"] = proc.returncode
    data["batch_stdout_tail"] = proc.stdout[-12000:]
    return data


def _decomp_for(decomp_dir: Path, cid: str, pid: str) -> dict[str, Any]:
    return load_json(decomp_dir / f"{cid}_{pid}.json")


def _counts_from_verification(result: dict[str, Any]) -> tuple[int, int, int, list[Any]]:
    pv = dict(result.get("precision_verification") or {})
    return (
        int(pv.get("observed_fp32_layers") or result.get("observed_fp32_layers") or 0),
        int(pv.get("observed_fp16_layers") or result.get("observed_fp16_layers") or 0),
        int(pv.get("observed_int8_layers") or result.get("observed_int8_layers") or 0),
        list(pv.get("failures") or result.get("precision_verification_failures") or []),
    )


def _sample_from_success(
    run_cid: str,
    base_cid: str,
    profile: dict[str, Any],
    result: dict[str, Any],
    decomp: dict[str, Any],
    structure_audit: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    fp32, fp16, int8, failures = _counts_from_verification(result)
    if not structure_audit.get("is_width_changed_subnet"):
        reasons.append("is_width_changed_subnet_false")
    if not structure_audit.get("is_pruned"):
        reasons.append("is_pruned_false")
    if not is_mixed_precision(profile.get("precision_profile") or {}):
        reasons.append("is_pruned_mixed_false")
    if int(structure_audit.get("num_changed_conv_layers") or 0) <= 0:
        reasons.append("num_changed_conv_layers_zero")
    if float(structure_audit.get("param_keep_ratio") or 1.0) >= 1.0:
        reasons.append("param_keep_ratio_not_reduced")
    if failures:
        reasons.append("precision_verification_failures_non_empty")
    if has_int8(profile.get("precision_profile") or {}) and int8 <= 0:
        reasons.append("requested_int8_observed_int8_zero")
    if decomp.get("T_lut_raw") is None or float(decomp.get("T_lut_raw") or 0.0) <= 0:
        reasons.append("T_lut_raw_missing_or_non_positive")
    if decomp.get("missing_keys"):
        reasons.append("missing_keys_non_empty")
    if decomp.get("unavailable_keys"):
        reasons.append("unavailable_keys_non_empty")
    if result.get("T_real_p50") is None or float(result.get("T_real_p50") or 0.0) <= 0:
        reasons.append("T_real_p50_missing")
    if result.get("T_real_p90") is None or float(result.get("T_real_p90") or 0.0) <= 0:
        reasons.append("T_real_p90_missing")
    if reasons:
        return None, reasons
    sample = {
        "candidate_id": run_cid,
        "base_structure_candidate_id": base_cid,
        "precision_profile_id": profile["precision_profile_id"],
        "is_baseline_topology": False,
        "is_width_changed_subnet": True,
        "is_pruned": True,
        "is_pruned_mixed": True,
        "num_changed_conv_layers": int(structure_audit.get("num_changed_conv_layers") or 0),
        "param_keep_ratio": float(structure_audit.get("param_keep_ratio") or 0.0),
        "precision_profile": profile.get("precision_profile"),
        "has_fp32": bool(profile.get("has_fp32")),
        "has_fp16": bool(profile.get("has_fp16")),
        "has_int8_qdq": bool(profile.get("has_int8_qdq")),
        "route2_explicit_precision": True,
        "uses_explicit_qdq": bool(result.get("uses_qdq")),
        "observed_fp32_layers": fp32,
        "observed_fp16_layers": fp16,
        "observed_int8_layers": int8,
        "precision_verification_failures": [],
        "T_lut_raw": float(decomp["T_lut_raw"]),
        "T_real_p50": float(result["T_real_p50"]),
        "T_real_p90": float(result["T_real_p90"]),
        "mAP": result.get("mAP"),
        "AP_0.70": result.get("AP_0_70", result.get("AP_0.70")),
        "missing_keys": [],
        "unavailable_keys": [],
        "structure_audit_path": structure_audit.get("structure_audit_path"),
        "lut_decomposition_path": decomp.get("path"),
        "engine_path": result.get("engine_path"),
        "onnx_path": result.get("onnx_path"),
        "checkpoint_used": result.get("checkpoint_used"),
        "result_path": result.get("result_path"),
        "deploy_mode": result.get("deploy_mode", "single_engine_maxK"),
        "fixed_K": int(result.get("fixed_K", 29696)),
        "mixed_precision_backend": "strongly_typed_route2",
    }
    return sample, []


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidates_payload = load_json(args.candidates, {"candidates": []})
    profiles_payload = load_json(args.profiles, {"profiles": []})
    candidates_by_id = {str(c["candidate_id"]): c for c in candidates_payload.get("candidates", [])}
    for path in Path(args.export_dir).glob("*/candidate.json"):
        c = load_json(path)
        if c.get("candidate_id"):
            candidates_by_id[str(c["candidate_id"])] = c
    profiles = list(profiles_payload.get("profiles") or [])
    if args.limit is not None:
        profiles = profiles[: int(args.limit)]
    existing = {row.get("candidate_id") for row in read_jsonl(args.output)}
    result_dir = Path(args.result_dir)
    candidate_dir = Path(args.candidate_dir)
    failure_path = Path(args.failure_output)
    failures: list[dict[str, Any]] = load_json(failure_path, []) if failure_path.is_file() else []
    successes_this_run = 0
    for idx, profile in enumerate(profiles, start=1):
        base_cid = str(profile["candidate_id"])
        base_candidate = candidates_by_id.get(base_cid)
        if not base_candidate:
            failures.append({"candidate_id": base_cid, "status": "candidate_missing", "failed_stage": "preflight"})
            write_json(failure_path, failures)
            continue
        run_cid = f"{base_cid}__{profile['precision_profile_id']}"
        if not is_mixed_precision(profile.get("precision_profile") or {}):
            failures.append(
                {
                    "candidate_id": run_cid,
                    "status": "skipped_not_pruned_mixed_profile",
                    "failed_stage": "preflight",
                    "reason": "v5 calibration labels must be pruned + mixed; FP16-only profiles are kept for decomposition but not full-engine label construction.",
                }
            )
            write_json(failure_path, failures)
            continue
        if args.resume and run_cid in existing:
            continue
        decomp = _decomp_for(Path(args.decomposition_dir), base_cid, str(profile["precision_profile_id"]))
        if not decomp:
            failures.append({"candidate_id": run_cid, "status": "lut_decomposition_missing", "failed_stage": "lut_decomposition"})
            write_json(failure_path, failures)
            continue
        decomp["path"] = str(Path(args.decomposition_dir) / f"{base_cid}_{profile['precision_profile_id']}.json")
        if decomp.get("missing_keys") or decomp.get("unavailable_keys") or decomp.get("T_lut_raw") is None:
            failures.append(
                {
                    "candidate_id": run_cid,
                    "status": "lut_key_missing",
                    "failed_stage": "lut_decomposition",
                    "missing_key_count": len(decomp.get("missing_keys") or []),
                    "unavailable_key_count": len(decomp.get("unavailable_keys") or []),
                    "decomposition_path": decomp["path"],
                }
            )
            write_json(failure_path, failures)
            continue
        run_candidate = dict(base_candidate)
        run_candidate["candidate_id"] = run_cid
        run_candidate["precision_config"] = profile["precision_profile"]
        candidate_path = candidate_dir / f"{run_cid}.json"
        result_path = result_dir / f"{run_cid}.result.json"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(json.dumps(run_candidate, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[pruned-route2-v5] {idx}/{len(profiles)} {run_cid}")
        result = load_json(result_path) if args.resume and result_path.is_file() else _run_one(args, candidate_path, result_path)
        result["result_path"] = str(result_path)
        if not result.get("success"):
            failures.append(
                {
                    "candidate_id": run_cid,
                    "status": result.get("status"),
                    "failed_stage": result.get("failed_stage"),
                    "error": result.get("error"),
                    "result_path": str(result_path),
                    "batch_stdout_tail": result.get("batch_stdout_tail"),
                }
            )
            write_json(failure_path, failures)
            continue
        structure_audit = audit_width_changed_onnx(run_cid, result.get("onnx_path") or "", candidate=run_candidate)
        structure_audit["structure_audit_path"] = str(result_dir / f"{run_cid}.structure_audit.json")
        write_json(structure_audit["structure_audit_path"], structure_audit)
        sample, reasons = _sample_from_success(run_cid, base_cid, profile, result, decomp, structure_audit)
        if sample is None:
            failures.append(
                {
                    "candidate_id": run_cid,
                    "status": "dataset_admission_failed",
                    "failed_stage": "dataset_admission",
                    "reasons": reasons,
                    "result_path": str(result_path),
                    "structure_audit_path": structure_audit["structure_audit_path"],
                    "decomposition_path": decomp["path"],
                }
            )
            write_json(failure_path, failures)
            continue
        append_jsonl(args.output, sample)
        existing.add(run_cid)
        successes_this_run += 1
    rows = read_jsonl(args.output)
    stats = {
        "profiles_selected": len(profiles),
        "successful_pruned_mixed_full_engine_labels": len(rows),
        "labels_with_int8_qdq": sum(1 for r in rows if r.get("has_int8_qdq") or int(r.get("observed_int8_layers") or 0) > 0),
        "labels_with_fp32_fp16_int8_qdq": sum(1 for r in rows if has_all_three(r.get("precision_profile") or {})),
        "successes_this_run": successes_this_run,
        "failures_total": len(failures),
        "failure_by_stage": dict(Counter(str(f.get("failed_stage")) for f in failures)),
        "failure_by_status": dict(Counter(str(f.get("status")) for f in failures)),
    }
    lines = ["# Pruned Route2 Full-Engine Batch v5", "", *[f"- {k}: {v}" for k, v in stats.items()]]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="outputs/latency_lut/pruned_width_changed_candidates_v5.json")
    parser.add_argument("--export-dir", "--export_dir", dest="export_dir", default="outputs/latency_lut/pruned_width_changed_onnx_v5")
    parser.add_argument("--profiles", default="outputs/latency_lut/pruned_precision_profiles_v5.json")
    parser.add_argument("--decomposition-dir", "--decomposition_dir", dest="decomposition_dir", default="outputs/latency_lut/pruned_candidate_lut_decomposition_v5")
    parser.add_argument("--candidate-dir", "--candidate_dir", dest="candidate_dir", default="outputs/latency_lut/pruned_route2_candidates_v5")
    parser.add_argument("--result-dir", "--result_dir", dest="result_dir", default="outputs/latency_lut/pruned_route2_results_v5")
    parser.add_argument("--output", default="outputs/latency_lut/full_engine_calibration_dataset_v5.jsonl")
    parser.add_argument("--failure-output", "--failure_output", dest="failure_output", default="outputs/latency_lut/pruned_route2_full_engine_failures_v5.json")
    parser.add_argument("--report", default="outputs/latency_lut/pruned_route2_full_engine_batch_v5_report.md")
    parser.add_argument("--val-subset-size", "--val_subset_size", dest="val_subset_size", type=int, default=50)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--conda-sh", "--conda_sh", dest="conda_sh", default="${CONDA_BASE}/etc/profile.d/conda.sh")
    parser.add_argument("--conda-env", "--conda_env", dest="conda_env", default="modelopt")
    parser.add_argument("--trt-lib", "--trt_lib", dest="trt_lib", default="${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib")
    parser.add_argument("--trtexec", default="${TENSORRT_ROOT}/targets/x86_64-linux-gnu/bin/trtexec")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
