from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.latency_lut.backfill_route2_full_engine_samples import backfill, parse_args as _unused_parse


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _has_int8(candidate: dict[str, Any]) -> bool:
    cfg = candidate.get("precision_config") or {}
    values = [cfg.get("default", "FP16")]
    values.extend((cfg.get("overrides") or {}).values())
    return any(str(v).upper() == "INT8" for v in values)


def _has_fp32_fp16_int8(candidate: dict[str, Any]) -> bool:
    cfg = candidate.get("precision_config") or {}
    values = {"FP16", str(cfg.get("default", "FP16")).upper()}
    values.update(str(v).upper() for v in (cfg.get("overrides") or {}).values())
    return {"FP32", "FP16", "INT8"}.issubset(values)


def _run_one(args: argparse.Namespace, candidate_path: Path, result_path: Path) -> dict[str, Any]:
    cmd = [
        "bash",
        "-lc",
        (
            f"source {args.conda_sh} && conda activate {args.conda_env} && "
            f"export LD_LIBRARY_PATH={args.trt_lib}:${{LD_LIBRARY_PATH:-}} && "
            f"python tools/latency_lut/run_full_engine_candidate_benchmark.py "
            f"--candidate {candidate_path} --output {result_path} "
            f"--val-subset-size {int(args.val_subset_size)} --deploy-mode single_engine_maxK --fixed-k 29696 "
            f"--trtexec {args.trtexec} --device {int(args.device)} --timeout {int(args.timeout)}"
        ),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False, timeout=int(args.timeout) + 120)
    data: dict[str, Any] = {}
    if result_path.is_file():
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data.setdefault("success", False)
    data.setdefault("status", "runner_failed" if proc.returncode else "unknown")
    data["batch_returncode"] = proc.returncode
    data["batch_stdout_tail"] = proc.stdout[-8000:]
    return data


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    candidates = list(plan.get("candidates") or [])
    if args.limit is not None:
        candidates = candidates[: int(args.limit)]
    candidate_dir = Path(args.candidate_dir)
    result_dir = Path(args.result_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    dataset = _read_jsonl(args.output)
    by_id = {row["candidate_id"]: row for row in dataset if row.get("candidate_id")}
    failures: list[dict[str, Any]] = []
    success_results: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        cid = candidate["candidate_id"]
        candidate_path = candidate_dir / f"{cid}.json"
        result_path = result_dir / f"{cid}.result.json"
        candidate_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.resume and cid in by_id:
            continue
        print(f"[route2-batch-v4] {idx}/{len(candidates)} {cid}")
        result = json.loads(result_path.read_text(encoding="utf-8")) if args.resume and result_path.is_file() else _run_one(args, candidate_path, result_path)
        if not result.get("success"):
            failures.append(
                {
                    "candidate_id": cid,
                    "status": result.get("status"),
                    "failed_stage": result.get("failed_stage"),
                    "error": result.get("error"),
                    "result_path": str(result_path),
                    "batch_returncode": result.get("batch_returncode"),
                    "batch_stdout_tail": result.get("batch_stdout_tail"),
                }
            )
            Path(args.failure_output).write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
            continue
        success_results.append(result)
        bargs = argparse.Namespace(lut=args.lut, output=args.output, report=str(Path(args.report).with_suffix(f".{cid}.backfill.md")), results=[str(result_path)])
        try:
            breport = backfill(bargs)
            dataset = _read_jsonl(args.output)
            by_id = {row["candidate_id"]: row for row in dataset if row.get("candidate_id")}
            if breport.get("num_skipped"):
                failures.append({"candidate_id": cid, "status": "lut_key_missing", "failed_stage": "lut_decomposition", "backfill": breport})
        except Exception as exc:
            failures.append({"candidate_id": cid, "status": "lut_key_missing", "failed_stage": "lut_decomposition", "error": str(exc)})
        Path(args.failure_output).write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    final_rows = _read_jsonl(args.output)
    stats = {
        "route2_batch_pipeline_implemented": True,
        "int8_batch_preflight_removed": True,
        "int8_scale_autofill_implemented": True,
        "plan_count": len(plan.get("candidates") or []),
        "selected_count": len(candidates),
        "full_engine_labels_actual": len(final_rows),
        "int8_full_engine_labels_actual": sum(1 for row in final_rows if (row.get("observed_int8_layers") or 0) > 0),
        "fp32_fp16_int8_mixed_labels_actual": sum(1 for row in final_rows if _has_fp32_fp16_int8({"precision_config": row.get("requested_precision_profile") or row.get("precision_config") or {}})),
        "pruned_mixed_labels_actual": sum(1 for row in final_rows if "prune" in row.get("candidate_id", "") and "route2" in row.get("candidate_id", "")),
        "success_results_this_run": len(success_results),
        "failures_this_run": len(failures),
        "failure_by_status": dict(Counter(row.get("status") for row in failures)),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("# Full Engine Route2 Batch Execution v4 Report\n\n" + "\n".join(f"- {k}: {v}" for k, v in stats.items()) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="outputs/latency_lut/full_engine_candidate_plan_v4.json")
    parser.add_argument("--output", default="outputs/latency_lut/full_engine_calibration_dataset_v4.jsonl")
    parser.add_argument("--candidate-dir", "--candidate_dir", dest="candidate_dir", default="outputs/latency_lut/full_engine_candidates_v4")
    parser.add_argument("--result-dir", "--result_dir", dest="result_dir", default="outputs/latency_lut/full_engine_results_v4")
    parser.add_argument("--failure-output", "--failure_output", dest="failure_output", default="outputs/latency_lut/full_engine_route2_batch_failures_v4.json")
    parser.add_argument("--report", default="outputs/latency_lut/full_engine_route2_batch_execution_v4_report.md")
    parser.add_argument("--lut", default="outputs/latency_lut/lut_records.jsonl")
    parser.add_argument("--val-subset-size", "--val_subset_size", dest="val_subset_size", type=int, default=50)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--conda-sh", "--conda_sh", dest="conda_sh", default="/home/lixingfeng/miniconda3/etc/profile.d/conda.sh")
    parser.add_argument("--conda-env", "--conda_env", dest="conda_env", default="modelopt")
    parser.add_argument("--trt-lib", "--trt_lib", dest="trt_lib", default="/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118/targets/x86_64-linux-gnu/lib")
    parser.add_argument("--trtexec", default="/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118/targets/x86_64-linux-gnu/bin/trtexec")
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
