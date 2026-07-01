from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant_deploy_utils import (
    DEFAULT_CHECKPOINT,
    DEFAULT_HEAL_REPO,
    DEFAULT_HYPES_YAML,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRT_ROOT,
    ensure_quant_deploy_run_dirs,
    find_trtexec_report,
    read_json,
    save_json,
)


FIXED_K = 29696
CALIBRATION_FRAMES = 200
SCRIPT_REQUIREMENTS = [
    "tests/quant_deploy/analyze_voxel_k_coverage.py",
    "tests/quant_deploy/dump_train_calibration_npz_for_all_strategies.py",
    "tests/quant_deploy/dump_padded_static_train_calibration_npz.py",
    "tests/quant_deploy/rebuild_fixedk_full_cover_engines.py",
    "tests/quant_deploy/build_padded_static_int8_engine.py",
    "tests/quant_deploy/evaluate_all_deployment_engines_full_val_idle_gpu.py",
    "tests/quant_deploy/summarize_engine_file_sizes.py",
    "tests/quant_deploy/summarize_final_fixedk_full_cover_traincalib.py",
    "tests/quant_deploy/select_idle_gpu.py",
    "tests/quant_deploy/agent_mode_fixed_k_plugin_ablation.py",
    "tests/quant_deploy/dynamic_single_engine_maxk_common.py",
]


@dataclass(frozen=True)
class EngineSpec:
    key: str
    scheme: str
    precision: str
    calibration: str | None
    expected_paths: tuple[str, ...]
    required_inputs: tuple[str, ...]
    expected_engine_count: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _run_command(args: list[str], *, timeout_sec: int = 20, env: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.time()
    try:
        completed = subprocess.run(
            args,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=int(timeout_sec),
            env=env,
        )
        return {
            "command": args,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timed_out": False,
            "elapsed_sec": round(time.time() - started, 3),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": args,
            "returncode": None,
            "stdout": exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else exc.stdout,
            "stderr": exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else exc.stderr,
            "timed_out": True,
            "elapsed_sec": round(time.time() - started, 3),
            "error": f"timed out after {timeout_sec}s",
        }
    except Exception as exc:
        return {
            "command": args,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "elapsed_sec": round(time.time() - started, 3),
            "error": str(exc),
        }


def _parse_csv_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in str(text or "").splitlines():
        if not line.strip() or "No running processes found" in line:
            continue
        rows.append([part.strip() for part in line.split(",")])
    return rows


def query_gpu_environment(gpu_indices: str | None, timeout_sec: int) -> dict[str, Any]:
    base = ["nvidia-smi"]
    if gpu_indices:
        base.extend(["-i", gpu_indices])
    gpu_query = base + [
        "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    proc_query = base + [
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    gpu_result = _run_command(gpu_query, timeout_sec=timeout_sec)
    proc_result = _run_command(proc_query, timeout_sec=timeout_sec)
    gpus: list[dict[str, Any]] = []
    for row in _parse_csv_rows(gpu_result.get("stdout") or ""):
        if len(row) < 6:
            continue
        gpus.append(
            {
                "index": int(float(row[0])),
                "uuid": row[1],
                "name": row[2],
                "utilization_gpu": int(float(row[3])),
                "memory_used_mb": int(float(row[4])),
                "memory_total_mb": int(float(row[5])),
                "processes": [],
            }
        )
    by_uuid = {gpu["uuid"]: gpu for gpu in gpus}
    processes: list[dict[str, Any]] = []
    for row in _parse_csv_rows(proc_result.get("stdout") or ""):
        if len(row) < 4:
            continue
        process = {
            "gpu_uuid": row[0],
            "pid": int(float(row[1])),
            "process_name": row[2],
            "used_memory_mb": int(float(row[3])),
        }
        processes.append(process)
        if process["gpu_uuid"] in by_uuid:
            by_uuid[process["gpu_uuid"]]["processes"].append(process)
    idle_candidates = [
        gpu
        for gpu in gpus
        if int(gpu["utilization_gpu"]) <= 5 and int(gpu["memory_used_mb"]) <= 2000 and not gpu["processes"]
    ]
    selected = sorted(idle_candidates, key=lambda item: (item["memory_used_mb"], item["utilization_gpu"], item["index"]))
    return {
        "nvidia_smi_available": gpu_result.get("returncode") == 0,
        "gpu_query": {k: gpu_result.get(k) for k in ("command", "returncode", "stderr", "timed_out", "elapsed_sec", "error")},
        "process_query": {k: proc_result.get(k) for k in ("command", "returncode", "stderr", "timed_out", "elapsed_sec", "error")},
        "gpu_indices_filter": gpu_indices,
        "gpus": gpus,
        "processes": processes,
        "idle_candidates": idle_candidates,
        "selected_idle_gpu": selected[0] if selected else None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixed_k_buckets(fixed_k: int) -> list[int]:
    return [0, 1, 2, 3]


def _route_requirements(output_root: Path, fixed_k: int) -> dict[str, Any]:
    path = output_root / "debug" / f"full_val_fixedK{int(fixed_k)}_trainCalib" / "full_val_route_requirements.json"
    report = read_json(path, default=None)
    if isinstance(report, dict) and report.get("required_dynamic_routes"):
        return {
            "source": str(path),
            "required_buckets": [int(item) for item in report.get("required_buckets") or _fixed_k_buckets(fixed_k)],
            "required_dynamic_routes": [(int(n), int(b)) for n, b in report.get("required_dynamic_routes")],
        }
    return {
        "source": "default_fixedK29696_routes",
        "required_buckets": _fixed_k_buckets(fixed_k),
        "required_dynamic_routes": [(1, 0), (1, 1), (2, 1), (2, 2), (2, 3)],
    }


def expected_engine_specs(output_root: Path, fixed_k: int) -> list[EngineSpec]:
    routes = _route_requirements(output_root, fixed_k)
    buckets = [int(item) for item in routes["required_buckets"]]
    dynamic_routes = [(int(n), int(b)) for n, b in routes["required_dynamic_routes"]]
    prefix = f"artifacts/engines/fixedK{int(fixed_k)}"

    def padded_paths(precision: str) -> tuple[str, ...]:
        if precision == "int8_train_calib200":
            return tuple(
                f"{prefix}/padded_agent_static/int8_train_calib200/"
                f"lidar_pyramid_padded_agent_static_fixedK{int(fixed_k)}_int8_train_calib200_bucket{bucket}.engine"
                for bucket in buckets
            )
        return tuple(
            f"{prefix}/fixed_k_scatter_plugin/{precision}/lidar_pyramid_fixed_k_scatter_plugin_bucket{bucket}_{precision}.engine"
            for bucket in buckets
        )

    def dynamic_paths(precision: str) -> tuple[str, ...]:
        return tuple(
            f"{prefix}/dynamic_agent_dim_fixed_k_scatter_plugin/N{n}/{precision}/"
            f"lidar_pyramid_dynamic_agent_dim_N{n}_fixed_k_scatter_plugin_bucket{bucket}_{precision}.engine"
            for n, bucket in dynamic_routes
        )

    def single_path(precision: str, calibration: int | None = None) -> tuple[str, ...]:
        if precision == "int8":
            suffix = f"int8_train_calib{int(calibration or CALIBRATION_FRAMES)}"
            return (
                f"{prefix}/dynamic_agent_single_engine_maxK/{suffix}/"
                f"lidar_pyramid_dynamic_agent_single_engine_maxK_{suffix}.engine",
            )
        return (
            f"{prefix}/dynamic_agent_single_engine_maxK/{precision}/"
            f"lidar_pyramid_dynamic_agent_single_engine_maxK_{precision}.engine",
        )

    return [
        EngineSpec("padded_agent_static_fp32", "padded_agent_static", "fp32", None, padded_paths("fp32"), ("valid_agent_mask", "valid_voxel_mask"), len(buckets)),
        EngineSpec("padded_agent_static_fp16", "padded_agent_static", "fp16", None, padded_paths("fp16"), ("valid_agent_mask", "valid_voxel_mask"), len(buckets)),
        EngineSpec("padded_agent_static_int8_train_calib200", "padded_agent_static", "int8", "train_calib200", padded_paths("int8_train_calib200"), ("valid_agent_mask", "valid_voxel_mask"), len(buckets)),
        EngineSpec("dynamic_bucket_fp32", "dynamic_agent_dim", "fp32", None, dynamic_paths("fp32"), ("pairwise_t_matrix", "valid_voxel_mask"), len(dynamic_routes)),
        EngineSpec("dynamic_bucket_fp16", "dynamic_agent_dim", "fp16", None, dynamic_paths("fp16"), ("pairwise_t_matrix", "valid_voxel_mask"), len(dynamic_routes)),
        EngineSpec("dynamic_bucket_int8_train_calib200", "dynamic_agent_dim", "int8", "train_calib200", dynamic_paths("int8_calib200"), ("pairwise_t_matrix", "valid_voxel_mask"), len(dynamic_routes)),
        EngineSpec("single_engine_maxK_fp32", "dynamic_agent_single_engine_maxK", "fp32", None, single_path("fp32"), ("pairwise_t_matrix", "valid_voxel_mask"), 1),
        EngineSpec("single_engine_maxK_fp16", "dynamic_agent_single_engine_maxK", "fp16", None, single_path("fp16"), ("pairwise_t_matrix", "valid_voxel_mask"), 1),
        EngineSpec("single_engine_maxK_int8_train_calib200", "dynamic_agent_single_engine_maxK", "int8", "train_calib200", single_path("int8", CALIBRATION_FRAMES), ("pairwise_t_matrix", "valid_voxel_mask"), 1),
    ]


def audit_scripts(repo: Path) -> list[dict[str, Any]]:
    return [
        {"path": rel, "exists": (repo / rel).is_file()}
        for rel in SCRIPT_REQUIREMENTS
    ]


def audit_model_files(hypes_yaml: Path, checkpoint: Path) -> dict[str, Any]:
    return {
        "config_path": str(hypes_yaml),
        "config_exists": hypes_yaml.is_file(),
        "checkpoint_path": str(checkpoint),
        "checkpoint_exists": checkpoint.is_file(),
        "checkpoint_missing": not checkpoint.is_file(),
        "cannot_rebuild_engines": not (hypes_yaml.is_file() and checkpoint.is_file()),
        "required_checkpoint_path": str(checkpoint),
        "required_config_path": str(hypes_yaml),
    }


def audit_calibration_dir(path: Path, *, expected_strategy: str, fixed_k: int, expected_count: int = CALIBRATION_FRAMES) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    manifest = read_json(manifest_path, default={}) if manifest_path.exists() else {}
    files = sorted(path.glob("*.npz")) if path.is_dir() else []
    input_shapes = manifest.get("input_shapes") if isinstance(manifest, dict) else None
    input_dtypes = manifest.get("input_dtypes") if isinstance(manifest, dict) else None
    hash_field = (
        manifest.get("sha256 for each NPZ")
        or manifest.get("sha256_for_each_npz")
        or manifest.get("files")
        if isinstance(manifest, dict)
        else None
    )
    npz_count = manifest.get("npz_file_count", manifest.get("num_samples")) if isinstance(manifest, dict) else None
    if npz_count is None:
        npz_count = len(files)
    checks = {
        "exists": path.is_dir(),
        "manifest_exists": manifest_path.is_file(),
        "calibration_split_train": manifest.get("calibration_split") == "train" if isinstance(manifest, dict) else False,
        "fixed_K_matches": int(manifest.get("fixed_K") or -1) == int(fixed_k) if isinstance(manifest, dict) else False,
        "strategy_matches": manifest.get("strategy") == expected_strategy if isinstance(manifest, dict) else False,
        "npz_count_matches": int(npz_count or 0) == int(expected_count),
        "actual_npz_files_present": len(files) == int(expected_count),
        "has_input_shapes": isinstance(input_shapes, dict) and bool(input_shapes),
        "has_input_dtypes": isinstance(input_dtypes, dict) and bool(input_dtypes),
        "has_hashes": bool(hash_field),
        "has_sample_indices": bool(manifest.get("sample_idx_list") or manifest.get("sample_idx list")),
        "calibration_eval_overlap_false": manifest.get("calibration_eval_overlap") is False or manifest.get("no_eval_overlap_checked") is True,
    }
    return {
        "path": str(path),
        "manifest_path": str(manifest_path),
        "npz_files_on_disk": len(files),
        "manifest_npz_count": npz_count,
        "checks": checks,
        "complete": all(checks.values()),
        "manifest_preview": {
            "strategy": manifest.get("strategy") if isinstance(manifest, dict) else None,
            "calibration_split": manifest.get("calibration_split") if isinstance(manifest, dict) else None,
            "fixed_K": manifest.get("fixed_K") if isinstance(manifest, dict) else None,
            "input_names": manifest.get("input_names") if isinstance(manifest, dict) else None,
            "input_shapes": input_shapes,
            "input_dtypes": input_dtypes,
        },
    }


def audit_calibration(output_root: Path, fixed_k: int) -> dict[str, Any]:
    specs = {
        "dynamic_bucket_train_calib200": (
            output_root / "artifacts" / "calibration" / f"train_calib_dynamic_bucket_fixedK{int(fixed_k)}_200",
            "dynamic_bucket",
        ),
        "single_engine_maxK_train_calib200": (
            output_root / "artifacts" / "calibration" / f"train_calib_single_engine_maxK{int(fixed_k)}_200",
            "single_engine_maxK",
        ),
        "padded_agent_static_train_calib200": (
            output_root / "artifacts" / "calibration" / f"train_calib_padded_agent_static_fixedK{int(fixed_k)}_200",
            "padded_agent_static",
        ),
    }
    rows = {
        key: audit_calibration_dir(path, expected_strategy=strategy, fixed_k=fixed_k)
        for key, (path, strategy) in specs.items()
    }
    return {
        "calibration_split": "train",
        "calibration_frames": CALIBRATION_FRAMES,
        "directories": rows,
        "missing_or_incomplete": [key for key, row in rows.items() if not row.get("complete")],
        "all_complete": all(row.get("complete") for row in rows.values()),
    }


def audit_onnx(output_root: Path, fixed_k: int) -> dict[str, Any]:
    root = output_root / "artifacts" / "onnx" / f"fixedK{int(fixed_k)}"
    expected = {
        "padded_agent_static": root / "fp32" / "lidar_pyramid_padded_agent_static_fixed_k_scatter_plugin.onnx",
        "dynamic_agent_dim_N1": root / "fp32" / "lidar_pyramid_dynamic_agent_dim_N1_fixed_k_scatter_plugin.onnx",
        "dynamic_agent_dim_N2": root / "fp32" / "lidar_pyramid_dynamic_agent_dim_N2_fixed_k_scatter_plugin.onnx",
        "single_engine_maxK": root / "dynamic_agent_single_engine_maxK" / "lidar_pyramid_dynamic_agent_single_engine_maxK.onnx",
    }
    rows = {}
    for key, path in expected.items():
        rows[key] = {
            "path": str(path),
            "exists": path.exists(),
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
    return {
        "root": str(root),
        "items": rows,
        "missing": [key for key, row in rows.items() if not (row["exists"] and (row["is_file"] or row["is_dir"]))],
        "needs_export": any(not (row["exists"] and (row["is_file"] or row["is_dir"])) for row in rows.values()),
    }


def _tensor_mode_name(trt: Any, mode: Any) -> str:
    try:
        return str(mode).split(".")[-1]
    except Exception:
        return str(mode)


def deserialize_engine(path: Path, plugin_path: Path | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "deserialization_success": False,
        "io_tensors": [],
        "input_names": [],
        "output_names": [],
        "PointPillarScatterTRT_present": False,
        "error": None,
    }
    if not path.is_file():
        report["error"] = "engine file missing"
        return report
    try:
        if plugin_path and plugin_path.is_file():
            ctypes.CDLL(str(plugin_path.resolve()), mode=ctypes.RTLD_GLOBAL)
        import tensorrt as trt

        try:
            trt.init_libnvinfer_plugins(None, "")
        except Exception:
            pass
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        with path.open("rb") as handle:
            engine = runtime.deserialize_cuda_engine(handle.read())
        if engine is None:
            report["error"] = "runtime.deserialize_cuda_engine returned None"
            return report
        report["deserialization_success"] = True
        report["num_io_tensors"] = int(engine.num_io_tensors)
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            mode = engine.get_tensor_mode(name)
            mode_name = _tensor_mode_name(trt, mode)
            item = {
                "index": int(index),
                "name": name,
                "mode": mode_name,
                "dtype": str(engine.get_tensor_dtype(name)),
                "shape": list(engine.get_tensor_shape(name)),
            }
            report["io_tensors"].append(item)
            if mode == trt.TensorIOMode.INPUT:
                report["input_names"].append(name)
            if mode == trt.TensorIOMode.OUTPUT:
                report["output_names"].append(name)
        try:
            inspector = engine.create_engine_inspector()
            layer_text = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
            report["PointPillarScatterTRT_present"] = "PointPillarScatterTRT" in layer_text
            report["engine_inspector_available"] = True
        except Exception as exc:
            report["engine_inspector_available"] = False
            report["engine_inspector_error"] = str(exc)
    except Exception as exc:
        report["error"] = str(exc)
    return report


def audit_plugin(plugin_path: Path, trt_root: Path) -> dict[str, Any]:
    env = os.environ.copy()
    lib_dirs = [trt_root / "lib", trt_root / "targets" / "x86_64-linux-gnu" / "lib"]
    env["LD_LIBRARY_PATH"] = ":".join(str(path) for path in lib_dirs if path.exists()) + ":" + env.get("LD_LIBRARY_PATH", "")
    ldd = _run_command(["ldd", str(plugin_path)], timeout_sec=20, env=env) if plugin_path.exists() else {"error": "plugin missing"}
    load_report: dict[str, Any] = {"loaded": False, "error": None}
    try:
        ctypes.CDLL(str(plugin_path.resolve()), mode=ctypes.RTLD_GLOBAL)
        load_report["loaded"] = True
    except Exception as exc:
        load_report["error"] = str(exc)
    return {
        "plugin_path": str(plugin_path),
        "exists": plugin_path.is_file(),
        "size_bytes": plugin_path.stat().st_size if plugin_path.is_file() else None,
        "sha256": _sha256(plugin_path) if plugin_path.is_file() else None,
        "ldd": {k: ldd.get(k) for k in ("returncode", "stdout", "stderr", "timed_out", "error")},
        "libnvinfer_resolved_by_ldd": "libnvinfer" in str(ldd.get("stdout") or "") and "not found" not in str(ldd.get("stdout") or ""),
        "ctypes_load": load_report,
    }


def audit_python_environment(trt_root: Path, gpu_indices: str | None, gpu_query_timeout_sec: int) -> dict[str, Any]:
    trt_import: dict[str, Any] = {"importable": False, "version": None, "error": None}
    try:
        import tensorrt as trt

        trt_import["importable"] = True
        trt_import["version"] = getattr(trt, "__version__", None)
    except Exception as exc:
        trt_import["error"] = str(exc)
    torch_cuda: dict[str, Any] = {"torch_importable": False, "cuda_available": False, "device_count": 0, "devices": [], "error": None}
    try:
        import torch

        torch_cuda["torch_importable"] = True
        torch_cuda["cuda_available"] = bool(torch.cuda.is_available())
        torch_cuda["device_count"] = int(torch.cuda.device_count()) if torch_cuda["cuda_available"] else 0
        for idx in range(int(torch_cuda["device_count"])):
            torch_cuda["devices"].append({"logical_index": idx, "name": torch.cuda.get_device_name(idx)})
    except Exception as exc:
        torch_cuda["error"] = str(exc)
    return {
        "hostname": platform.node(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_env_is_modelopt": os.environ.get("CONDA_DEFAULT_ENV") == "modelopt",
        "trt_root": str(trt_root),
        "trt_root_exists": trt_root.is_dir(),
        "libnvinfer_so": str(trt_root / "lib" / "libnvinfer.so"),
        "libnvinfer_so_exists": (trt_root / "lib" / "libnvinfer.so").exists(),
        "trtexec": find_trtexec_report(trt_root=str(trt_root)),
        "python_tensorrt": trt_import,
        "torch_cuda": torch_cuda,
        "gpu": query_gpu_environment(gpu_indices, gpu_query_timeout_sec),
    }


def audit_engines(output_root: Path, fixed_k: int, plugin_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in expected_engine_specs(output_root, fixed_k):
        path_reports = []
        for rel in spec.expected_paths:
            path = output_root / rel
            engine_report = deserialize_engine(path, plugin_path=plugin_path)
            input_names = set(engine_report.get("input_names") or [])
            engine_report["has_required_inputs"] = {
                name: name in input_names
                for name in spec.required_inputs
            }
            engine_report["has_valid_agent_mask"] = "valid_agent_mask" in input_names
            engine_report["has_valid_voxel_mask"] = "valid_voxel_mask" in input_names
            path_reports.append(engine_report)
        missing = [item["path"] for item in path_reports if not item.get("exists")]
        corrupt = [item["path"] for item in path_reports if item.get("exists") and not item.get("deserialization_success")]
        input_failures = [
            item["path"]
            for item in path_reports
            if item.get("deserialization_success")
            and not all((item.get("has_required_inputs") or {}).values())
        ]
        rows.append(
            {
                "key": spec.key,
                "scheme": spec.scheme,
                "precision": spec.precision,
                "calibration": spec.calibration,
                "expected_engine_count": spec.expected_engine_count,
                "actual_engine_count": sum(1 for item in path_reports if item.get("exists")),
                "deserialized_engine_count": sum(1 for item in path_reports if item.get("deserialization_success")),
                "missing_engine_files": missing,
                "corrupt_engine_files": corrupt,
                "input_contract_failures": input_failures,
                "deserialization_success": not missing and not corrupt,
                "needs_rebuild": bool(missing or corrupt or input_failures),
                "rebuild_reason": (
                    "missing_engine_files" if missing else "corrupt_engine_files" if corrupt else "input_contract_failures" if input_failures else None
                ),
                "required_inputs": list(spec.required_inputs),
                "engines": path_reports,
            }
        )
    return {
        "fixed_K": int(fixed_k),
        "engine_groups": rows,
        "missing_or_rebuild_required": [row["key"] for row in rows if row.get("needs_rebuild")],
        "all_engine_groups_complete": all(not row.get("needs_rebuild") for row in rows),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    repo = _repo_root()
    trt_root = Path(args.trt_root).expanduser()
    hypes_yaml = Path(args.hypes_yaml).expanduser()
    checkpoint = Path(args.checkpoint).expanduser()
    plugin_path = Path(args.plugin_path).expanduser()
    dirs = ensure_quant_deploy_run_dirs(output_root)

    scripts = audit_scripts(repo)
    model_files = audit_model_files(hypes_yaml, checkpoint)
    env = audit_python_environment(trt_root, args.gpu_indices, int(args.gpu_query_timeout_sec))
    plugin = audit_plugin(plugin_path, trt_root)
    calibration = audit_calibration(dirs["output_root"], int(args.fixed_k))
    onnx = audit_onnx(dirs["output_root"], int(args.fixed_k))
    engines = audit_engines(dirs["output_root"], int(args.fixed_k), plugin_path)
    script_missing = [item["path"] for item in scripts if not item["exists"]]

    cannot_rebuild = bool(model_files["cannot_rebuild_engines"])
    report = {
        "server_hostname": platform.node(),
        "generated_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo": str(repo),
        "output_root": str(dirs["output_root"]),
        "fixed_K": int(args.fixed_k),
        "calibration_policy": {
            "calibration_split": "train",
            "evaluation_split": "val",
            "calibration_frames": CALIBRATION_FRAMES,
            "calibration_eval_overlap": False,
        },
        "environment": env,
        "required_scripts": scripts,
        "missing_required_scripts": script_missing,
        "model_files": model_files,
        "plugin": plugin,
        "calibration_npz": calibration,
        "onnx": onnx,
        "engines": engines,
        "checkpoint_missing": bool(model_files["checkpoint_missing"]),
        "cannot_rebuild_engines": cannot_rebuild,
        "required_checkpoint_path": str(checkpoint),
        "required_config_path": str(hypes_yaml),
        "needs_calibration_redump": not calibration.get("all_complete"),
        "needs_onnx_export": bool(onnx.get("needs_export")),
        "needs_engine_rebuild": not engines.get("all_engine_groups_complete"),
        "blocked": bool(script_missing or cannot_rebuild),
        "blockers": [
            *[f"missing required script: {path}" for path in script_missing],
            *(["missing config/checkpoint required to rebuild engines"] if cannot_rebuild else []),
        ],
        "rebuild_required_engine_groups": engines.get("missing_or_rebuild_required", []),
        "HEAL_OpenCOOD_source_modified": False,
    }
    save_json(report, dirs["debug"] / "new_server_fixedK29696_artifact_audit.json")
    write_markdown(report, dirs["summary"] / "new_server_fixedK29696_artifact_audit.md")
    return report


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# New Server fixedK29696 Artifact Audit",
        "",
        f"- hostname: {report.get('server_hostname')}",
        f"- fixed_K: {report.get('fixed_K')}",
        f"- output_root: {report.get('output_root')}",
        f"- conda env: {(report.get('environment') or {}).get('conda_default_env')}",
        f"- TensorRT importable: {((report.get('environment') or {}).get('python_tensorrt') or {}).get('importable')}",
        f"- CUDA available: {(((report.get('environment') or {}).get('torch_cuda') or {}).get('cuda_available'))}",
        f"- nvidia-smi available: {((((report.get('environment') or {}).get('gpu') or {}).get('nvidia_smi_available')))}",
        f"- selected idle GPU: {((((report.get('environment') or {}).get('gpu') or {}).get('selected_idle_gpu') or {}).get('index'))}",
        f"- plugin exists: {(report.get('plugin') or {}).get('exists')}",
        f"- plugin ctypes load: {(((report.get('plugin') or {}).get('ctypes_load') or {}).get('loaded'))}",
        f"- checkpoint_missing: {report.get('checkpoint_missing')}",
        f"- needs_calibration_redump: {report.get('needs_calibration_redump')}",
        f"- needs_onnx_export: {report.get('needs_onnx_export')}",
        f"- needs_engine_rebuild: {report.get('needs_engine_rebuild')}",
        f"- blocked: {report.get('blocked')}",
        "",
        "## Calibration NPZ",
        "",
        "name | complete | npz_files | manifest_count | path",
        "--- | --- | --- | --- | ---",
    ]
    for key, row in (report.get("calibration_npz") or {}).get("directories", {}).items():
        lines.append(
            " | ".join(
                [
                    key,
                    _fmt(row.get("complete")),
                    _fmt(row.get("npz_files_on_disk")),
                    _fmt(row.get("manifest_npz_count")),
                    row.get("path", ""),
                ]
            )
        )
    lines.extend(["", "## Engine Groups", "", "key | expected | actual | deserialized | needs_rebuild | reason", "--- | --- | --- | --- | --- | ---"])
    for row in (report.get("engines") or {}).get("engine_groups", []):
        lines.append(
            " | ".join(
                [
                    row.get("key", ""),
                    _fmt(row.get("expected_engine_count")),
                    _fmt(row.get("actual_engine_count")),
                    _fmt(row.get("deserialized_engine_count")),
                    _fmt(row.get("needs_rebuild")),
                    _fmt(row.get("rebuild_reason")),
                ]
            )
        )
    lines.extend(["", "## Blockers", ""])
    blockers = report.get("blockers") or []
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker}")
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit fixedK29696 deployment artifacts after cloning on a new server.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_DIR / "lidar_pyramid_agent_export_strategy_compare"))
    parser.add_argument("--fixed_k", type=int, default=FIXED_K)
    parser.add_argument("--hypes_yaml", default=str(DEFAULT_HYPES_YAML))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal_repo", default=str(DEFAULT_HEAL_REPO))
    parser.add_argument("--trt_root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--plugin_path", default="tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/plugins/pointpillar_scatter_trt_build/libpointpillar_scatter_trt.so")
    parser.add_argument("--gpu_indices", default=None, help="Optional comma-separated physical GPU ids to query, for example 0,2,3,4.")
    parser.add_argument("--gpu_query_timeout_sec", type=int, default=30)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    report = build_report(parse_args(argv))
    print(
        json.dumps(
            {
                "blocked": report.get("blocked"),
                "needs_calibration_redump": report.get("needs_calibration_redump"),
                "needs_onnx_export": report.get("needs_onnx_export"),
                "needs_engine_rebuild": report.get("needs_engine_rebuild"),
                "rebuild_required_engine_groups": report.get("rebuild_required_engine_groups"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 2 if report.get("blocked") else 0


if __name__ == "__main__":
    raise SystemExit(main())
