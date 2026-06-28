from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional


SUPPORTED_PRECISIONS = ("fp32", "fp16", "int8")
DEFAULT_OUTPUT_DIR = Path("tests") / "quant_deploy" / "outputs"
DEFAULT_CHECKPOINT = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/"
    "LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth"
)
DEFAULT_HYPES_YAML = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/"
    "LiDAROnly/lidar_pyramid/config.yaml"
)
DEFAULT_HEAL_REPO = Path("/home/lixingfeng/UniAD_examine/HEAL")
DEFAULT_TRT_ROOT = Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118")
INT8_NOT_IMPLEMENTED_MESSAGE = (
    "INT8 native TensorRT build is enabled with --int8. ModelOpt, explicit Q/DQ, "
    "and custom plugins are intentionally not used in this deployment path."
)


def now_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def project_path(path: str | os.PathLike[str]) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return repo_root() / p


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _validate_output_dir(output_dir: Path) -> None:
    forbidden = project_path(Path("tests") / "outputs")
    if output_dir.name == "outputs" and output_dir.parent.name == "tests":
        raise ValueError("Quant deployment outputs must not be written to tests/outputs.")
    if output_dir.exists() and _is_within(output_dir, forbidden):
        raise ValueError("Quant deployment outputs must not be written to tests/outputs.")
    if _is_within(output_dir, forbidden):
        raise ValueError("Quant deployment outputs must not be written to tests/outputs.")


def _layout_for_root(output_root: Path) -> dict[str, Path]:
    return {
        "output_root": output_root,
        "configs": output_root / "configs",
        "onnx": output_root / "artifacts" / "onnx",
        "onnx_fp32": output_root / "artifacts" / "onnx" / "fp32",
        "onnx_qdq_int8": output_root / "artifacts" / "onnx" / "qdq_int8",
        "engines": output_root / "artifacts" / "engines",
        "engine_fp32": output_root / "artifacts" / "engines" / "fp32",
        "engine_fp16": output_root / "artifacts" / "engines" / "fp16",
        "engine_int8": output_root / "artifacts" / "engines" / "int8",
        "calibration": output_root / "calibration",
        "calibration_samples": output_root / "calibration" / "calibration_samples",
        "calibration_caches": output_root / "calibration" / "caches",
        "calibration_reports": output_root / "calibration" / "reports",
        "benchmark": output_root / "benchmark",
        "benchmark_fp32": output_root / "benchmark" / "fp32",
        "benchmark_fp16": output_root / "benchmark" / "fp16",
        "benchmark_int8": output_root / "benchmark" / "int8",
        "evaluation": output_root / "evaluation",
        "evaluation_fp32": output_root / "evaluation" / "fp32",
        "evaluation_fp16": output_root / "evaluation" / "fp16",
        "evaluation_int8": output_root / "evaluation" / "int8",
        "logs": output_root / "logs",
        "logs_export": output_root / "logs" / "export",
        "logs_build": output_root / "logs" / "build",
        "logs_benchmark": output_root / "logs" / "benchmark",
        "logs_evaluation": output_root / "logs" / "evaluation",
        "summary": output_root / "summary",
        "debug": output_root / "debug",
    }


def create_quant_deploy_run_dirs(
    output_dir: str | os.PathLike[str] | None = None,
    run_name: str | None = None,
    overwrite: bool = False,
    timestamp: str | None = None,
) -> dict[str, Path]:
    output_dir_path = project_path(output_dir or DEFAULT_OUTPUT_DIR)
    _validate_output_dir(output_dir_path)
    ts = timestamp or now_timestamp()
    final_run_name = run_name or f"lidar_pyramid_deploy_{ts}"
    output_root = output_dir_path / final_run_name

    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
        else:
            base = output_root
            output_root = output_dir_path / f"{final_run_name}_{ts}"
            suffix = 1
            while output_root.exists():
                output_root = output_dir_path / f"{base.name}_{ts}_run{suffix:03d}"
                suffix += 1

    dirs = _layout_for_root(output_root)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def ensure_quant_deploy_run_dirs(output_root: str | os.PathLike[str]) -> dict[str, Path]:
    root = project_path(output_root)
    _validate_output_dir(root)
    dirs = _layout_for_root(root)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def dirs_for_summary(dirs: dict[str, Path]) -> dict[str, str]:
    keys = ("configs", "onnx", "engines", "calibration", "benchmark", "evaluation", "logs", "summary", "debug")
    return {key: str(dirs[key]) for key in keys}


def parse_precisions(precision: str | None = None, precisions: Iterable[str] | None = None) -> list[str]:
    requested = list(precisions or [])
    if not requested and precision:
        requested = [precision]
    if not requested:
        requested = ["fp16"]
    normalized: list[str] = []
    for item in requested:
        value = item.lower()
        if value not in SUPPORTED_PRECISIONS:
            raise ValueError(f"unsupported precision '{item}', expected one of {SUPPORTED_PRECISIONS}")
        if value not in normalized:
            normalized.append(value)
    return normalized


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def find_trtexec_report(
    trt_root: Optional[str] = None,
    explicit_trtexec: Optional[str] = None,
) -> dict[str, Any]:
    env_trt_root = trt_root or os.environ.get("TRT_ROOT")
    checked: list[str] = []

    if explicit_trtexec:
        explicit = Path(explicit_trtexec).expanduser()
        checked.append(str(explicit))
        if _is_executable(explicit):
            return {
                "trtexec_found": True,
                "trtexec_path": str(explicit),
                "trt_root": env_trt_root,
                "checked_paths": checked,
                "suggestion": None,
            }

    if env_trt_root:
        root = Path(env_trt_root).expanduser()
        for rel in (Path("bin") / "trtexec", Path("targets") / "x86_64-linux-gnu" / "bin" / "trtexec"):
            candidate = root / rel
            checked.append(str(candidate))
            if _is_executable(candidate):
                return {
                    "trtexec_found": True,
                    "trtexec_path": str(candidate),
                    "trt_root": str(root),
                    "checked_paths": checked,
                    "suggestion": None,
                }

    checked.append("PATH")
    path_hit = shutil.which("trtexec")
    if path_hit:
        return {
            "trtexec_found": True,
            "trtexec_path": path_hit,
            "trt_root": env_trt_root,
            "checked_paths": checked,
            "suggestion": None,
        }
    return {
        "trtexec_found": False,
        "trtexec_path": None,
        "trt_root": env_trt_root,
        "checked_paths": checked,
        "suggestion": "Run: conda activate modelopt && source tests/quant_deploy/env_modelopt_trt.sh",
    }


def find_trtexec(
    trt_root: Optional[str] = None,
    explicit_trtexec: Optional[str] = None,
) -> Optional[str]:
    return find_trtexec_report(trt_root=trt_root, explicit_trtexec=explicit_trtexec).get("trtexec_path")


def collect_env_report(
    trt_root: Optional[str] = None,
    explicit_trtexec: Optional[str] = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python_executable": sys.executable if "sys" in globals() else None,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "TRT_ROOT": trt_root or os.environ.get("TRT_ROOT"),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import sys as _sys

        report["python_executable"] = _sys.executable
    except Exception as exc:
        report["python_executable_error"] = repr(exc)
    try:
        import torch

        report["torch_available"] = True
        report["torch_version"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_device_count"] = int(torch.cuda.device_count())
        report["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    except Exception as exc:
        report.update({"torch_available": False, "torch_error": repr(exc), "cuda_available": False, "gpu_names": []})
    try:
        import tensorrt as trt

        report["tensorrt_available"] = True
        report["tensorrt_version"] = getattr(trt, "__version__", "unknown")
    except Exception as exc:
        report["tensorrt_available"] = False
        report["tensorrt_error"] = repr(exc)
        report["tensorrt_version"] = None
    try:
        import modelopt

        report["modelopt_available"] = True
        report["modelopt_version"] = getattr(modelopt, "__version__", "unknown")
    except Exception as exc:
        report["modelopt_available"] = False
        report["modelopt_error"] = repr(exc)
        report["modelopt_version"] = None
    report["trtexec"] = find_trtexec_report(trt_root=trt_root, explicit_trtexec=explicit_trtexec)
    report["trtexec_path"] = report["trtexec"].get("trtexec_path")
    return report


def _shape_profile_args(profile_shapes: dict[str, Any] | None) -> list[str]:
    if not profile_shapes:
        return []
    min_items: list[str] = []
    opt_items: list[str] = []
    max_items: list[str] = []
    for name, profile in profile_shapes.items():
        try:
            min_shape = "x".join(str(int(v)) for v in profile["min"])
            opt_shape = "x".join(str(int(v)) for v in profile["opt"])
            max_shape = "x".join(str(int(v)) for v in profile["max"])
        except Exception as exc:
            raise ValueError(f"Invalid profile shape for input '{name}': {profile}") from exc
        min_items.append(f"{name}:{min_shape}")
        opt_items.append(f"{name}:{opt_shape}")
        max_items.append(f"{name}:{max_shape}")
    return [
        f"--minShapes={','.join(min_items)}",
        f"--optShapes={','.join(opt_items)}",
        f"--maxShapes={','.join(max_items)}",
    ]


def build_trtexec_command(
    precision: str,
    onnx_path: str | os.PathLike[str],
    engine_path: str | os.PathLike[str],
    layerinfo_path: str | os.PathLike[str],
    profile_shapes: dict[str, Any] | None = None,
    trtexec_path: str = "trtexec",
    no_tf32: bool = True,
    strict_fp16: bool = False,
    calib_cache: str | os.PathLike[str] | None = None,
    qdq_onnx_path: str | os.PathLike[str] | None = None,
    int8_mode: str = "qdq",
    allow_fp16_fallback: bool = False,
    skip_inference: bool = False,
    static_plugins: list[str | os.PathLike[str]] | None = None,
) -> list[str]:
    precision = precision.lower()
    if precision not in ("fp32", "fp16", "int8"):
        raise ValueError(f"unsupported precision '{precision}'")

    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--profilingVerbosity=detailed",
        "--dumpLayerInfo",
        f"--exportLayerInfo={layerinfo_path}",
        "--verbose",
    ]
    cmd.extend(_shape_profile_args(profile_shapes))
    for plugin_path in static_plugins or []:
        cmd.append(f"--staticPlugins={plugin_path}")
    if precision == "fp32":
        if no_tf32:
            cmd.append("--noTF32")
    elif precision == "fp16":
        cmd.append("--fp16")
        if strict_fp16:
            cmd.extend(["--precisionConstraints=obey", "--layerPrecisions=*:fp16", "--layerOutputTypes=*:fp16"])
    elif precision == "int8":
        cmd.append("--int8")
        if calib_cache:
            cmd.append(f"--calib={calib_cache}")
    if skip_inference:
        cmd.append("--skipInference")
    return cmd


def run_command(command: list[str], log_path: str | os.PathLike[str], timeout: int | None = None) -> dict[str, Any]:
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        success = proc.returncode == 0
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        returncode = proc.returncode
        error = None if success else f"command failed with return code {returncode}"
    except FileNotFoundError as exc:
        success = False
        stdout = ""
        stderr = str(exc)
        returncode = None
        error = str(exc)
    except subprocess.TimeoutExpired as exc:
        success = False
        stdout = exc.stdout or ""
        stderr = exc.stderr or f"command timed out after {timeout}s"
        returncode = None
        error = f"command timed out after {timeout}s"
    elapsed = time.time() - started
    log.write_text(
        "=== COMMAND ===\n"
        + " ".join(command)
        + "\n\n=== STDOUT ===\n"
        + stdout
        + "\n\n=== STDERR ===\n"
        + stderr
        + f"\n\n=== ELAPSED_SECONDS ===\n{elapsed:.3f}\n",
        encoding="utf-8",
    )
    return {
        "success": success,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "error": error,
        "elapsed_seconds": elapsed,
        "log_path": str(log),
        "command": command,
    }


def save_json(data: Any, path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def save_csv(rows: list[dict[str, Any]], path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return p
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return p


def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def _empty_special_ops_report() -> dict[str, Any]:
    return {
        "GridSample": [],
        "AffineGrid": [],
        "Scatter": [],
        "Gather": [],
        "NonZero": [],
        "Squeeze": [],
        "Squeeze_without_axes": [],
        "SequenceEmpty": [],
        "SequenceInsert": [],
        "SequenceAt": [],
        "Sequence": [],
        "sequence_op_count": 0,
        "Inverse": [],
        "aten_ops": [],
        "org_pytorch_ops": [],
        "unsupported_ops": [],
    }


def detect_special_ops_in_onnx_model(model: Any) -> dict[str, Any]:
    report = _empty_special_ops_report()
    for node in model.graph.node:
        op_type = node.op_type
        domain = getattr(node, "domain", "") or ""
        item = {"name": node.name or "", "op_type": op_type, "domain": domain, "outputs": list(node.output)}
        if op_type == "Squeeze":
            inputs = list(getattr(node, "input", []))
            axes_attrs = [attr for attr in getattr(node, "attribute", []) if getattr(attr, "name", "") == "axes"]
            squeeze_item = dict(item)
            squeeze_item.update(
                {
                    "inputs": inputs,
                    "has_axes_input": len(inputs) >= 2,
                    "has_axes_attribute": bool(axes_attrs),
                }
            )
            report["Squeeze"].append(squeeze_item)
            if len(inputs) < 2 and not axes_attrs:
                report["Squeeze_without_axes"].append(squeeze_item)
        low = f"{domain}::{op_type}".lower()
        if op_type == "GridSample" or "grid_sample" in low or "grid_sampler" in low:
            report["GridSample"].append(item)
        if op_type == "AffineGrid" or "affine_grid" in low or "affine_grid_generator" in low:
            report["AffineGrid"].append(item)
        if op_type.startswith("Scatter") or "scatter" in low:
            report["Scatter"].append(item)
        if op_type == "Gather" or op_type.startswith("Gather"):
            report["Gather"].append(item)
        if op_type == "NonZero":
            report["NonZero"].append(item)
        if op_type.startswith("Sequence"):
            report["Sequence"].append(item)
            report["sequence_op_count"] += 1
            if op_type in {"SequenceEmpty", "SequenceInsert", "SequenceAt"}:
                report[op_type].append(item)
        if op_type in {"MatrixInverse", "Inverse", "Solve"} or "inverse" in low or "solve" in low:
            report["Inverse"].append(item)
        if domain == "aten" or op_type.startswith("aten::") or "aten::" in low:
            report["aten_ops"].append(item)
        if domain.startswith("org.pytorch"):
            report["org_pytorch_ops"].append(item)
    return report


def detect_special_ops_in_onnx(onnx_path: str | os.PathLike[str]) -> dict[str, Any]:
    report = _empty_special_ops_report()
    try:
        import onnx

        model = onnx.load(str(onnx_path))
        return detect_special_ops_in_onnx_model(model)
    except Exception as exc:
        report["unsupported_ops"].append({"source": "onnx_scan", "error": str(exc)})
        return report


def onnx_graph_info(onnx_path: str | os.PathLike[str]) -> dict[str, Any]:
    try:
        import onnx

        model = onnx.load(str(onnx_path))
        return {
            "ir_version": model.ir_version,
            "opset": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
            "inputs": [
                {"name": value.name, "type": str(value.type)}
                for value in model.graph.input
            ],
            "outputs": [
                {"name": value.name, "type": str(value.type)}
                for value in model.graph.output
            ],
            "num_nodes": len(model.graph.node),
            "op_types": sorted({node.op_type for node in model.graph.node}),
        }
    except Exception as exc:
        return {"error": str(exc)}


def parse_trtexec_failure(log_text: str) -> dict[str, Any]:
    unsupported: list[dict[str, str]] = []
    failed_nodes: list[dict[str, str]] = []
    patterns = [
        re.compile(r"Unsupported(?: ONNX)? (?:operation|op|node).*", re.IGNORECASE),
        re.compile(r"No importer registered for op: (?P<op>\S+)", re.IGNORECASE),
        re.compile(r"While parsing node number (?P<number>\d+) \[(?P<op>[^\s\]]+).*", re.IGNORECASE),
        re.compile(r"Could not find any implementation for node (?P<node>.*)", re.IGNORECASE),
        re.compile(r"Cuda failure.*", re.IGNORECASE),
        re.compile(r"CUDA.*(?:failure|error).*", re.IGNORECASE),
        re.compile(r"no CUDA-capable device is detected", re.IGNORECASE),
        re.compile(r"Assertion failed:.*", re.IGNORECASE),
        re.compile(r"ERROR:.*", re.IGNORECASE),
    ]
    for line in log_text.splitlines():
        text = line.strip()
        if not text:
            continue
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                payload = {"line": text}
                payload.update({k: v for k, v in match.groupdict().items() if v})
                if "node" in payload or "number" in payload:
                    failed_nodes.append(payload)
                else:
                    unsupported.append(payload)
                break
    return {"unsupported_ops": unsupported, "failed_nodes": failed_nodes}


def write_debug_reports(dirs: dict[str, Path], special_ops: dict[str, Any] | None = None, build_log: str | None = None) -> dict[str, Path]:
    special_path = dirs["debug"] / "special_ops_report.json"
    if special_ops is None and special_path.exists():
        special = read_json(special_path, default=_empty_special_ops_report())
    else:
        special = special_ops or _empty_special_ops_report()
    parsed = parse_trtexec_failure(build_log or "")
    paths = {
        "special_ops": save_json(special, special_path),
        "unsupported_ops": save_json(parsed["unsupported_ops"], dirs["debug"] / "unsupported_ops.json"),
        "failed_nodes": save_json(parsed["failed_nodes"], dirs["debug"] / "failed_nodes.json"),
    }
    return paths


def default_benchmark_result(precision: str, engine_path: str | os.PathLike[str] | None, num_frames: int, warmup_frames: int) -> dict[str, Any]:
    size_mb = None
    if engine_path and Path(engine_path).exists():
        size_mb = Path(engine_path).stat().st_size / (1024 * 1024)
    return {
        "precision": precision,
        "engine_path": str(engine_path) if engine_path else None,
        "num_frames": num_frames,
        "warmup_frames": warmup_frames,
        "forward_mean_ms": None,
        "forward_p50_ms": None,
        "forward_p90_ms": None,
        "forward_p95_ms": None,
        "forward_min_ms": None,
        "forward_max_ms": None,
        "fps": None,
        "engine_size_MB": size_mb,
        "latency_scope": "engine_forward_only",
        "data_loading_time_ms": None,
        "data_to_gpu_time_ms": None,
        "forward_time_ms": None,
        "postprocess_time_ms": None,
        "total_time_ms": None,
        "success": False,
        "error": None,
    }


def markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# LiDAR Pyramid TensorRT Deployment Summary",
        "",
        f"- model: {summary.get('model')}",
        f"- checkpoint: {summary.get('checkpoint')}",
        f"- hypes_yaml: {summary.get('hypes_yaml')}",
        f"- output_root: {summary.get('output_root')}",
        f"- onnx_path: {summary.get('onnx_path')}",
        "",
        "## Environment",
        "",
    ]
    env = summary.get("env_report") or {}
    trtexec = env.get("trtexec") or {}
    lines.extend([
        f"- python: {env.get('python_executable')}",
        f"- conda_prefix: {env.get('conda_prefix')}",
        f"- TRT_ROOT: {env.get('TRT_ROOT')}",
        f"- trtexec_found: {trtexec.get('trtexec_found')}",
        f"- trtexec_path: {trtexec.get('trtexec_path')}",
        f"- tensorrt_available: {env.get('tensorrt_available')}",
        f"- tensorrt_version: {env.get('tensorrt_version')}",
        f"- modelopt_available: {env.get('modelopt_available')}",
        f"- modelopt_version: {env.get('modelopt_version')}",
        f"- cuda_available: {env.get('cuda_available')}",
        f"- gpu_names: {env.get('gpu_names')}",
        "",
        "## ONNX Export",
        "",
    ])
    export = summary.get("onnx_export", {})
    lines.extend([
        f"- success: {export.get('success')}",
        f"- export_boundary: {export.get('export_boundary')}",
        f"- opset: {export.get('opset')}",
        f"- error: {export.get('error')}",
        "",
        "## Deployment Equivalence",
        "",
        f"- export_forward_mode: {summary.get('export_forward_mode')}",
        f"- is_export_specialized_wrapper: {summary.get('is_export_specialized_wrapper')}",
        f"- is_original_forward: {summary.get('is_original_forward')}",
        f"- num_pyramid_scales: {summary.get('num_pyramid_scales')}",
        f"- sequence_ops_removed: {summary.get('sequence_ops_removed')}",
        f"- wrapper_equivalence_num_frames: {summary.get('wrapper_equivalence_num_frames')}",
        f"- wrapper_equivalence_max_abs_error: {_fmt(summary.get('wrapper_equivalence_max_abs_error'))}",
        f"- wrapper_equivalence_mean_abs_error: {_fmt(summary.get('wrapper_equivalence_mean_abs_error'))}",
        f"- trt_fp32_vs_pytorch_error: {summary.get('trt_fp32_vs_pytorch_error')}",
        f"- trt_fp16_vs_pytorch_error: {summary.get('trt_fp16_vs_pytorch_error')}",
        "",
        "## Precision Results",
        "",
        "precision | implemented | build | benchmark | engine size MB | p50 ms | p90 ms | p95 ms | FPS | speedup vs FP32 | error",
        "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---",
    ])
    for precision, item in (summary.get("precisions") or {}).items():
        lines.append(
            " | ".join(
                [
                    precision,
                    "yes" if item.get("implemented") else "no",
                    "yes" if item.get("build_success") else "no",
                    "yes" if item.get("benchmark_success") else "no",
                    _fmt(item.get("engine_size_MB")),
                    _fmt(item.get("forward_p50_ms")),
                    _fmt(item.get("forward_p90_ms")),
                    _fmt(item.get("forward_p95_ms")),
                    _fmt(item.get("fps")),
                    _fmt(item.get("speedup_vs_fp32")),
                    str(item.get("error")),
                ]
            )
        )
    fixed_eval = summary.get("fixed_trt_eval_summary") or {}
    if fixed_eval:
        lines.extend(
            [
                "",
                "## Fixed TRT Evaluation",
                "",
                f"- previous_trt_ap_zero_invalidated: {summary.get('previous_trt_ap_zero_invalidated')}",
                f"- invalid_reason: {summary.get('invalid_reason')}",
                f"- trt_dtype_binding_mismatch_found: {summary.get('trt_dtype_binding_mismatch_found')}",
                f"- trt_dtype_binding_mismatched_inputs: {summary.get('trt_dtype_binding_mismatched_inputs')}",
                f"- plugin_needed: {summary.get('plugin_needed')}",
                "",
                "engine | AP@0.30 | AP@0.50 | AP@0.70 | mAP | forward p50 ms | speedup vs PyTorch | mean AP drop",
                "--- | --- | --- | --- | --- | --- | --- | ---",
            ]
        )
        for key in ("pytorch", "fp32", "fp16"):
            item = fixed_eval.get(key) or {}
            if not item:
                continue
            lines.append(
                " | ".join(
                    [
                        str(item.get("engine") or key),
                        _fmt(item.get("AP@0.30")),
                        _fmt(item.get("AP@0.50")),
                        _fmt(item.get("AP@0.70")),
                        _fmt(item.get("map")),
                        _fmt(item.get("forward_p50_ms")),
                        _fmt(item.get("speedup_vs_pytorch")),
                        _fmt(item.get("mean_ap_drop_vs_pytorch")),
                    ]
                )
            )
    five_way_ap = summary.get("five_way_ap_table") or []
    if five_way_ap:
        lines.extend(
            [
                "",
                "## Five-Way AP Comparison",
                "",
                "backend | AP@0.30 | AP@0.50 | AP@0.70 | mAP | mAP drop vs PyTorch | actual frames",
                "--- | --- | --- | --- | --- | --- | ---",
            ]
        )
        for item in five_way_ap:
            lines.append(
                " | ".join(
                    [
                        str(item.get("backend")),
                        _fmt(item.get("AP@0.30")),
                        _fmt(item.get("AP@0.50")),
                        _fmt(item.get("AP@0.70")),
                        _fmt(item.get("mAP")),
                        _fmt(item.get("mAP_drop_vs_PyTorch")),
                        _fmt(item.get("actual_frames")),
                    ]
                )
            )
        branch = summary.get("precision_drop_branch") or {}
        lines.extend(
            [
                "",
                f"- onnxruntime_fp32_close_to_pytorch: {branch.get('onnxruntime_fp32_close_to_pytorch')}",
                f"- tensorrt_fp32_close_to_pytorch: {branch.get('tensorrt_fp32_close_to_pytorch')}",
                f"- suspected_area: {branch.get('suspected_area')}",
            ]
        )
    special = summary.get("detected_special_ops") or {}
    lines.extend([
        "",
        "## Special Ops",
        "",
        f"- GridSample: {len(special.get('GridSample', []))}",
        f"- AffineGrid: {len(special.get('AffineGrid', []))}",
        f"- Scatter: {len(special.get('Scatter', []))}",
        f"- Gather: {len(special.get('Gather', []))}",
        f"- NonZero: {len(special.get('NonZero', []))}",
        f"- Inverse: {len(special.get('Inverse', []))}",
        f"- unsupported_ops: {len(special.get('unsupported_ops', []))}",
        "",
        "## Evaluation",
        "",
    ])
    if fixed_eval:
        lines.extend(
            [
                "- evaluation_status: rerun_after_dtype_fix",
                "- latency_scope: engine_forward_only for TensorRT p50/p90/p95/FPS",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "- evaluation_status: not_run",
                "- reason: current stage only benchmarks TensorRT engine forward latency",
                "",
            ]
        )
    lines.extend([
        "## Output Directories",
        "",
    ])
    for key, value in (summary.get("dirs") or {}).items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_summary_files(summary: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Path]:
    json_path = save_json(summary, dirs["summary"] / "summary_all.json")
    md_path = dirs["summary"] / "summary_all.md"
    md_path.write_text(markdown_summary(summary), encoding="utf-8")
    for precision, item in (summary.get("precisions") or {}).items():
        save_json(item, dirs["summary"] / f"summary_{precision}.json")
    return {"json": json_path, "md": md_path}


def load_profile_shapes(dirs: dict[str, Path], explicit_path: str | os.PathLike[str] | None = None) -> dict[str, Any] | None:
    candidate = Path(explicit_path) if explicit_path else dirs["configs"] / "profile_shapes.json"
    return read_json(candidate, default=None)
