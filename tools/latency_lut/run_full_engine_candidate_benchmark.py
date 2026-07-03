from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEPLOY_MODE = "single_engine_maxK"
FIXED_K = 29696


DEFAULT_CONFIG = Path("/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml")
DEFAULT_CHECKPOINT = Path("/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth")
DEFAULT_HEAL_REPO = Path("/home/lixingfeng/UniAD_examine/HEAL")
DEFAULT_TRT_ROOT = Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118")
DEFAULT_LAYER_MAPPING = Path("configs/latency_lut/full_engine_layer_precision_mapping.yaml")
DEFAULT_ATOMIC_INVENTORY = Path("outputs/latency_lut/atomic_deployment_unit_inventory_v2.json")
DEFAULT_SCALE_CACHE = Path("outputs/latency_lut/activation_scale_cache_calib200.json")
DEFAULT_PLUGIN_CANDIDATES = (
    Path("quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so"),
    Path("tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/plugins/pointpillar_scatter_trt_build/libpointpillar_scatter_trt.so"),
    Path("tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/plugins/pointpillar_scatter_trt_build_clean/libpointpillar_scatter_trt.so"),
)
DEFAULT_TRACE_REPORT = Path("tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/tracer_reports/lidar_pyramid/coupled_channel_groups.json")


def file_hash(path: str | Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_trtexec_latency(log_text: str) -> dict[str, float]:
    gpu_lines = [
        line.strip()
        for line in log_text.splitlines()
        if "GPU Compute Time:" in line and "Total GPU Compute Time" not in line
    ]
    if not gpu_lines:
        raise ValueError("trtexec log does not contain a GPU Compute Time summary")
    line = gpu_lines[-1]

    def extract(pattern: str) -> float | None:
        match = re.search(pattern, line, flags=re.IGNORECASE)
        return float(match.group(1)) if match else None

    mean = extract(r"mean\s*=\s*([0-9.+\-eE]+)\s*ms")
    median = extract(r"median\s*=\s*([0-9.+\-eE]+)\s*ms")
    p90 = extract(r"percentile\(90%\)\s*=\s*([0-9.+\-eE]+)\s*ms")
    p95 = extract(r"percentile\(95%\)\s*=\s*([0-9.+\-eE]+)\s*ms")
    p99 = extract(r"percentile\(99%\)\s*=\s*([0-9.+\-eE]+)\s*ms")
    std = extract(r"(?:std|stddev|standard deviation)\s*=\s*([0-9.+\-eE]+)\s*ms")
    single_value = extract(r"GPU Compute Time:\s*([0-9.+\-eE]+)\s*ms")
    if mean is None and single_value is not None:
        mean = single_value
    if median is None and mean is not None:
        median = mean
    if p90 is None and mean is not None:
        p90 = mean
    if p95 is None and p90 is not None:
        p95 = p90
    if p99 is None and p95 is not None:
        p99 = p95
    if std is None:
        std = 0.0
    required = {
        "latency_p50_ms": median,
        "latency_p90_ms": p90,
        "latency_p95_ms": p95,
        "latency_p99_ms": p99,
        "latency_mean_ms": mean,
        "latency_std_ms": std,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"trtexec GPU Compute Time summary is missing fields: {missing}; line={line}")
    return {name: float(value) for name, value in required.items() if value is not None}


def _env_for_trtexec(trtexec: str, device: int | None) -> dict[str, str]:
    env = os.environ.copy()
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(int(device))
    executable = Path(trtexec).resolve()
    candidate_lib_dirs = [
        executable.parent.parent / "lib",
        executable.parent.parent.parent / "lib",
        executable.parent.parent.parent.parent / "lib",
    ]
    lib_dirs = [str(path) for path in candidate_lib_dirs if (path / "libnvinfer.so.10").exists() or (path / "libnvinfer_plugin.so.10").exists()]
    if lib_dirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))
    return env


def _tensorrt_lib_dirs(ctx: "FullEngineContext") -> list[Path]:
    candidates = []
    if ctx.trtexec:
        executable = Path(ctx.trtexec).expanduser().resolve()
        candidates.extend(
            [
                executable.parent.parent / "lib",
                executable.parent.parent.parent / "lib",
                executable.parent.parent.parent.parent / "lib",
            ]
        )
    candidates.extend(
        [
            ctx.trt_root / "targets" / "x86_64-linux-gnu" / "lib",
            ctx.trt_root / "lib",
        ]
    )
    seen = set()
    out = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


@dataclass
class FullEngineContext:
    candidate_id: str
    output_path: Path
    work_dir: Path
    output_root: Path
    config: Path
    checkpoint: Path
    heal_repo: Path
    plugin: Path
    trt_root: Path
    trtexec: str | None
    device: int
    fixed_k: int
    precision: str
    val_subset_size: int
    timeout: int
    rebuild: bool

    @property
    def device_str(self) -> str:
        return "cuda:0"

    @property
    def onnx_path(self) -> Path:
        return (
            self.output_root
            / "artifacts"
            / "onnx"
            / f"fixedK{int(self.fixed_k)}"
            / "dynamic_agent_single_engine_maxK"
            / "lidar_pyramid_dynamic_agent_single_engine_maxK.onnx"
        )

    @property
    def engine_path(self) -> Path:
        if self.precision == "int8":
            subdir = "int8_train_calib200"
            suffix = "int8_train_calib200"
        else:
            subdir = self.precision
            suffix = self.precision
        return (
            self.output_root
            / "artifacts"
            / "engines"
            / f"fixedK{int(self.fixed_k)}"
            / "dynamic_agent_single_engine_maxK"
            / subdir
            / f"lidar_pyramid_dynamic_agent_single_engine_maxK_{suffix}.engine"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one full-engine calibration candidate.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-subset-size", "--val_subset_size", dest="val_subset_size", type=int, default=50)
    parser.add_argument("--deploy-mode", "--deploy_mode", dest="deploy_mode", default=DEPLOY_MODE)
    parser.add_argument("--fixed-k", "--fixed_k", dest="fixed_k", type=int, default=FIXED_K)
    parser.add_argument("--trtexec", default="/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118/targets/x86_64-linux-gnu/bin/trtexec")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--precision-profile", "--precision_profile", dest="precision_profile", default=None, help="Global full-engine precision for this candidate: FP32/FP16/INT8.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal-repo", "--heal_repo", dest="heal_repo", default=str(DEFAULT_HEAL_REPO))
    parser.add_argument("--plugin", default=None)
    parser.add_argument("--trt-root", "--trt_root", dest="trt_root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--output-root", "--output_root", dest="output_root", default=None)
    parser.add_argument("--trace-report", "--trace_report", dest="trace_report", default=str(DEFAULT_TRACE_REPORT))
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--onnx", default=None, help="Optional pre-exported full-engine ONNX. If omitted, candidate export must be connected by upstream tooling.")
    parser.add_argument("--engine-dir", "--engine_dir", dest="engine_dir", default="outputs/latency_lut/full_engine_engines")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=1800)
    return parser.parse_args(argv)


def _write(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _precision_flags(candidate: dict[str, Any]) -> list[str]:
    values = {str(v).upper() for v in dict(candidate.get("precision_config") or {}).values()}
    flags: list[str] = []
    if any("FP16" in value for value in values):
        flags.append("--fp16")
    if any("INT8" in value for value in values):
        flags.append("--int8")
    return flags


def _candidate_onnx(args: argparse.Namespace, candidate: dict[str, Any]) -> Path | None:
    raw = (
        args.onnx
        or candidate.get("qdq_onnx_path")
        or candidate.get("route2_onnx_path")
        or candidate.get("mixed_precision_onnx")
        or candidate.get("onnx_path")
        or candidate.get("full_engine_onnx")
    )
    if not raw:
        return None
    return Path(raw).expanduser()


def _repo_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else _ROOT / value


def _resolve_plugin(explicit: str | None) -> Path:
    if explicit:
        return _repo_path(explicit)
    for path in DEFAULT_PLUGIN_CANDIDATES:
        resolved = _repo_path(path)
        if resolved.is_file():
            return resolved
    return _repo_path(DEFAULT_PLUGIN_CANDIDATES[0])


def _precision_from_candidate(args: argparse.Namespace, candidate: dict[str, Any]) -> str:
    raw = args.precision_profile
    if raw is None:
        precision_config = dict(candidate.get("precision_config") or {})
        raw = precision_config.get("default")
        if raw is None and precision_config:
            unique = {str(value).upper() for value in precision_config.values()}
            if len(unique) == 1:
                raw = next(iter(unique))
    raw = raw or "FP16"
    value = str(raw).upper()
    if value in {"TRT_FP16", "FP16", "HALF"}:
        return "fp16"
    if value in {"TRT_FP32", "FP32", "FLOAT"}:
        return "fp32"
    if value in {"TRT_INT8_QDQ", "INT8", "TRT_INT8"}:
        return "int8"
    raise ValueError(f"unsupported full-engine precision profile: {raw}")


def _candidate_precision_set(candidate: dict[str, Any]) -> set[str]:
    data = dict(candidate.get("precision_config") or {})
    values = [data.get("default", "FP16")]
    nested = data.get("overrides")
    if isinstance(nested, dict):
        values.extend(nested.values())
    values.extend(value for key, value in data.items() if key not in {"default", "overrides"})
    normalized: set[str] = set()
    for value in values:
        label = str(value).upper()
        if label in {"FP32", "TRT_FP32"}:
            normalized.add("FP32")
        elif label in {"FP16", "TRT_FP16", "HALF"}:
            normalized.add("FP16")
        elif label in {"INT8", "TRT_INT8", "TRT_INT8_QDQ"}:
            normalized.add("INT8")
        else:
            raise ValueError(f"unsupported full-engine layer precision_profile: {value}")
    return normalized


def _importance_mode(value: Any) -> str:
    label = str(value or "l1").lower()
    mapping = {
        "l1": "l1_norm",
        "l1_norm": "l1_norm",
        "l2": "l2_norm",
        "l2_norm": "l2_norm",
        "taylor": "first_order_taylor",
        "taylor1": "first_order_taylor",
        "taylor_first_order": "first_order_taylor",
        "first_order_taylor": "first_order_taylor",
        "fisher": "second_order_fisher",
        "fisher2": "second_order_fisher",
        "fisher_second_order": "second_order_fisher",
        "second_order_fisher": "second_order_fisher",
    }
    if label not in mapping:
        raise ValueError(f"unsupported pruning importance mode: {value}")
    return mapping[label]


def _selection_mode(pruning: dict[str, Any]) -> str:
    scope = str(pruning.get("scope", "global")).lower()
    local_scope = str(pruning.get("local_scope", "")).lower()
    if scope in {"root_node_local", "root_node_local_unit_ratio"} or local_scope in {"root_node", "root_node_local", "root_node_local_unit_ratio"}:
        return "root_node_local_unit_ratio"
    if scope in {"local", "local_scope", "prune_domain", "domain"}:
        return "local_scope"
    if scope in {"global", "constrained_global"}:
        return "constrained_global"
    if scope in {"global_coupled_channel"}:
        return "global_coupled_channel"
    raise ValueError(f"unsupported pruning scope: {scope}")


def _mixed_precision_failure(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any] | None:
    precisions = _candidate_precision_set(candidate)
    if len(precisions) <= 1:
        return None
    if "INT8" in precisions:
        qdq_path = candidate.get("qdq_onnx_path") or candidate.get("route2_onnx_path") or candidate.get("mixed_precision_onnx")
        if not qdq_path:
            return _failure(
                base,
                "int8_mixed_full_engine_qdq_path_missing",
                "full-engine mixed INT8 requires an explicit W8A8 Q/DQ ONNX and TensorRT layer verification; no T_real_* was generated",
                failed_stage="precision_config",
                requested_layer_precisions=sorted(precisions),
            )
        if not Path(str(qdq_path)).expanduser().is_file():
            return _failure(
                base,
                "int8_mixed_full_engine_qdq_onnx_missing",
                f"explicit W8A8 Q/DQ ONNX not found: {qdq_path}; no T_real_* was generated",
                failed_stage="precision_config",
                requested_layer_precisions=sorted(precisions),
                qdq_onnx_path=str(qdq_path),
            )
    return None


def _precision_overrides(candidate: dict[str, Any]) -> dict[str, str]:
    cfg = dict(candidate.get("precision_config") or {})
    overrides: dict[str, str] = {}
    nested = cfg.get("overrides")
    if isinstance(nested, dict):
        overrides.update({str(k): str(v).upper() for k, v in nested.items()})
    for key, value in cfg.items():
        if key not in {"default", "overrides"}:
            overrides[str(key)] = str(value).upper()
    return overrides


def _has_int8_override(candidate: dict[str, Any]) -> bool:
    return any(value in {"INT8", "TRT_INT8_QDQ"} for value in _precision_overrides(candidate).values())


def _load_json_optional(path: str | Path) -> dict[str, Any]:
    p = _repo_path(path)
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _scale_cache_has_requested_units(candidate: dict[str, Any], scale_cache: dict[str, Any]) -> bool:
    units = scale_cache.get("units") or {}
    if not _has_int8_override(candidate):
        return True
    for unit_id, precision in _precision_overrides(candidate).items():
        if precision not in {"INT8", "TRT_INT8_QDQ"}:
            continue
        if unit_id in units:
            continue
        if any(str(key) == unit_id or str(key).startswith(unit_id + ".") or unit_id in str(key) for key in units):
            continue
        return False
    return True


def _autofill_scale_cache(ctx: FullEngineContext, candidate: dict[str, Any]) -> dict[str, Any]:
    from tools.latency_lut.collect_activation_scales import collect_scales, write_report

    output = _repo_path(DEFAULT_SCALE_CACHE)
    inventory = _repo_path(DEFAULT_ATOMIC_INVENTORY)
    args = SimpleNamespace(
        onnx=str(ctx.onnx_path),
        inventory=str(inventory),
        output=str(output),
        report=str(ctx.work_dir / "activation_scale_autofill_report.md"),
        units=[unit for unit, precision in _precision_overrides(candidate).items() if precision in {"INT8", "TRT_INT8_QDQ"}],
        trt_calibration_cache=str(_repo_path("tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/calibration/lidar_pyramid_dynamic_agent_single_engine_maxK_int8_train_calib200.cache")),
        calib200_dir=str(_repo_path("tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/calibration/train_calib_single_engine_maxK29696_200")),
        num_frames=200,
    )
    payload = collect_scales(args)
    existing = _load_json_optional(output)
    merged = dict(existing) if existing else {}
    merged["success"] = bool((merged.get("units") or {}) or payload.get("success"))
    merged["status"] = payload.get("status") if payload.get("success") else payload.get("status", "activation_scale_autofill_failed")
    merged["scale_source"] = payload.get("scale_source") or merged.get("scale_source")
    merged["num_calib_frames"] = payload.get("num_calib_frames") or merged.get("num_calib_frames")
    merged_units = dict(merged.get("units") or {})
    merged_units.update(payload.get("units") or {})
    merged["units"] = merged_units
    merged.setdefault("missing_units", [])
    merged["missing_units"] = list(merged.get("missing_units") or []) + list(payload.get("missing_units") or [])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(payload, Path(args.report))
    return {"payload": payload, "scale_cache": merged, "output": str(output), "report": str(args.report)}


def _prepare_route2_mixed_onnx(ctx: FullEngineContext, candidate: dict[str, Any]) -> dict[str, Any]:
    _prefer_local_compression_opencood(ctx)
    from tools.latency_lut.insert_full_graph_qdq import insert_qdq
    from tools.latency_lut.normalize_pruned_qdq_dtype_boundaries_v6 import normalize_pruned_qdq_dtype_boundaries
    from tools.latency_lut.rewrite_full_onnx_precision import rewrite_onnx_precision
    from tools.latency_lut.validate_onnx_dtype_closure_v6 import validate_onnx_dtype_closure

    route_dir = ctx.work_dir / "route2_mixed_onnx"
    route_dir.mkdir(parents=True, exist_ok=True)
    mapping = _repo_path(DEFAULT_LAYER_MAPPING)
    inventory_path = _repo_path(DEFAULT_ATOMIC_INVENTORY)
    inventory_data = _load_json_optional(inventory_path)
    inventory = list(inventory_data.get("units") or inventory_data or [])
    typed_onnx = route_dir / f"{ctx.candidate_id}.typed.onnx"
    typed_report = route_dir / f"{ctx.candidate_id}.typed_rewrite_report.json"
    rewrite_report = rewrite_onnx_precision(
        ctx.onnx_path,
        typed_onnx,
        candidate,
        typed_report,
        layer_mapping=mapping,
    )
    if not rewrite_report.get("success", rewrite_report.get("status") == "success"):
        return {
            "success": False,
            "status": "fp16_fp32_rewrite_failed",
            "failed_stage": "fp16_fp32_rewrite",
            "error": rewrite_report.get("error") or "FP16/FP32 typed rewrite failed",
            "rewrite_report": str(typed_report),
        }
    if not _has_int8_override(candidate):
        candidate["route2_onnx_path"] = str(typed_onnx)
        candidate["mixed_precision_onnx"] = str(typed_onnx)
        return {
            "success": True,
            "status": "route2_fp_rewrite_ready",
            "route2_onnx_path": str(typed_onnx),
            "typed_report": str(typed_report),
        }
    scale_cache_path = _repo_path(DEFAULT_SCALE_CACHE)
    scale_cache = _load_json_optional(scale_cache_path)
    autofill = None
    if not _scale_cache_has_requested_units(candidate, scale_cache):
        autofill = _autofill_scale_cache(ctx, candidate)
        scale_cache = dict(autofill.get("scale_cache") or {})
    if not _scale_cache_has_requested_units(candidate, scale_cache):
        return {
            "success": False,
            "status": "activation_scale_missing_after_autofill",
            "failed_stage": "activation_scale_autofill",
            "error": "requested INT8 units still lack activation scale after autofill",
            "autofill": autofill,
            "scale_cache": str(scale_cache_path),
        }
    qdq_onnx = route_dir / f"{ctx.candidate_id}.qdq.onnx"
    qdq_report_path = route_dir / f"{ctx.candidate_id}.qdq_rewrite_report.json"
    qdq_report = insert_qdq(
        typed_onnx,
        qdq_onnx,
        candidate,
        scale_cache,
        qdq_report_path,
        inventory=inventory,
    )
    if not qdq_report.get("success", qdq_report.get("status") == "success"):
        return {
            "success": False,
            "status": "qdq_rewrite_failed",
            "failed_stage": "qdq_rewrite",
            "error": qdq_report.get("error") or "Q/DQ insertion failed",
            "typed_report": str(typed_report),
            "qdq_report": str(qdq_report_path),
            "autofill": autofill,
        }
    dtype_closed_onnx = route_dir / f"{ctx.candidate_id}.dtype_closed.onnx"
    dtype_closure_report_path = route_dir / f"{ctx.candidate_id}.dtype_closure_report.json"
    dtype_report = normalize_pruned_qdq_dtype_boundaries(
        qdq_onnx,
        dtype_closed_onnx,
        candidate,
        dtype_closure_report_path,
    )
    if not dtype_report.get("success"):
        return {
            "success": False,
            "status": "pruned_qdq_dtype_closure_failed",
            "failed_stage": "pruned_qdq_dtype_closure_failed",
            "error": dtype_report.get("error") or "pruned QDQ dtype closure failed",
            "typed_report": str(typed_report),
            "qdq_report": str(qdq_report_path),
            "dtype_closure_report": str(dtype_closure_report_path),
            "autofill": autofill,
        }
    dtype_validation_report_path = route_dir / f"{ctx.candidate_id}.dtype_closure_validation.json"
    validation = validate_onnx_dtype_closure(
        dtype_closed_onnx,
        candidate,
        dtype_validation_report_path,
    )
    if not validation.get("valid"):
        return {
            "success": False,
            "status": "pruned_qdq_dtype_closure_failed",
            "failed_stage": "pruned_qdq_dtype_closure_failed",
            "error": "dtype closure validation failed before TensorRT build",
            "typed_report": str(typed_report),
            "qdq_report": str(qdq_report_path),
            "dtype_closure_report": str(dtype_closure_report_path),
            "dtype_closure_validation_report": str(dtype_validation_report_path),
            "validation": validation,
            "autofill": autofill,
        }
    candidate["qdq_onnx_path"] = str(qdq_onnx)
    candidate["dtype_closed_onnx_path"] = str(dtype_closed_onnx)
    candidate["route2_onnx_path"] = str(dtype_closed_onnx)
    candidate["mixed_precision_onnx"] = str(dtype_closed_onnx)
    candidate["qdq_report"] = str(qdq_report_path)
    candidate["dtype_closure_report"] = str(dtype_closure_report_path)
    candidate["dtype_closure_validation_report"] = str(dtype_validation_report_path)
    return {
        "success": True,
        "status": "route2_qdq_dtype_closed_ready",
        "route2_onnx_path": str(dtype_closed_onnx),
        "typed_report": str(typed_report),
        "qdq_report": str(qdq_report_path),
        "qdq_onnx_path": str(qdq_onnx),
        "dtype_closed_onnx_path": str(dtype_closed_onnx),
        "dtype_closure_report": str(dtype_closure_report_path),
        "dtype_closure_validation_report": str(dtype_validation_report_path),
        "dtype_closure": dtype_report,
        "dtype_closure_validation": validation,
        "autofill": autofill,
    }


def _is_fp_mixed_candidate(candidate: dict[str, Any]) -> bool:
    precisions = _candidate_precision_set(candidate)
    return len(precisions) > 1 and precisions.issubset({"FP16", "FP32"})


def _is_mixed_candidate(candidate: dict[str, Any]) -> bool:
    return len(_candidate_precision_set(candidate)) > 1


def _mixed_precision_profile_name(candidate: dict[str, Any]) -> str:
    values = sorted(_candidate_precision_set(candidate))
    return "MIXED_" + "_".join(values)


def _context(args: argparse.Namespace, candidate: dict[str, Any], candidate_id: str, output: Path) -> FullEngineContext:
    work_dir = output.parent / f"{candidate_id}.work"
    output_root = Path(args.output_root).expanduser() if args.output_root else work_dir / "quant_deploy"
    return FullEngineContext(
        candidate_id=candidate_id,
        output_path=output,
        work_dir=work_dir,
        output_root=output_root,
        config=Path(args.config).expanduser(),
        checkpoint=Path(args.checkpoint).expanduser(),
        heal_repo=Path(args.heal_repo).expanduser(),
        plugin=_resolve_plugin(args.plugin),
        trt_root=Path(args.trt_root).expanduser(),
        trtexec=str(Path(args.trtexec).expanduser()) if args.trtexec else None,
        device=int(args.device),
        fixed_k=int(args.fixed_k),
        precision=_precision_from_candidate(args, candidate),
        val_subset_size=int(args.val_subset_size),
        timeout=int(args.timeout),
        rebuild=bool(args.rebuild),
    )


def _apply_runtime_env(ctx: FullEngineContext) -> None:
    env = _env_for_trtexec(ctx.trtexec or str(ctx.trt_root / "bin" / "trtexec"), int(ctx.device))
    lib_dirs = [str(path) for path in _tensorrt_lib_dirs(ctx)]
    if lib_dirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))
    for name in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"):
        value = env.get(name)
        if value:
            os.environ[name] = value
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    for lib_name in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
        for lib_dir in _tensorrt_lib_dirs(ctx):
            lib_path = lib_dir / lib_name
            if lib_path.is_file():
                try:
                    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass
                break


def _load_quant_deploy_module(name: str) -> Any:
    tests_dir = _ROOT / "tests" / "quant_deploy"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    return __import__(name)


def _prefer_heal_opencood(ctx: FullEngineContext) -> None:
    heal = str(ctx.heal_repo.resolve())
    if heal in sys.path:
        sys.path.remove(heal)
    sys.path.insert(0, heal)
    local_opencood = str((_ROOT / "opencood").resolve())
    for name, module in list(sys.modules.items()):
        if name == "opencood" or name.startswith("opencood."):
            mod_file = getattr(module, "__file__", "") or ""
            if mod_file and str(Path(mod_file).resolve()).startswith(local_opencood):
                sys.modules.pop(name, None)


def _prefer_local_compression_opencood(ctx: FullEngineContext) -> None:
    """Use this repo's lightweight compression package for Route2 tooling.

    Full HEAL export/eval intentionally prioritizes HEAL's own ``opencood``.
    The latency LUT Route2 rewriters live under this repository's
    ``opencood.tools.compression`` namespace, so the import order must be
    flipped only while preparing typed/QDQ ONNX graphs.
    """
    heal = str(ctx.heal_repo.resolve())
    if heal in sys.path:
        sys.path.remove(heal)
    root = str(_ROOT.resolve())
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    heal_opencood = str((ctx.heal_repo / "opencood").resolve())
    for name, module in list(sys.modules.items()):
        if name == "opencood" or name.startswith("opencood."):
            mod_file = getattr(module, "__file__", "") or ""
            if mod_file and str(Path(mod_file).resolve()).startswith(heal_opencood):
                sys.modules.pop(name, None)


def _failure(base: dict[str, Any], status: str, error: str, *, failed_stage: str, **extra: Any) -> dict[str, Any]:
    return {
        **base,
        "success": False,
        "status": status,
        "failed_stage": failed_stage,
        "error": error,
        **extra,
    }


def _export_single_engine_onnx(ctx: FullEngineContext) -> dict[str, Any]:
    _prefer_heal_opencood(ctx)
    mod = _load_quant_deploy_module("export_dynamic_single_engine_maxk_onnx")
    legacy_args = SimpleNamespace(
        output_root=str(ctx.output_root),
        hypes_yaml=str(ctx.config),
        checkpoint=str(ctx.checkpoint),
        heal_repo=str(ctx.heal_repo),
        device=ctx.device_str,
        max_cav=2,
        fixed_k=int(ctx.fixed_k),
        opset=17,
        trt_root=str(ctx.trt_root),
        trtexec_path=ctx.trtexec,
        export_sample_split="train",
        max_scan_samples=128,
        overwrite=bool(ctx.rebuild),
    )
    return mod.export_onnx(legacy_args)


def _build_single_engine(ctx: FullEngineContext) -> dict[str, Any]:
    mod = _load_quant_deploy_module("build_dynamic_single_engine_maxk_trt_engine")
    legacy_args = SimpleNamespace(
        output_root=str(ctx.output_root),
        plugin_path=str(ctx.plugin),
        trt_root=str(ctx.trt_root),
        trtexec_path=ctx.trtexec,
        timeout=int(ctx.timeout),
        precisions=[ctx.precision],
        calibration_frames=[200],
        profile_calibration_frames=200,
        fixed_k=int(ctx.fixed_k),
        rebuild=bool(ctx.rebuild),
        force_recalibrate=False,
    )
    report = mod.build_all(legacy_args)
    builds = [row for row in report.get("builds", []) if row.get("precision") == ctx.precision]
    if ctx.precision == "int8":
        builds = [row for row in builds if int(row.get("calibration_frames") or 0) == 200]
    return builds[0] if builds else {"build_success": False, "error": f"no build row for precision {ctx.precision}", "build_report": report}


def _build_mixed_single_engine(ctx: FullEngineContext, candidate: dict[str, Any]) -> dict[str, Any]:
    from tools.latency_lut.build_mixed_precision_engine import build_mixed_engine

    report_dir = ctx.work_dir / "mixed_precision"
    report_dir.mkdir(parents=True, exist_ok=True)
    candidate_json = report_dir / f"{ctx.candidate_id}.mixed_candidate.json"
    build_report_path = report_dir / f"{ctx.candidate_id}.mixed_build_report.json"
    candidate_json.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
    mapping = _repo_path(DEFAULT_LAYER_MAPPING)
    mixed_onnx = (
        candidate.get("dtype_closed_onnx_path")
        or candidate.get("route2_onnx_path")
        or candidate.get("qdq_onnx_path")
        or candidate.get("mixed_precision_onnx")
        or str(ctx.onnx_path)
    )
    precisions = _candidate_precision_set(candidate)
    args = SimpleNamespace(
        onnx=str(Path(str(mixed_onnx)).expanduser()),
        engine=str(ctx.engine_path),
        candidate=str(candidate_json),
        plugin=str(ctx.plugin),
        layer_mapping=str(mapping),
        trt_root=str(ctx.trt_root),
        strongly_typed=False,
        route=str(candidate.get("mixed_precision_route") or ("strongly_typed" if "INT8" in precisions else "auto")),
        workspace_mib=4096,
        max_agents=2,
        device=int(ctx.device),
        report=str(build_report_path),
    )
    report = build_mixed_engine(args)
    build_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["build_report"] = str(build_report_path)
    report["build_success"] = bool(report.get("success") and ctx.engine_path.is_file())
    report["built_onnx"] = str(Path(str(mixed_onnx)).expanduser())
    return report


def _evaluate_single_engine_subset(ctx: FullEngineContext) -> dict[str, Any]:
    _prefer_heal_opencood(ctx)
    mod = _load_quant_deploy_module("run_dynamic_single_engine_maxk")
    utils = _load_quant_deploy_module("quant_deploy_utils")
    dirs = utils.ensure_quant_deploy_run_dirs(str(ctx.output_root))
    legacy_args = SimpleNamespace(
        output_root=str(ctx.output_root),
        plugin_path=str(ctx.plugin),
        hypes_yaml=str(ctx.config),
        checkpoint=str(ctx.checkpoint),
        heal_repo=str(ctx.heal_repo),
        device=ctx.device_str,
        trt_root=str(ctx.trt_root),
        precision=ctx.precision,
        calibration_frames=[200],
        eval_frames=[int(ctx.val_subset_size)],
        fixed_k=int(ctx.fixed_k),
        max_cav=2,
        ap_iou_backend="gpu",
    )
    return mod.evaluate_one(
        legacy_args,
        dirs,
        precision=ctx.precision,
        frames=int(ctx.val_subset_size),
        calibration_frames=200 if ctx.precision == "int8" else None,
    )


def _prepare_candidate_checkpoint(ctx: FullEngineContext, candidate: dict[str, Any]) -> dict[str, Any]:
    pruning = dict(candidate.get("pruning") or {})
    if not pruning.get("enabled", False):
        return {"success": True, "status": "no_pruning", "checkpoint": str(ctx.checkpoint), "failed_stage": None}
    if pruning.get("source") != "pruning_tool":
        return {
            "success": False,
            "status": "failed_candidate_export_not_connected",
            "failed_stage": "candidate_export",
            "error": "raw channel_config/group_mask to physical prune export is not connected for this runner",
        }
    target_keep_ratio = float(pruning.get("target_keep_ratio", 1.0 - float(pruning.get("target_prune_ratio", 0.2))))
    target_prune_ratio = max(0.0, min(0.95, 1.0 - target_keep_ratio))
    plan_dir = ctx.work_dir / "pruning_plan"
    export_dir = ctx.work_dir / "pruned_model"
    trace_report = _repo_path(str(candidate.get("trace_report") or pruning.get("trace_report") or DEFAULT_TRACE_REPORT))
    if not trace_report.is_file():
        return {
            "success": False,
            "status": "coupled_group_generation_failed",
            "failed_stage": "coupled_group_generation",
            "error": f"coupled channel group report not found: {trace_report}",
        }
    try:
        from pruning.planner.physical_prune_plan import generate_prune_plan, parse_args as parse_plan_args

        plan = generate_prune_plan(
            parse_plan_args(
                [
                    "--config",
                    str(ctx.config),
                    "--checkpoint",
                    str(ctx.checkpoint),
                    "--trace-report",
                    str(trace_report),
                    "--importance",
                    str(pruning.get("importance", "l1")),
                    "--target-prune-ratio",
                    str(target_prune_ratio),
                    "--min-keep-ratio",
                    str(pruning.get("min_keep_ratio", 0.5)),
                    "--output-dir",
                    str(plan_dir),
                ]
            )
        )
        if not (plan.get("legality") or {}).get("legal", False):
            return {
                "success": False,
                "status": "group_keep_map_failed",
                "failed_stage": "group_keep_map",
                "error": "generated prune plan is illegal",
                "prune_plan": str(plan_dir / "prune_plan.json"),
            }
        from pruning.export.export_pruned_model import export_pruned_model, parse_args as parse_export_args

        export_argv = [
            "--config",
            str(ctx.config),
            "--checkpoint",
            str(ctx.checkpoint),
            "--prune-plan",
            str(plan_dir / "prune_plan.json"),
            "--output-dir",
            str(export_dir),
            "--device",
            ctx.device_str,
            "--execute-general-pruner",
            "--target-prune-ratio",
            str(target_prune_ratio),
            "--importance-mode",
            _importance_mode(pruning.get("importance", "l1")),
            "--selection-mode",
            _selection_mode(pruning),
            "--align",
            str(int(pruning.get("align", 8))),
            "--group-conv-align",
            str(int(pruning.get("group_conv_align", pruning.get("align", 8)))),
            "--group-conv-selection-mode",
            str(pruning.get("group_conv_selection_mode", "independent_group_topk")),
        ]
        if pruning.get("num_calib_batches") is not None:
            export_argv.extend(["--num-calib-batches", str(int(pruning["num_calib_batches"]))])
        if pruning.get("run_forward_check"):
            export_argv.append("--run-forward-check")
        for prefix in pruning.get("extra_protected_prefixes", pruning.get("protected_prefixes", [])) or []:
            export_argv.extend(["--extra-protected-prefix", str(prefix)])
        export_report = export_pruned_model(
            parse_export_args(export_argv)
        )
    except Exception as exc:
        return {
            "success": False,
            "status": "physical_prune_failed",
            "failed_stage": "physical_prune",
            "error": str(exc),
        }
    pruned_checkpoint = Path(str(export_report.get("pruned_checkpoint") or export_dir / "pruned_model.pth")).expanduser()
    if not pruned_checkpoint.is_file():
        candidates = sorted(ctx.work_dir.glob("pruned_model*/pruned_model.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
        if candidates:
            pruned_checkpoint = candidates[0]
    if not export_report.get("success") or not pruned_checkpoint.is_file():
        failure_status = str(export_report.get("status") or "physical_prune_failed")
        failure_stage = "importance" if failure_status in {
            "taylor_importance_gradient_missing",
            "fisher_importance_gradient_missing",
            "importance_gradient_missing",
        } else "physical_prune"
        return {
            "success": False,
            "status": failure_status,
            "failed_stage": failure_stage,
            "error": export_report.get("error") or export_report.get("reason") or "pruned_model.pth was not produced",
            "prune_plan": str(plan_dir / "prune_plan.json"),
            "export_report": str(export_dir / "export_pruned_model_report.json"),
        }
    return {
        "success": True,
        "status": "pruned_checkpoint_ready",
        "checkpoint": str(pruned_checkpoint),
        "prune_plan": str(plan_dir / "prune_plan.json"),
        "export_report": str(export_dir / "export_pruned_model_report.json"),
        "failed_stage": None,
    }


def _std_from_eval(eval_report: dict[str, Any]) -> float | None:
    forward_ms = eval_report.get("forward_ms")
    if isinstance(forward_ms, dict):
        value = forward_ms.get("std") or forward_ms.get("std_ms")
        if value is not None:
            return float(value)
    frames = eval_report.get("frames")
    if isinstance(frames, list):
        values = [float(row["forward_ms"]) for row in frames if row.get("forward_ms") is not None]
        if values:
            import statistics

            return float(statistics.pstdev(values))
    return None


def _run_quant_deploy_pipeline(args: argparse.Namespace, candidate: dict[str, Any], base: dict[str, Any], output: Path) -> dict[str, Any]:
    ctx = _context(args, candidate, str(base["candidate_id"]), output)
    if not ctx.config.is_file():
        return _failure(base, "missing_config", f"config not found: {ctx.config}", failed_stage="preflight")
    if not ctx.checkpoint.is_file():
        return _failure(base, "missing_checkpoint", f"checkpoint not found: {ctx.checkpoint}", failed_stage="preflight")
    if not ctx.plugin.is_file():
        return _failure(base, "missing_plugin", f"PointPillarScatterTRT plugin not found: {ctx.plugin}", failed_stage="preflight")
    _apply_runtime_env(ctx)
    prepared = _prepare_candidate_checkpoint(ctx, candidate)
    if not prepared.get("success"):
        return _failure(
            base,
            str(prepared.get("status") or "physical_prune_failed"),
            str(prepared.get("error") or "candidate checkpoint preparation failed"),
            failed_stage=str(prepared.get("failed_stage") or "physical_prune"),
            **{k: v for k, v in prepared.items() if k not in {"success", "status", "error", "failed_stage"}},
        )
    ctx.checkpoint = Path(str(prepared["checkpoint"])).expanduser()
    export_report = _export_single_engine_onnx(ctx)
    if not export_report.get("success") or not ctx.onnx_path.is_file():
        return _failure(
            base,
            "onnx_export_failed",
            str(export_report.get("error") or f"ONNX was not produced: {ctx.onnx_path}"),
            failed_stage="onnx_export",
            export_report=export_report,
        )
    mixed_precision = _is_mixed_candidate(candidate)
    route2_prepare = None
    if mixed_precision:
        route2_prepare = _prepare_route2_mixed_onnx(ctx, candidate)
        if not route2_prepare.get("success"):
            return _failure(
                base,
                str(route2_prepare.get("status") or "route2_prepare_failed"),
                str(route2_prepare.get("error") or "Route2 mixed ONNX preparation failed"),
                failed_stage=str(route2_prepare.get("failed_stage") or "route2_prepare"),
                route2_prepare=route2_prepare,
            )
    build_report = _build_mixed_single_engine(ctx, candidate) if mixed_precision else _build_single_engine(ctx)
    if not build_report.get("build_success") or not ctx.engine_path.is_file():
        return _failure(
            base,
            "engine_build_failed",
            str(build_report.get("error") or f"engine was not produced: {ctx.engine_path}"),
            failed_stage="engine_build",
            build_report=build_report,
            route2_prepare=route2_prepare,
        )
    try:
        eval_report = _evaluate_single_engine_subset(ctx)
    except Exception as exc:
        return _failure(base, "runner_eval_failed", str(exc), failed_stage="runner_eval")
    if not eval_report.get("success", eval_report.get("status") == "success"):
        return _failure(
            base,
            "runner_eval_failed",
            str(eval_report.get("error") or "runner evaluation failed"),
            failed_stage="runner_eval",
            eval_report=eval_report,
        )
    p50 = eval_report.get("forward_p50_ms") or (eval_report.get("forward_ms") or {}).get("p50")
    p90 = eval_report.get("forward_p90_ms") or (eval_report.get("forward_ms") or {}).get("p90")
    p95 = eval_report.get("forward_p95_ms") or (eval_report.get("forward_ms") or {}).get("p95")
    mean = eval_report.get("forward_mean_ms") or (eval_report.get("forward_ms") or {}).get("mean")
    if p50 is None:
        return _failure(base, "runner_eval_failed", "runner report does not contain forward_p50_ms", failed_stage="runner_eval", eval_report=eval_report)
    return {
        **base,
        "success": True,
        "status": "success",
        "precision_profile": _mixed_precision_profile_name(candidate) if mixed_precision else ctx.precision.upper(),
        "mixed_precision": bool(mixed_precision),
        "precision_verification": build_report.get("precision_verification") if mixed_precision else None,
        "mixed_build_report": build_report.get("build_report") if mixed_precision else None,
        "uses_qdq": bool(mixed_precision and "INT8" in _candidate_precision_set(candidate)),
        "T_real_p50": float(p50),
        "T_real_p90": float(p90) if p90 is not None else None,
        "T_real_p95": float(p95) if p95 is not None else None,
        "T_real_mean": float(mean) if mean is not None else None,
        "T_real_std": _std_from_eval(eval_report),
        "num_val_frames": int(eval_report.get("actual_frames") or eval_report.get("evaluated_samples") or ctx.val_subset_size),
        "engine_hash": file_hash(ctx.engine_path),
        "onnx_hash": file_hash(build_report.get("built_onnx") or ctx.onnx_path),
        "mAP": eval_report.get("mAP"),
        "AP_0_70": eval_report.get("AP@0.70"),
        "engine_path": str(ctx.engine_path),
        "onnx_path": str(build_report.get("built_onnx") or ctx.onnx_path),
        "output_root": str(ctx.output_root),
        "plugin_path": str(ctx.plugin),
        "checkpoint_used": str(ctx.checkpoint),
        "candidate_preparation": prepared,
        "route2_prepare": route2_prepare,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    candidate_id = str(candidate.get("candidate_id") or Path(args.candidate).stem)
    output = Path(args.output)
    base: dict[str, Any] = {
        "candidate_id": candidate_id,
        "deploy_mode": args.deploy_mode,
        "fixed_K": int(args.fixed_k),
        "num_val_frames": int(args.val_subset_size),
        "mAP": None,
        "AP_0_70": None,
        "success": False,
    }
    if args.deploy_mode != DEPLOY_MODE or int(args.fixed_k) != FIXED_K:
        payload = {**base, "status": "failed_unsupported_deploy_mode", "error": f"only {DEPLOY_MODE} fixedK{FIXED_K} is supported"}
        _write(output, payload)
        return payload
    if not _candidate_onnx(args, candidate):
        pruning = dict(candidate.get("pruning") or {})
        if not pruning.get("enabled", False) or pruning.get("source") == "pruning_tool":
            payload = _run_quant_deploy_pipeline(args, candidate, base, output)
            _write(output, payload)
            return payload
    onnx_path = _candidate_onnx(args, candidate)
    if onnx_path is None:
        payload = {
            **base,
            "status": "failed_candidate_export_not_connected",
            "error": "candidate channel_config/precision_config to physical prune + single_engine_maxK ONNX export is not connected; no T_real_* was generated",
        }
        _write(output, payload)
        return payload
    if not onnx_path.is_file():
        payload = {**base, "status": "failed_missing_onnx", "error": f"ONNX not found: {onnx_path}", "onnx_path": str(onnx_path)}
        _write(output, payload)
        return payload
    trtexec = Path(args.trtexec).expanduser()
    if not trtexec.is_file():
        payload = {**base, "status": "failed_missing_trtexec", "error": f"trtexec not found: {trtexec}", "onnx_path": str(onnx_path)}
        _write(output, payload)
        return payload
    engine_dir = Path(args.engine_dir)
    engine_dir.mkdir(parents=True, exist_ok=True)
    engine_path = engine_dir / f"{candidate_id}.engine"
    log_path = engine_dir / f"{candidate_id}.trtexec.log"
    cmd = [
        str(trtexec),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--warmUp={int(args.warmup)}",
        f"--iterations={int(args.repeat)}",
        f"--device={int(args.device)}",
        "--useCudaGraph",
    ] + _precision_flags(candidate)
    env = _env_for_trtexec(str(trtexec), int(args.device))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, check=False, timeout=int(args.timeout), env=env)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        payload = {
            **base,
            "status": "failed_trtexec",
            "error": f"trtexec failed with returncode {proc.returncode}",
            "command": cmd,
            "onnx_path": str(onnx_path),
            "log_path": str(log_path),
            "onnx_hash": file_hash(onnx_path),
            "engine_hash": file_hash(engine_path),
        }
        _write(output, payload)
        return payload
    parsed = parse_trtexec_latency(log_text)
    payload = {
        **base,
        "success": True,
        "status": "success",
        "T_real_p50": parsed["latency_p50_ms"],
        "T_real_p90": parsed["latency_p90_ms"],
        "T_real_p95": parsed["latency_p95_ms"],
        "T_real_mean": parsed["latency_mean_ms"],
        "T_real_std": parsed["latency_std_ms"],
        "engine_hash": file_hash(engine_path),
        "onnx_hash": file_hash(onnx_path),
        "engine_path": str(engine_path),
        "onnx_path": str(onnx_path),
        "log_path": str(log_path),
        "command": cmd,
    }
    _write(output, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
