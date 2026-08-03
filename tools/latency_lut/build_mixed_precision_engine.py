from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


FALLBACK_STATUS = "current_tensorrt_10x_mixed_precision_not_reliable_for_final_pipeline"


def probe_tensorrt() -> dict[str, Any]:
    try:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        has_strong = hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED")
        if has_strong:
            flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        network = builder.create_network(flags)
        return {
            "import_success": True,
            "tensorrt_version": trt.__version__,
            "has_strongly_typed": has_strong,
            "strongly_typed_network_created": network is not None,
            "has_obey_precision_constraints": hasattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS"),
            "has_prefer_precision_constraints": hasattr(trt.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS"),
        }
    except Exception as exc:
        return {"import_success": False, "error": repr(exc)}


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _precision_config(candidate: dict[str, Any]) -> tuple[str, dict[str, str]]:
    cfg = dict(candidate.get("precision_config") or {})
    default = str(cfg.get("default", "FP16")).upper()
    overrides = {}
    nested = cfg.get("overrides")
    if isinstance(nested, dict):
        overrides.update({str(k): str(v).upper() for k, v in nested.items()})
    for key, value in cfg.items():
        if key not in {"default", "overrides"}:
            overrides[str(key)] = str(value).upper()
    return default, overrides


def _load_mapping(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _patterns(unit: str, mapping: dict[str, Any], key: str) -> list[str]:
    entry = mapping.get(unit) or {}
    values = entry.get(key) if isinstance(entry, dict) else None
    return [str(v).lower() for v in values] if values else [unit.lower()]


def _matches(name: str, patterns: list[str]) -> bool:
    low = name.lower()
    normalized_name = re.sub(r"[^a-z0-9]+", "", low)
    for pattern in patterns:
        p = pattern.lower()
        normalized_pattern = re.sub(r"[^a-z0-9]+", "", p)
        if (
            p in low
            or p.replace(".", "_") in low
            or p.replace(".", "/") in low
            or (normalized_pattern and normalized_pattern in normalized_name)
        ):
            return True
    return False


def _set_layer_precision(network: Any, mapping: dict[str, Any], candidate: dict[str, Any], trt: Any) -> dict[str, Any]:
    _default, overrides = _precision_config(candidate)
    requested: dict[str, str] = {}
    matched: dict[str, list[str]] = {unit: [] for unit in overrides}
    for idx in range(network.num_layers):
        layer = network.get_layer(idx)
        name = str(layer.name)
        precision = None
        for unit, override_precision in overrides.items():
            if _matches(name, _patterns(unit, mapping, "trt_layer_patterns")):
                precision = override_precision
                matched[unit].append(name)
                break
        if precision is None:
            continue
        if precision in {"FP32", "TRT_FP32"}:
            layer.precision = trt.float32
            for out_idx in range(layer.num_outputs):
                layer.set_output_type(out_idx, trt.float32)
            requested[name] = "FP32"
        elif precision in {"FP16", "TRT_FP16"}:
            layer.precision = trt.float16
            for out_idx in range(layer.num_outputs):
                layer.set_output_type(out_idx, trt.float16)
            requested[name] = "FP16"
        elif precision in {"INT8", "TRT_INT8_QDQ"}:
            requested[name] = "INT8"
        else:
            raise ValueError(f"unsupported precision override for layer {name}: {precision}")
    return {"requested_by_layer": requested, "matched_overrides": matched}


def _parse_inspector(raw: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(raw)
    except Exception:
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("Layers", "layers"):
            if isinstance(data.get(key), list):
                rows = []
                for x in data[key]:
                    if isinstance(x, dict):
                        rows.append(x)
                    elif isinstance(x, str):
                        rows.append({"Name": x})
                return rows
    return []


def _layer_name(row: dict[str, Any]) -> str:
    for key in ("Name", "name", "LayerName", "LayerName"):
        if row.get(key):
            return str(row[key])
    return json.dumps(row, sort_keys=True)


def _layer_precision(row: dict[str, Any]) -> str:
    text = json.dumps(row, sort_keys=True).upper()
    if "INT8" in text:
        return "INT8"
    if "FP32" in text or "FLOAT" in text:
        return "FP32"
    if "FP16" in text or "HALF" in text:
        return "FP16"
    return "UNKNOWN"


def _verify_precision(layers: list[dict[str, Any]], mapping: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    default, overrides = _precision_config(candidate)
    observed_by_unit: dict[str, list[dict[str, str]]] = {}
    failures: list[dict[str, Any]] = []
    for unit, precision in overrides.items():
        expected = "INT8" if precision in {"INT8", "TRT_INT8_QDQ"} else ("FP32" if precision in {"FP32", "TRT_FP32"} else "FP16")
        pats = _patterns(unit, mapping, "trt_layer_patterns")
        rows = []
        for layer in layers:
            name = _layer_name(layer)
            if _matches(name, pats):
                rows.append({"name": name, "precision": _layer_precision(layer)})
        observed_by_unit[unit] = rows
        if not rows:
            failures.append({"unit": unit, "issue": "trt_layer_mapping_failed", "expected": expected})
        else:
            bad = [row for row in rows if row["precision"] not in {expected, "UNKNOWN"}]
            unknown = [row for row in rows if row["precision"] == "UNKNOWN"]
            if bad:
                failures.append({"unit": unit, "issue": "precision_constraint_not_obeyed", "expected": expected, "bad_layers": bad[:10]})
            if unknown:
                failures.append({"unit": unit, "issue": "engine_inspector_precision_unavailable", "expected": expected, "unknown_layers": unknown[:10]})
    observed_fp32 = sum(1 for row in layers if _layer_precision(row) == "FP32")
    observed_fp16 = sum(1 for row in layers if _layer_precision(row) == "FP16")
    observed_int8 = sum(1 for row in layers if _layer_precision(row) == "INT8")
    return {
        "success": not failures,
        "failures": failures,
        "observed_by_unit": observed_by_unit,
        "observed_fp32_layers": observed_fp32,
        "observed_fp16_layers": observed_fp16,
        "observed_int8_layers": observed_int8,
        "num_inspector_layers": len(layers),
    }


def _attempt_build(args: argparse.Namespace, *, strongly_typed: bool, obey: bool) -> dict[str, Any]:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    if args.plugin:
        ctypes.CDLL(str(Path(args.plugin).resolve()), mode=ctypes.RTLD_GLOBAL)
    mapping = _load_mapping(args.layer_mapping)
    candidate = _load_json(args.candidate)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if strongly_typed:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    builder = trt.Builder(logger)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(args.onnx)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        return {"success": False, "failed_stage": "onnx_parse", "status": "onnx_parse_failed", "errors": errors}
    layer_precision_request = None
    config = builder.create_builder_config()
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_mib) << 20)
    except Exception:
        pass
    if not strongly_typed:
        config.set_flag(trt.BuilderFlag.FP16)
        if obey and hasattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS"):
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        elif hasattr(trt.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS"):
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        layer_precision_request = _set_layer_precision(network, mapping, candidate, trt)
    if hasattr(trt, "ProfilingVerbosity"):
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    profile = builder.create_optimization_profile()
    has_profile_shape = False
    for input_idx in range(network.num_inputs):
        tensor = network.get_input(input_idx)
        shape = tuple(int(dim) for dim in tensor.shape)
        if any(dim < 0 for dim in shape):
            min_shape = tuple(1 if dim < 0 else int(dim) for dim in shape)
            opt_shape = tuple(int(args.max_agents) if dim < 0 else int(dim) for dim in shape)
            max_shape = opt_shape
            profile.set_shape(tensor.name, min=min_shape, opt=opt_shape, max=max_shape)
            has_profile_shape = True
    if has_profile_shape:
        config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        return {"success": False, "failed_stage": "engine_build", "status": "engine_build_failed"}
    engine_path = Path(args.engine)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        return {"success": False, "failed_stage": "engine_deserialize", "status": "engine_deserialize_failed", "engine": str(engine_path)}
    inspector = engine.create_engine_inspector()
    raw_info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    layers = _parse_inspector(raw_info)
    verification = _verify_precision(layers, mapping, candidate)
    inspector_path = Path(args.report).with_suffix(".engine_inspector.json")
    inspector_path.write_text(raw_info, encoding="utf-8")
    return {
        "success": bool(verification["success"]),
        "status": "success" if verification["success"] else FALLBACK_STATUS,
        "failed_stage": None if verification["success"] else "precision_verification",
        "error": None if verification["success"] else "engine built, but requested layer precision could not be proven",
        "engine": str(engine_path),
        "route": "strongly_typed" if strongly_typed else ("weakly_typed_obey_constraints" if obey else "weakly_typed_prefer_constraints"),
        "layer_precision_request": layer_precision_request,
        "precision_verification": verification,
        "engine_inspector": str(inspector_path),
    }


def build_mixed_engine(args: argparse.Namespace) -> dict[str, Any]:
    probe = probe_tensorrt()
    if not args.layer_mapping or not Path(args.layer_mapping).is_file():
        return {
            "success": False,
            "status": FALLBACK_STATUS,
            "failed_stage": "layer_mapping",
            "error": "layer precision override cannot be reliably mapped to TensorRT layers without a validated full-engine layer mapping",
            "probe": probe,
            "onnx": args.onnx,
            "engine": args.engine,
            "precision_verification": None,
        }
    attempts: list[dict[str, Any]] = []
    routes: list[tuple[bool, bool]] = []
    if args.route in {"auto", "strongly_typed"}:
        routes.append((True, True))
    if args.route in {"auto", "constraints"}:
        routes.extend([(False, True), (False, False)])
    for strongly_typed, obey in routes:
        try:
            attempt = _attempt_build(args, strongly_typed=strongly_typed, obey=obey)
        except Exception as exc:
            attempt = {
                "success": False,
                "status": "mixed_engine_attempt_exception",
                "failed_stage": "exception",
                "error": repr(exc),
                "route": "strongly_typed" if strongly_typed else ("weakly_typed_obey_constraints" if obey else "weakly_typed_prefer_constraints"),
            }
        attempts.append(attempt)
        if attempt.get("success"):
            attempt.update({"probe": probe, "onnx": args.onnx})
            return attempt
    return {
        "success": False,
        "status": FALLBACK_STATUS,
        "failed_stage": "precision_verification",
        "error": "all TensorRT 10.x mixed-precision routes failed or could not prove actual requested layer precision",
        "probe": probe,
        "onnx": args.onnx,
        "engine": args.engine,
        "attempts": attempts,
        "precision_verification": None,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--plugin", default=None)
    parser.add_argument("--layer-mapping", default=None)
    parser.add_argument("--trt-root", default="${TENSORRT_ROOT}")
    parser.add_argument("--strongly-typed", action="store_true")
    parser.add_argument("--route", choices=["auto", "strongly_typed", "constraints"], default="auto")
    parser.add_argument("--workspace-mib", type=int, default=4096)
    parser.add_argument("--max-agents", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--report", default="outputs/latency_lut/mixed_precision_engine_build_report.json")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    lib = Path(args.trt_root) / "targets" / "x86_64-linux-gnu" / "lib"
    if lib.is_dir():
        os.environ["LD_LIBRARY_PATH"] = str(lib) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    report = build_mixed_engine(args)
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
