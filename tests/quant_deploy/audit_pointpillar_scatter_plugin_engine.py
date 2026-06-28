from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import onnx

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pointpillar_scatter_plugin_check import PLUGIN_DIR, build_plugin
from quant_deploy_utils import DEFAULT_TRT_ROOT, ensure_quant_deploy_run_dirs, read_json, save_json


PLUGIN_NAME = "PointPillarScatterTRT"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether fixed-K engines really contain PointPillarScatterTRT.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--trt_root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--trtexec_path", default=None)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--onnx_path", default=None)
    parser.add_argument("--plugin_so", default=None)
    parser.add_argument("--try_build_plugin", action="store_true")
    return parser.parse_args(argv)


def _run_text(cmd: list[str], timeout: int = 120) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        return {"command": cmd, "returncode": proc.returncode, "success": proc.returncode == 0, "output": proc.stdout}
    except Exception as exc:
        return {"command": cmd, "returncode": None, "success": False, "error": repr(exc), "output": ""}


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _find_plugin_so(dirs: dict[str, Path], explicit: str | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(sorted(PLUGIN_DIR.glob("**/*.so")))
    candidates.extend(sorted((dirs["output_root"] / "artifacts" / "plugins").glob("**/*.so")))
    dedup: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            dedup.append(path)
    return dedup


def _onnx_shape(value_info: onnx.ValueInfoProto) -> list[Any]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return []
    dims = []
    for dim in tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        else:
            dims.append(None)
    return dims


def _onnx_dtype(value_info: onnx.ValueInfoProto) -> str:
    try:
        return onnx.TensorProto.DataType.Name(value_info.type.tensor_type.elem_type)
    except Exception:
        return str(value_info.type.tensor_type.elem_type)


def _inspect_onnx(onnx_path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "onnx_path": str(onnx_path),
        "exists": onnx_path.exists(),
        "graph_inputs": [],
        "valid_voxel_mask_is_graph_input": False,
        "onnx_contains_pointpillar_scatter_plugin_node": False,
        "plugin_nodes": [],
        "custom_op_domains": [],
    }
    if not onnx_path.exists():
        return report
    model = onnx.load(str(onnx_path))
    report["graph_inputs"] = [
        {"name": item.name, "dtype": _onnx_dtype(item), "shape": _onnx_shape(item)}
        for item in model.graph.input
    ]
    report["valid_voxel_mask_is_graph_input"] = any(item["name"] == "valid_voxel_mask" for item in report["graph_inputs"])
    domains = sorted({node.domain for node in model.graph.node if node.domain})
    report["custom_op_domains"] = domains
    for node in model.graph.node:
        if node.op_type == PLUGIN_NAME:
            attrs = {}
            for attr in node.attribute:
                attrs[attr.name] = onnx.helper.get_attribute_value(attr)
                if isinstance(attrs[attr.name], bytes):
                    attrs[attr.name] = attrs[attr.name].decode("utf-8", errors="replace")
            report["plugin_nodes"].append(
                {
                    "name": node.name,
                    "op_type": node.op_type,
                    "domain": node.domain,
                    "inputs": list(node.input),
                    "outputs": list(node.output),
                    "attributes": attrs,
                }
            )
    report["onnx_contains_pointpillar_scatter_plugin_node"] = bool(report["plugin_nodes"])
    return report


def _layerinfo_contains_plugin(path: Path) -> dict[str, Any]:
    result = {"path": str(path), "exists": path.exists(), "contains_plugin": False, "matches": []}
    if not path.exists():
        return result
    text = path.read_text(encoding="utf-8", errors="replace")
    result["contains_plugin"] = PLUGIN_NAME in text or "pointpillar" in text.lower()
    for match in re.finditer(r".{0,120}(?:PointPillarScatterTRT|pointpillar.{0,30}scatter).{0,120}", text, re.I):
        result["matches"].append(match.group(0))
        if len(result["matches"]) >= 10:
            break
    return result


def _inspect_engine_python(engine_path: Path, plugin_so: Path | None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "engine_path": str(engine_path),
        "exists": engine_path.exists(),
        "plugin_loaded_during_engine_deserialize": False,
        "deserialize_success": False,
        "engine_contains_pointpillar_scatter_plugin_layer": False,
        "valid_voxel_mask_is_engine_input": False,
        "io_tensors": [],
        "inspector_contains_plugin": False,
        "error": None,
    }
    if not engine_path.exists():
        return report
    try:
        if plugin_so is not None and plugin_so.exists():
            ctypes.CDLL(str(plugin_so), mode=ctypes.RTLD_GLOBAL)
            report["plugin_loaded_during_engine_deserialize"] = True
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.ERROR)
        with engine_path.open("rb") as f:
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            report["error"] = "deserialize_cuda_engine returned None"
            return report
        report["deserialize_success"] = True
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            mode = engine.get_tensor_mode(name)
            shape = list(engine.get_tensor_shape(name))
            dtype = str(engine.get_tensor_dtype(name))
            report["io_tensors"].append({"name": name, "mode": str(mode), "shape": shape, "dtype": dtype})
        report["valid_voxel_mask_is_engine_input"] = any(
            item["name"] == "valid_voxel_mask" and "INPUT" in item["mode"] for item in report["io_tensors"]
        )
        try:
            inspector = engine.create_engine_inspector()
            info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
            report["engine_inspector_text_prefix"] = info[:2000]
            report["inspector_contains_plugin"] = PLUGIN_NAME in info or "pointpillar" in info.lower()
            report["engine_contains_pointpillar_scatter_plugin_layer"] = bool(report["inspector_contains_plugin"])
        except Exception as exc:
            report["inspector_error"] = repr(exc)
    except Exception as exc:
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
    return report


def _audit_engines(dirs: dict[str, Path], plugin_so: Path | None) -> dict[str, Any]:
    engine_root = dirs["engines"] / "fixed_k_scatter_plugin"
    engines = sorted(engine_root.glob("*/*.engine"))
    reports = []
    for engine in engines:
        layerinfo = engine.with_name("layerinfo_" + engine.stem.replace("lidar_pyramid_", "") + ".json")
        if not layerinfo.exists():
            layer_candidates = sorted(engine.parent.glob(f"layerinfo*{engine.stem.split('_bucket')[-1].replace('.engine', '')}*.json"))
            layerinfo = layer_candidates[0] if layer_candidates else layerinfo
        reports.append(
            {
                "engine_path": str(engine),
                "size_MB": engine.stat().st_size / (1024 * 1024),
                "layerinfo": _layerinfo_contains_plugin(layerinfo),
                "python_inspector": _inspect_engine_python(engine, plugin_so),
                "engine_strings_contains_plugin": PLUGIN_NAME in (_run_text(["strings", str(engine)], timeout=120).get("output") or ""),
            }
        )
    return {
        "engine_root": str(engine_root),
        "num_engines": len(engines),
        "engines": reports,
        "engine_contains_pointpillar_scatter_plugin_layer": bool(
            reports
            and all(
                item["layerinfo"].get("contains_plugin")
                or item["python_inspector"].get("engine_contains_pointpillar_scatter_plugin_layer")
                or item.get("engine_strings_contains_plugin")
                for item in reports
            )
        ),
        "valid_voxel_mask_is_engine_input": bool(
            reports and all(item["python_inspector"].get("valid_voxel_mask_is_engine_input") for item in reports if item.get("python_inspector"))
        ),
    }


def _source_feasibility() -> dict[str, Any]:
    cpp = PLUGIN_DIR / "pointpillar_scatter_plugin.cpp"
    header = PLUGIN_DIR / "pointpillar_scatter_plugin.h"
    source_text = (cpp.read_text(encoding="utf-8", errors="replace") if cpp.exists() else "") + "\n" + (
        header.read_text(encoding="utf-8", errors="replace") if header.exists() else ""
    )
    fixed_output = "exprBuilder.constant(mParams.numAgents)" in source_text
    enqueue_uses_param = "mParams.numAgents" in source_text and "launchPointPillarScatter" in source_text
    report = {
        "dynamic_agent_dim_single_engine_supported": False if fixed_output else None,
        "dynamic_agent_dim_per_N_engine_required": True if fixed_output else None,
        "plugin_output_agent_dim_source": "serialized plugin attribute num_agents" if fixed_output else "unknown",
        "getOutputDimensions_uses_fixed_num_agents": fixed_output,
        "enqueue_uses_fixed_num_agents": enqueue_uses_param,
        "required_plugin_changes": [
            "To support true dynamic N in one engine, add an input or shape-tensor source for N and return that dimension in getOutputDimensions.",
            "Update enqueue to derive numAgents from runtime shape/input instead of mParams.numAgents.",
            "Revalidate TensorRT dynamic output dimensions and bucket profiles for pairwise_t_matrix [1,N,N,4,4].",
        ]
        if fixed_output
        else [],
        "selected_test_plan": "Use per-N fallback engines dynamic_agent_dim_N1 and dynamic_agent_dim_N2; do not silently use padded max_cav=2 for this comparison.",
    }
    return report


def _current_agent_mode(onnx_report: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    input_shapes = {item["name"]: item["shape"] for item in onnx_report.get("graph_inputs") or []}
    pairwise_shape = input_shapes.get("pairwise_t_matrix")
    has_valid_agent_mask = "valid_agent_mask" in input_shapes
    has_record_len = "record_len" in input_shapes
    max_cav = None
    if isinstance(pairwise_shape, list) and len(pairwise_shape) >= 3:
        for dim in (pairwise_shape[1], pairwise_shape[2]):
            if isinstance(dim, int):
                max_cav = dim
                break
    tensor_shapes = read_json(dirs["debug"] / "tensor_shapes_report.json", default={}) or {}
    export_pairwise = ((tensor_shapes.get("input_shapes") or {}).get("pairwise_t_matrix") or [])
    if max_cav is None and len(export_pairwise) >= 3 and isinstance(export_pairwise[1], int):
        max_cav = int(export_pairwise[1])
    return {
        "current_agent_export_mode": "padded_agent_static_fixed_k_scatter_plugin" if has_valid_agent_mask else "unknown",
        "current_max_cav": max_cav,
        "uses_valid_agent_mask": has_valid_agent_mask,
        "uses_dynamic_agent_dim": False,
        "uses_padded_agent_static": has_valid_agent_mask,
        "record_len_exists": has_record_len,
        "valid_agent_mask_exists": has_valid_agent_mask,
        "pairwise_t_matrix_shape": pairwise_shape,
        "onnx_input_shapes": input_shapes,
        "evidence": "fixed-K plugin ONNX has valid_agent_mask input and no record_len input; plugin output uses static num_agents.",
    }


def _write_md(path: Path, title: str, rows: list[tuple[str, Any]]) -> None:
    lines = [f"# {title}", "", "field | value", "--- | ---"]
    for key, value in rows:
        if isinstance(value, (dict, list)):
            text = "`" + json.dumps(value, ensure_ascii=False)[:1000].replace("|", "/") + "`"
        else:
            text = str(value).replace("|", "/")
        lines.append(f"{key} | {text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    onnx_path = Path(args.onnx_path).expanduser() if args.onnx_path else dirs["onnx_fp32"] / "lidar_pyramid_fixed_k_scatter_plugin_fp32_dynamic.onnx"
    plugin_candidates = _find_plugin_so(dirs, args.plugin_so)
    plugin_so = _first_existing(plugin_candidates)
    build_report: dict[str, Any] | None = None
    if plugin_so is None and args.try_build_plugin:
        build_args = argparse.Namespace(
            output_root=args.output_root,
            trt_root=args.trt_root,
            trtexec_path=args.trtexec_path,
            timeout=args.timeout,
            skip_build_plugin=False,
        )
        build_report = build_plugin(build_args, dirs)
        if build_report.get("success"):
            plugin_so = Path(build_report["plugin_so"])

    so_report: dict[str, Any] = {
        "plugin_source_dir": str(PLUGIN_DIR),
        "plugin_source_dir_exists": PLUGIN_DIR.exists(),
        "plugin_compiled": plugin_so is not None and plugin_so.exists(),
        "plugin_shared_library_path": str(plugin_so) if plugin_so else None,
        "plugin_candidates": [str(path) for path in plugin_candidates],
        "build_report": build_report,
    }
    if plugin_so is not None and plugin_so.exists():
        so_report["ldd"] = _run_text(["ldd", str(plugin_so)])
        so_report["nm_D"] = _run_text(["nm", "-D", str(plugin_so)])
        so_report["strings_contains_pointpillar"] = PLUGIN_NAME in (_run_text(["strings", str(plugin_so)]).get("output") or "")
        nm_text = so_report["nm_D"].get("output") or ""
        strings_text = _run_text(["strings", str(plugin_so)]).get("output") or ""
        so_report["symbols_contain_plugin_creator"] = "PointPillarScatterPluginCreator" in nm_text or "PointPillarScatterPluginCreator" in strings_text
        so_report["symbols_contain_enqueue"] = "enqueue" in nm_text or "enqueue" in strings_text

    onnx_report = _inspect_onnx(onnx_path)
    engine_report = _audit_engines(dirs, plugin_so)
    build_logs = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in sorted((dirs["logs_build"]).glob("build_fixed_k_scatter_plugin*.log")) if path.exists())
    equivalence = read_json(dirs["debug"] / "pointpillar_scatter_plugin_equivalence.json", default={}) or {}
    precision_equivalence = equivalence.get("precisions") or {}
    valid_mask_consumed = bool(
        onnx_report.get("valid_voxel_mask_is_graph_input")
        and onnx_report.get("onnx_contains_pointpillar_scatter_plugin_node")
        and any("valid_voxel_mask" in node.get("inputs", []) for node in onnx_report.get("plugin_nodes") or [])
    )
    audit = {
        "plugin_compiled": bool(so_report["plugin_compiled"]),
        "plugin_shared_library_path": so_report["plugin_shared_library_path"],
        "plugin_loaded_during_engine_build": bool("--staticPlugins" in build_logs and (so_report["plugin_shared_library_path"] or "") in build_logs),
        "plugin_loaded_during_engine_deserialize": bool(
            engine_report.get("engines")
            and all((item.get("python_inspector") or {}).get("plugin_loaded_during_engine_deserialize") for item in engine_report.get("engines") or [])
        ),
        "onnx_contains_pointpillar_scatter_plugin_node": bool(onnx_report.get("onnx_contains_pointpillar_scatter_plugin_node")),
        "engine_contains_pointpillar_scatter_plugin_layer": bool(engine_report.get("engine_contains_pointpillar_scatter_plugin_layer")),
        "valid_voxel_mask_is_engine_input": bool(engine_report.get("valid_voxel_mask_is_engine_input")),
        "valid_voxel_mask_consumed_by_plugin": valid_mask_consumed,
        "current_fixed_k_results_are_plugin_results": bool(
            so_report["plugin_compiled"]
            and onnx_report.get("onnx_contains_pointpillar_scatter_plugin_node")
            and engine_report.get("engine_contains_pointpillar_scatter_plugin_layer")
            and valid_mask_consumed
        ),
        "plugin_shared_library": so_report,
        "onnx": onnx_report,
        "engines": engine_report,
        "plugin_equivalence": equivalence,
        "plugin_equivalence_fp32_max_abs_error": (precision_equivalence.get("fp32") or {}).get("max_abs_error"),
        "plugin_equivalence_fp16_max_abs_error": (precision_equivalence.get("fp16") or {}).get("max_abs_error"),
        "invalid_padded_voxel_affects_spatial_features": any(
            bool((item or {}).get("invalid_padded_voxel_affects_spatial_features")) for item in precision_equivalence.values()
        ),
    }
    save_json(audit, dirs["debug"] / "pointpillar_scatter_plugin_engine_audit.json")
    _write_md(
        dirs["summary"] / "pointpillar_scatter_plugin_engine_audit.md",
        "PointPillarScatterTRT Engine Audit",
        [(key, audit.get(key)) for key in (
            "plugin_compiled",
            "plugin_shared_library_path",
            "plugin_loaded_during_engine_build",
            "plugin_loaded_during_engine_deserialize",
            "onnx_contains_pointpillar_scatter_plugin_node",
            "engine_contains_pointpillar_scatter_plugin_layer",
            "valid_voxel_mask_is_engine_input",
            "valid_voxel_mask_consumed_by_plugin",
            "current_fixed_k_results_are_plugin_results",
            "plugin_equivalence_fp32_max_abs_error",
            "plugin_equivalence_fp16_max_abs_error",
            "invalid_padded_voxel_affects_spatial_features",
        )],
    )

    mode = _current_agent_mode(onnx_report, dirs)
    save_json(mode, dirs["debug"] / "current_fixed_k_plugin_agent_mode_audit.json")
    _write_md(
        dirs["summary"] / "current_fixed_k_plugin_agent_mode_audit.md",
        "Current Fixed-K Plugin Agent Mode Audit",
        [(key, mode.get(key)) for key in (
            "current_agent_export_mode",
            "current_max_cav",
            "uses_valid_agent_mask",
            "uses_dynamic_agent_dim",
            "uses_padded_agent_static",
            "record_len_exists",
            "valid_agent_mask_exists",
            "pairwise_t_matrix_shape",
            "evidence",
        )],
    )

    feasibility = _source_feasibility()
    save_json(feasibility, dirs["debug"] / "dynamic_agent_dim_scatter_plugin_feasibility.json")
    _write_md(
        dirs["summary"] / "dynamic_agent_dim_scatter_plugin_feasibility.md",
        "Dynamic Agent Dim Scatter Plugin Feasibility",
        [(key, feasibility.get(key)) for key in (
            "dynamic_agent_dim_single_engine_supported",
            "dynamic_agent_dim_per_N_engine_required",
            "plugin_output_agent_dim_source",
            "getOutputDimensions_uses_fixed_num_agents",
            "enqueue_uses_fixed_num_agents",
            "required_plugin_changes",
            "selected_test_plan",
        )],
    )
    return {"audit": audit, "current_agent_mode": mode, "dynamic_feasibility": feasibility}


def main(argv: list[str] | None = None) -> int:
    try:
        result = run_audit(parse_args(argv))
        print(json.dumps(result, indent=2, ensure_ascii=False)[:8000])
        return 0 if result["audit"].get("plugin_compiled") else 2
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
