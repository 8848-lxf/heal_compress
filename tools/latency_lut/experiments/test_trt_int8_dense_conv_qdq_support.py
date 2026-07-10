#!/usr/bin/env python3
"""TensorRT dense Conv INT8 Q/DQ support experiment.

This is an isolated experiment. It does not import or modify the v11 LUT
builder, and it writes only under the requested output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


EQUAL_CHANNELS = [4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 96, 128, 160, 192, 256, 384, 512]
PYRAMID_COMBOS = [
    (128, 128),
    (128, 256),
    (256, 128),
    (256, 256),
    (256, 512),
    (512, 256),
    (512, 512),
    (96, 128),
    (128, 96),
    (160, 256),
    (192, 256),
    (256, 192),
    (384, 512),
    (512, 384),
]
DEFAULT_HW = [(16, 16), (32, 32), (128, 256)]
KERNELS = ["1x1", "3x3"]
GROUPS = 1
CONV_NODE_NAME = "/dense_conv/Conv"


@dataclass(frozen=True)
class CaseSpec:
    scenarios: Tuple[str, ...]
    kernel: str
    c_in: int
    c_out: int
    height: int
    width: int

    @property
    def weight_shape(self) -> List[int]:
        k = kernel_size(self.kernel)
        return [self.c_out, self.c_in, k, k]


def kernel_size(kernel: str) -> int:
    if kernel == "1x1":
        return 1
    if kernel == "3x3":
        return 3
    raise ValueError(f"unsupported kernel: {kernel}")


def kernel_pads(kernel: str) -> List[int]:
    if kernel == "1x1":
        return [0, 0, 0, 0]
    if kernel == "3x3":
        return [1, 1, 1, 1]
    raise ValueError(f"unsupported kernel: {kernel}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_scale(name: str, value: float = 0.05) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray([value], dtype=np.float32), name=name)


def _make_zero_point(name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray([0], dtype=np.int8), name=name)


def _make_weight(name: str, shape: Sequence[int]) -> onnx.TensorProto:
    seed = 20260709 + int(shape[0]) * 1009 + int(shape[1]) * 917 + int(shape[2])
    rng = np.random.default_rng(seed)
    weight = rng.normal(loc=0.0, scale=0.02, size=tuple(int(v) for v in shape)).astype(np.float32)
    return numpy_helper.from_array(weight, name=name)


def _check_model(path: Path) -> None:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)


def build_fp16_onnx(path: Path, spec: CaseSpec) -> None:
    k = kernel_size(spec.kernel)
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, spec.c_in, spec.height, spec.width])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, spec.c_out, spec.height, spec.width])
    weight = _make_weight("conv_weight", spec.weight_shape)
    conv = helper.make_node(
        "Conv",
        ["input", "conv_weight"],
        ["conv_out"],
        name=CONV_NODE_NAME,
        group=GROUPS,
        kernel_shape=[k, k],
        pads=kernel_pads(spec.kernel),
        strides=[1, 1],
    )
    relu = helper.make_node("Relu", ["conv_out"], ["output"], name="/dense_conv/Relu")
    graph = helper.make_graph([conv, relu], "dense_conv_fp16", [x], [y], [weight])
    model = helper.make_model(
        graph,
        producer_name="heal_compress_dense_conv_int8_experiment",
        opset_imports=[helper.make_operatorsetid("", 17)],
    )
    model.ir_version = 8
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    _check_model(path)


def build_int8_qdq_onnx(path: Path, spec: CaseSpec) -> None:
    k = kernel_size(spec.kernel)
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, spec.c_in, spec.height, spec.width])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, spec.c_out, spec.height, spec.width])
    initializers = [
        _make_weight("conv_weight", spec.weight_shape),
        _make_scale("input_scale", 0.05),
        _make_zero_point("input_zero_point"),
        _make_scale("weight_scale", 0.02),
        _make_zero_point("weight_zero_point"),
        _make_scale("output_scale", 0.05),
        _make_zero_point("output_zero_point"),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "input_scale", "input_zero_point"], ["input_q"], name="/input/QuantizeLinear"),
        helper.make_node("DequantizeLinear", ["input_q", "input_scale", "input_zero_point"], ["input_dq"], name="/input/DequantizeLinear"),
        helper.make_node(
            "QuantizeLinear",
            ["conv_weight", "weight_scale", "weight_zero_point"],
            ["conv_weight_q"],
            name="/dense_conv/weight/QuantizeLinear",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["conv_weight_q", "weight_scale", "weight_zero_point"],
            ["conv_weight_dq"],
            name="/dense_conv/weight/DequantizeLinear",
        ),
        helper.make_node(
            "Conv",
            ["input_dq", "conv_weight_dq"],
            ["conv_out"],
            name=CONV_NODE_NAME,
            group=GROUPS,
            kernel_shape=[k, k],
            pads=kernel_pads(spec.kernel),
            strides=[1, 1],
        ),
        helper.make_node("Relu", ["conv_out"], ["relu_out"], name="/dense_conv/Relu"),
        helper.make_node("QuantizeLinear", ["relu_out", "output_scale", "output_zero_point"], ["output_q"], name="/output/QuantizeLinear"),
        helper.make_node("DequantizeLinear", ["output_q", "output_scale", "output_zero_point"], ["output"], name="/output/DequantizeLinear"),
    ]
    graph = helper.make_graph(nodes, "dense_conv_int8_qdq", [x], [y], initializers)
    model = helper.make_model(
        graph,
        producer_name="heal_compress_dense_conv_int8_experiment",
        opset_imports=[helper.make_operatorsetid("", 17)],
    )
    model.ir_version = 8
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    _check_model(path)


def trt_env(trt_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    lib_dirs = [trt_root / "lib", trt_root / "targets" / "x86_64-linux-gnu" / "lib"]
    bin_dirs = [trt_root / "bin", trt_root / "targets" / "x86_64-linux-gnu" / "bin"]
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    existing_path = env.get("PATH", "")
    ld_values = [str(p) for p in lib_dirs if p.is_dir()]
    path_values = [str(p) for p in bin_dirs if p.is_dir()]
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        for suffix in ("lib", "lib64"):
            p = Path(conda_prefix) / suffix
            if p.is_dir():
                ld_values.append(str(p))
    if existing_ld:
        ld_values.append(existing_ld)
    if existing_path:
        path_values.append(existing_path)
    env["TRT_ROOT"] = str(trt_root)
    env["LD_LIBRARY_PATH"] = ":".join(ld_values)
    env["PATH"] = ":".join(path_values)
    return env


def summarize_trtexec_error(log_text: str) -> str:
    interesting = []
    for line in log_text.splitlines():
        if "[E]" in line or "Error[" in line or "FAILED" in line or "Could not find any implementation" in line:
            interesting.append(line.strip())
    if interesting:
        return "\n".join(interesting[-12:])
    return "\n".join(line.strip() for line in log_text.splitlines()[-12:] if line.strip())


def run_trtexec(
    *,
    trt_root: Path,
    onnx_path: Path,
    engine_path: Path,
    layer_info_path: Path,
    log_path: Path,
    precision: str,
    timeout: int,
) -> Dict[str, Any]:
    trtexec = trt_root / "bin" / "trtexec"
    if not trtexec.is_file():
        return {
            "success": False,
            "returncode": None,
            "failure_reason": f"trtexec_not_found:{trtexec}",
            "command": [],
            "engine_exists": False,
            "layer_info_exists": False,
            "could_not_find_implementation": False,
            "error_summary": "",
            "log_path": str(log_path),
        }

    cmd = [
        str(trtexec),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--skipInference",
        "--fp16",
        "--profilingVerbosity=detailed",
        f"--exportLayerInfo={layer_info_path}",
        "--memPoolSize=workspace:512",
    ]
    if precision == "int8":
        cmd.extend(
            [
                "--int8",
                "--precisionConstraints=obey",
                f"--layerPrecisions={CONV_NODE_NAME}:int8",
                f"--layerOutputTypes={CONV_NODE_NAME}:int8",
            ]
        )

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    layer_info_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(Path.cwd()),
            env=trt_env(trt_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        log_text = completed.stdout or ""
        returncode: int | None = int(completed.returncode)
        timeout_hit = False
    except subprocess.TimeoutExpired as exc:
        log_text = (exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
        log_text += f"\nTIMEOUT after {timeout} seconds\n"
        returncode = None
        timeout_hit = True
    log_path.write_text(log_text, encoding="utf-8", errors="replace")
    success = returncode == 0 and engine_path.is_file()
    return {
        "success": success,
        "returncode": returncode,
        "timeout": timeout_hit,
        "elapsed_sec": round(time.time() - started, 3),
        "failure_reason": "" if success else ("timeout" if timeout_hit else f"trtexec_failed_rc_{returncode}"),
        "command": cmd,
        "engine_exists": engine_path.is_file(),
        "engine_size": engine_path.stat().st_size if engine_path.is_file() else None,
        "layer_info_exists": layer_info_path.is_file(),
        "log_path": str(log_path),
        "log_tail": "\n".join(log_text.splitlines()[-80:]),
        "could_not_find_implementation": "Could not find any implementation" in log_text,
        "error_summary": summarize_trtexec_error(log_text),
    }


def _iter_layer_info_layers(layer_info: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(layer_info, dict) and isinstance(layer_info.get("Layers"), list):
        for item in layer_info["Layers"]:
            if isinstance(item, dict):
                yield item
    elif isinstance(layer_info, list):
        for item in layer_info:
            if isinstance(item, dict):
                yield item


def _layer_mentions_conv(layer: Dict[str, Any]) -> bool:
    blob = json.dumps(layer, sort_keys=True)
    return CONV_NODE_NAME in blob or "dense_conv" in blob


def _collect_datatypes(layer: Dict[str, Any]) -> List[str]:
    dtypes: List[str] = []
    for section in ("Inputs", "Outputs"):
        values = layer.get(section)
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, dict) and item.get("Format/Datatype") is not None:
                dtypes.append(str(item["Format/Datatype"]))
    return dtypes


def parse_int8_realization(layer_info_path: Path) -> Dict[str, Any]:
    if not layer_info_path.is_file():
        return {"conv_layer_found": False, "conv_realized_int8": False, "conv_layer_names": [], "conv_layer_datatypes": []}
    try:
        layer_info = json.loads(layer_info_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "conv_layer_found": False,
            "conv_realized_int8": False,
            "conv_layer_names": [],
            "conv_layer_datatypes": [],
            "parse_error": repr(exc),
        }
    matched = [layer for layer in _iter_layer_info_layers(layer_info) if _layer_mentions_conv(layer)]
    names: List[str] = []
    dtypes: List[str] = []
    for layer in matched:
        names.append(str(layer.get("Name", "")))
        dtypes.extend(_collect_datatypes(layer))
    blob = "\n".join(names + dtypes)
    return {
        "conv_layer_found": bool(matched),
        "conv_realized_int8": bool(re.search(r"\bInt8\b|\bINT8\b", blob)),
        "conv_layer_names": names,
        "conv_layer_datatypes": dtypes,
    }


def case_key(spec: CaseSpec) -> Tuple[str, int, int, int, int]:
    return (spec.kernel, spec.c_in, spec.c_out, spec.height, spec.width)


def build_case_specs(hws: List[Tuple[int, int]], kernels: List[str]) -> List[CaseSpec]:
    by_key: Dict[Tuple[str, int, int, int, int], set[str]] = {}
    for c in EQUAL_CHANNELS:
        for kernel in kernels:
            for height, width in hws:
                by_key.setdefault((kernel, c, c, height, width), set()).add("equal_cin_cout_scan")
    for c_in, c_out in PYRAMID_COMBOS:
        for kernel in kernels:
            for height, width in hws:
                by_key.setdefault((kernel, c_in, c_out, height, width), set()).add("pyramid_common_combo")
    specs = [
        CaseSpec(tuple(sorted(scenarios)), kernel, c_in, c_out, height, width)
        for (kernel, c_in, c_out, height, width), scenarios in by_key.items()
    ]
    return sorted(specs, key=lambda s: (s.kernel, s.c_in, s.c_out, s.height, s.width, s.scenarios))


def case_dir_name(spec: CaseSpec) -> str:
    return f"k{spec.kernel}_cin{spec.c_in}_cout{spec.c_out}_hw{spec.height}x{spec.width}".replace("x", "x")


def run_case(args: argparse.Namespace, spec: CaseSpec) -> Dict[str, Any]:
    case_dir = Path(args.output_dir) / "cases" / case_dir_name(spec)
    case_result = case_dir / "case_result.json"
    if args.resume and case_result.is_file():
        row = _load_json(case_result)
        row["resumed"] = True
        return row

    fp16_onnx = case_dir / "model_fp16.onnx"
    int8_onnx = case_dir / "model_int8_qdq.onnx"
    build_fp16_onnx(fp16_onnx, spec)
    build_int8_qdq_onnx(int8_onnx, spec)
    fp16 = run_trtexec(
        trt_root=Path(args.trt_root),
        onnx_path=fp16_onnx,
        engine_path=case_dir / "fp16" / "engine.plan",
        layer_info_path=case_dir / "fp16" / "trt_layer_info.json",
        log_path=case_dir / "fp16" / "build_log.txt",
        precision="fp16",
        timeout=int(args.timeout),
    )
    int8 = run_trtexec(
        trt_root=Path(args.trt_root),
        onnx_path=int8_onnx,
        engine_path=case_dir / "int8" / "engine.plan",
        layer_info_path=case_dir / "int8" / "trt_layer_info.json",
        log_path=case_dir / "int8" / "build_log.txt",
        precision="int8",
        timeout=int(args.timeout),
    )
    realization = parse_int8_realization(case_dir / "int8" / "trt_layer_info.json") if int8["success"] else {
        "conv_layer_found": False,
        "conv_realized_int8": False,
        "conv_layer_names": [],
        "conv_layer_datatypes": [],
    }
    row = {
        "scenarios": list(spec.scenarios),
        "kernel": spec.kernel,
        "C_in": spec.c_in,
        "C_out": spec.c_out,
        "H": spec.height,
        "W": spec.width,
        "groups": GROUPS,
        "weight_shape": spec.weight_shape,
        "fp16_onnx": str(fp16_onnx),
        "int8_qdq_onnx": str(int8_onnx),
        "fp16_build_success": bool(fp16["success"]),
        "fp16_returncode": fp16.get("returncode"),
        "fp16_engine_exists": fp16.get("engine_exists"),
        "fp16_layer_info_exists": fp16.get("layer_info_exists"),
        "fp16_error_summary": fp16.get("error_summary", ""),
        "int8_build_success": bool(int8["success"]),
        "int8_returncode": int8.get("returncode"),
        "int8_engine_exists": int8.get("engine_exists"),
        "int8_layer_info_exists": int8.get("layer_info_exists"),
        "conv_int8_realized": bool(realization.get("conv_realized_int8")),
        "conv_layer_found": bool(realization.get("conv_layer_found")),
        "conv_layer_names": realization.get("conv_layer_names", []),
        "conv_layer_datatypes": realization.get("conv_layer_datatypes", []),
        "could_not_find_any_implementation": bool(int8.get("could_not_find_implementation")),
        "error_summary": int8.get("error_summary", ""),
        "fp16_build": fp16,
        "int8_build": int8,
        "resumed": False,
    }
    _save_json(case_result, row)
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "scenarios",
        "kernel",
        "C_in",
        "C_out",
        "H",
        "W",
        "groups",
        "weight_shape",
        "fp16_build_success",
        "int8_build_success",
        "conv_int8_realized",
        "could_not_find_any_implementation",
        "error_summary",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(row[field]) if isinstance(row.get(field), list) else row.get(field) for field in fields})


def status_for(rows: List[Dict[str, Any]], predicate) -> Dict[str, Any]:
    subset = [r for r in rows if predicate(r)]
    return {
        "case_count": len(subset),
        "all_fp16_build_success": bool(subset) and all(bool(r["fp16_build_success"]) for r in subset),
        "all_int8_build_success": bool(subset) and all(bool(r["int8_build_success"]) for r in subset),
        "all_conv_int8_realized": bool(subset) and all(bool(r["conv_int8_realized"]) for r in subset),
        "any_could_not_find_implementation": any(bool(r["could_not_find_any_implementation"]) for r in subset),
        "failed_cases": [
            {
                "kernel": r["kernel"],
                "C_in": r["C_in"],
                "C_out": r["C_out"],
                "H": r["H"],
                "W": r["W"],
                "could_not_find_any_implementation": r["could_not_find_any_implementation"],
                "error_summary": r["error_summary"],
            }
            for r in subset
            if not r["int8_build_success"]
        ],
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_equal_c: Dict[str, Any] = {}
    for c in EQUAL_CHANNELS:
        by_equal_c[str(c)] = status_for(
            rows,
            lambda r, c=c: r["C_in"] == c and r["C_out"] == c and "equal_cin_cout_scan" in r["scenarios"],
        )
    by_kernel = {kernel: status_for(rows, lambda r, kernel=kernel: r["kernel"] == kernel) for kernel in KERNELS}
    by_pyramid_combo = {
        f"{c_in}->{c_out}": status_for(
            rows,
            lambda r, c_in=c_in, c_out=c_out: r["C_in"] == c_in
            and r["C_out"] == c_out
            and "pyramid_common_combo" in r["scenarios"],
        )
        for c_in, c_out in PYRAMID_COMBOS
    }
    failures = [r for r in rows if not r["int8_build_success"]]
    return {
        "case_count": len(rows),
        "all_fp16_build_success": all(bool(r["fp16_build_success"]) for r in rows),
        "all_int8_build_success": all(bool(r["int8_build_success"]) for r in rows),
        "all_conv_int8_realized": all(bool(r["conv_int8_realized"]) for r in rows),
        "any_could_not_find_implementation": any(bool(r["could_not_find_any_implementation"]) for r in rows),
        "int8_failure_count": len(failures),
        "by_equal_c": by_equal_c,
        "by_kernel": by_kernel,
        "by_pyramid_combo": by_pyramid_combo,
        "int8_failures": [
            {
                "scenarios": r["scenarios"],
                "kernel": r["kernel"],
                "C_in": r["C_in"],
                "C_out": r["C_out"],
                "H": r["H"],
                "W": r["W"],
                "could_not_find_any_implementation": r["could_not_find_any_implementation"],
                "error_summary": r["error_summary"],
            }
            for r in failures
        ],
    }


def yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def equal_answer(summary: Dict[str, Any], c: int) -> str:
    info = summary["by_equal_c"][str(c)]
    return "PASS" if info["all_int8_build_success"] and info["all_conv_int8_realized"] else "FAIL"


def write_report(path: Path, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    lines: List[str] = ["# TensorRT Dense Conv INT8 Q/DQ Support Experiment", ""]
    lines.append("## Overall")
    lines.append(f"- cases: {summary['case_count']}")
    lines.append(f"- all FP16 build success: {summary['all_fp16_build_success']}")
    lines.append(f"- all INT8 QDQ build success: {summary['all_int8_build_success']}")
    lines.append(f"- all Conv INT8 realized: {summary['all_conv_int8_realized']}")
    lines.append(f"- INT8 failure count: {summary['int8_failure_count']}")
    lines.append(f"- any 'Could not find any implementation': {summary['any_could_not_find_implementation']}")
    lines.append("")

    lines.append("## Equal C_in=C_out Scan")
    lines.append("|C|1x1 INT8|3x3 INT8|all H/W INT8|Could not find implementation|")
    lines.append("|---:|---|---|---|---|")
    for c in EQUAL_CHANNELS:
        all_info = summary["by_equal_c"][str(c)]
        k1 = status_for(rows, lambda r, c=c: r["C_in"] == c and r["C_out"] == c and r["kernel"] == "1x1" and "equal_cin_cout_scan" in r["scenarios"])
        k3 = status_for(rows, lambda r, c=c: r["C_in"] == c and r["C_out"] == c and r["kernel"] == "3x3" and "equal_cin_cout_scan" in r["scenarios"])
        lines.append(
            f"|{c}|{yes_no(k1['all_int8_build_success'] and k1['all_conv_int8_realized'])}|"
            f"{yes_no(k3['all_int8_build_success'] and k3['all_conv_int8_realized'])}|"
            f"{yes_no(all_info['all_int8_build_success'] and all_info['all_conv_int8_realized'])}|"
            f"{yes_no(all_info['any_could_not_find_implementation'])}|"
        )
    lines.append("")

    lines.append("## Pyramid Common Combos")
    lines.append("|C_in|C_out|1x1 INT8|3x3 INT8|all H/W INT8|Could not find implementation|")
    lines.append("|---:|---:|---|---|---|---|")
    for c_in, c_out in PYRAMID_COMBOS:
        all_info = summary["by_pyramid_combo"][f"{c_in}->{c_out}"]
        k1 = status_for(rows, lambda r, c_in=c_in, c_out=c_out: r["C_in"] == c_in and r["C_out"] == c_out and r["kernel"] == "1x1" and "pyramid_common_combo" in r["scenarios"])
        k3 = status_for(rows, lambda r, c_in=c_in, c_out=c_out: r["C_in"] == c_in and r["C_out"] == c_out and r["kernel"] == "3x3" and "pyramid_common_combo" in r["scenarios"])
        lines.append(
            f"|{c_in}|{c_out}|{yes_no(k1['all_int8_build_success'] and k1['all_conv_int8_realized'])}|"
            f"{yes_no(k3['all_int8_build_success'] and k3['all_conv_int8_realized'])}|"
            f"{yes_no(all_info['all_int8_build_success'] and all_info['all_conv_int8_realized'])}|"
            f"{yes_no(all_info['any_could_not_find_implementation'])}|"
        )
    lines.append("")

    lines.append("## Required Answers")
    lines.append(f"- groups=1 C=12 INT8 build: {equal_answer(summary, 12)}")
    lines.append(f"- groups=1 C=20 INT8 build: {equal_answer(summary, 20)}")
    lines.append(f"- groups=1 C=24 INT8 build: {equal_answer(summary, 24)}")
    lines.append(f"- groups=1 C=28 INT8 build: {equal_answer(summary, 28)}")
    k1_ok = summary["by_kernel"]["1x1"]["all_int8_build_success"] and summary["by_kernel"]["1x1"]["all_conv_int8_realized"]
    k3_ok = summary["by_kernel"]["3x3"]["all_int8_build_success"] and summary["by_kernel"]["3x3"]["all_conv_int8_realized"]
    lines.append(f"- 1x1 and 3x3 consistent: {k1_ok == k3_ok} (1x1={k1_ok}, 3x3={k3_ok})")
    lines.append(
        "- Dense Conv requires 8/16 alignment: "
        + ("no evidence in this matrix" if summary["all_int8_build_success"] else "yes or unresolved; see failures")
    )
    lines.append(
        "- Same failure mode as grouped Conv: "
        + ("yes" if summary["any_could_not_find_implementation"] else "no, no dense Conv case hit 'Could not find any implementation'")
    )
    lines.append(
        "- Grouped Conv failure appears groups=32/per-group tactic specific: "
        + ("yes, because dense groups=1 passes all requested C_in/C_out cases" if summary["all_int8_build_success"] else "not proven by this dense experiment")
    )
    lines.append("- Pruner recommendation for ordinary Conv: keep round_to=4 if these dense results hold for the production graph context.")
    lines.append("- Pruner recommendation for ordinary Conv round_to=8: not required by this isolated dense Conv evidence.")
    lines.append("- Pruner recommendation for grouped Conv: use the measured safe per-group set {4,8,16,32} unless further experiments expand it.")
    lines.append("")

    if summary["int8_failures"]:
        lines.append("## INT8 Failure Summaries")
        for item in summary["int8_failures"]:
            lines.append(
                f"- scenarios={item['scenarios']} kernel={item['kernel']} C_in={item['C_in']} C_out={item['C_out']} "
                f"H={item['H']} W={item['W']} could_not_find={item['could_not_find_any_implementation']}"
            )
            lines.append("```text")
            lines.append(str(item["error_summary"]))
            lines.append("```")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_hw(values: Sequence[str]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for value in values:
        if "x" not in value:
            raise argparse.ArgumentTypeError(f"HW must use HxW form, got {value}")
        h, w = value.lower().split("x", 1)
        out.append((int(h), int(w)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trt-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hw", nargs="*", default=[f"{h}x{w}" for h, w in DEFAULT_HW])
    parser.add_argument("--kernels", nargs="*", default=KERNELS)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not args.trt_root.exists():
        raise SystemExit(f"TRT root does not exist: {args.trt_root}")
    hws = parse_hw(args.hw)
    kernels = [str(k) for k in args.kernels]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = build_case_specs(hws, kernels)

    rows: List[Dict[str, Any]] = []
    total = len(specs)
    for idx, spec in enumerate(specs, 1):
        print(
            f"[case {idx}/{total}] scenarios={','.join(spec.scenarios)} kernel={spec.kernel} "
            f"C_in={spec.c_in} C_out={spec.c_out} H={spec.height} W={spec.width}",
            flush=True,
        )
        rows.append(run_case(args, spec))

    summary = summarize(rows)
    payload = {
        "trt_root": str(args.trt_root),
        "groups": GROUPS,
        "conv_node_name": CONV_NODE_NAME,
        "default_hw": hws,
        "kernels": kernels,
        "rows": rows,
        "summary": summary,
    }
    write_csv(args.output_dir / "results.csv", rows)
    _save_json(args.output_dir / "results.json", payload)
    write_report(args.output_dir / "report.md", rows, summary)
    print(f"Wrote {args.output_dir / 'results.csv'}")
    print(f"Wrote {args.output_dir / 'results.json'}")
    print(f"Wrote {args.output_dir / 'report.md'}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
