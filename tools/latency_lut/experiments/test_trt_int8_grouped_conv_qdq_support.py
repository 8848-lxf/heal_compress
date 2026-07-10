#!/usr/bin/env python3
"""TensorRT grouped Conv INT8 Q/DQ support experiment.

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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


DEFAULT_PER_GROUP_CHANNELS = [4, 8, 12, 16, 20, 24, 28, 32]
DEFAULT_HW = [(16, 16), (32, 32)]
GROUPS = 32
CONV_NODE_NAME = "/grouped_conv/Conv"


def _shape_text(shape: Sequence[int]) -> str:
    return "x".join(str(int(v)) for v in shape)


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


def _make_scale(name: str, value: float = 0.05) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray([value], dtype=np.float32), name=name)


def _make_zero_point(name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray([0], dtype=np.int8), name=name)


def _make_weight(name: str, shape: Sequence[int]) -> onnx.TensorProto:
    rng = np.random.default_rng(20260709 + int(shape[0]) + int(shape[1]))
    weight = rng.normal(loc=0.0, scale=0.02, size=tuple(int(v) for v in shape)).astype(np.float32)
    return numpy_helper.from_array(weight, name=name)


def _check_model(path: Path) -> None:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)


def build_fp16_onnx(path: Path, *, per_group: int, height: int, width: int) -> None:
    cin = GROUPS * per_group
    cout = GROUPS * per_group
    weight_shape = [cout, per_group, 3, 3]

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, cin, height, width])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, cout, height, width])
    weight = _make_weight("conv_weight", weight_shape)
    conv = helper.make_node(
        "Conv",
        ["input", "conv_weight"],
        ["conv_out"],
        name=CONV_NODE_NAME,
        group=GROUPS,
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    relu = helper.make_node("Relu", ["conv_out"], ["output"], name="/grouped_conv/Relu")
    graph = helper.make_graph([conv, relu], "grouped_conv_fp16", [x], [y], [weight])
    model = helper.make_model(
        graph,
        producer_name="heal_compress_grouped_conv_int8_experiment",
        opset_imports=[helper.make_operatorsetid("", 17)],
    )
    model.ir_version = 8
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    _check_model(path)


def build_int8_qdq_onnx(path: Path, *, per_group: int, height: int, width: int) -> None:
    cin = GROUPS * per_group
    cout = GROUPS * per_group
    weight_shape = [cout, per_group, 3, 3]

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, cin, height, width])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, cout, height, width])
    initializers = [
        _make_weight("conv_weight", weight_shape),
        _make_scale("input_scale", 0.05),
        _make_zero_point("input_zero_point"),
        _make_scale("weight_scale", 0.02),
        _make_zero_point("weight_zero_point"),
        _make_scale("output_scale", 0.05),
        _make_zero_point("output_zero_point"),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["input", "input_scale", "input_zero_point"],
            ["input_q"],
            name="/input/QuantizeLinear",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["input_q", "input_scale", "input_zero_point"],
            ["input_dq"],
            name="/input/DequantizeLinear",
        ),
        helper.make_node(
            "QuantizeLinear",
            ["conv_weight", "weight_scale", "weight_zero_point"],
            ["conv_weight_q"],
            name="/grouped_conv/weight/QuantizeLinear",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["conv_weight_q", "weight_scale", "weight_zero_point"],
            ["conv_weight_dq"],
            name="/grouped_conv/weight/DequantizeLinear",
        ),
        helper.make_node(
            "Conv",
            ["input_dq", "conv_weight_dq"],
            ["conv_out"],
            name=CONV_NODE_NAME,
            group=GROUPS,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        helper.make_node("Relu", ["conv_out"], ["relu_out"], name="/grouped_conv/Relu"),
        helper.make_node(
            "QuantizeLinear",
            ["relu_out", "output_scale", "output_zero_point"],
            ["output_q"],
            name="/output/QuantizeLinear",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["output_q", "output_scale", "output_zero_point"],
            ["output"],
            name="/output/DequantizeLinear",
        ),
    ]
    graph = helper.make_graph(nodes, "grouped_conv_int8_qdq", [x], [y], initializers)
    model = helper.make_model(
        graph,
        producer_name="heal_compress_grouped_conv_int8_experiment",
        opset_imports=[helper.make_operatorsetid("", 17)],
    )
    model.ir_version = 8
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    _check_model(path)


def trt_env(trt_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    lib_dirs = [
        trt_root / "lib",
        trt_root / "targets" / "x86_64-linux-gnu" / "lib",
    ]
    bin_dirs = [
        trt_root / "bin",
        trt_root / "targets" / "x86_64-linux-gnu" / "bin",
    ]
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
    success = (returncode == 0) and engine_path.is_file()
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


def summarize_trtexec_error(log_text: str) -> str:
    interesting = []
    for line in log_text.splitlines():
        if "[E]" in line or "Error[" in line or "FAILED" in line or "Could not find any implementation" in line:
            interesting.append(line.strip())
    if interesting:
        return "\n".join(interesting[-12:])
    tail = [line.strip() for line in log_text.splitlines()[-12:] if line.strip()]
    return "\n".join(tail)


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
    return CONV_NODE_NAME in blob or "grouped_conv" in blob


def _collect_datatypes(layer: Dict[str, Any]) -> List[str]:
    dtypes: List[str] = []
    for section in ("Inputs", "Outputs"):
        values = layer.get(section)
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, dict):
                value = item.get("Format/Datatype")
                if value is not None:
                    dtypes.append(str(value))
    return dtypes


def parse_int8_realization(layer_info_path: Path) -> Dict[str, Any]:
    if not layer_info_path.is_file():
        return {
            "conv_layer_found": False,
            "conv_realized_int8": False,
            "conv_layer_names": [],
            "conv_layer_datatypes": [],
        }
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
    dtypes: List[str] = []
    names: List[str] = []
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


def case_dir_name(per_group: int, height: int, width: int) -> str:
    return f"pg{per_group:02d}_hw{height}x{width}"


def run_case(args: argparse.Namespace, *, per_group: int, height: int, width: int) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    case_dir = output_dir / "cases" / case_dir_name(per_group, height, width)
    fp16_onnx = case_dir / "model_fp16.onnx"
    int8_onnx = case_dir / "model_int8_qdq.onnx"
    build_fp16_onnx(fp16_onnx, per_group=per_group, height=height, width=width)
    build_int8_qdq_onnx(int8_onnx, per_group=per_group, height=height, width=width)

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
    cin = GROUPS * per_group
    cout = GROUPS * per_group
    row = {
        "per_group": per_group,
        "C_in_total": cin,
        "C_out_total": cout,
        "groups": GROUPS,
        "weight_shape": [cout, per_group, 3, 3],
        "H": height,
        "W": width,
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
        "int8_error_summary": int8.get("error_summary", ""),
        "int8_could_not_find_implementation": bool(int8.get("could_not_find_implementation")),
        "int8_conv_layer_found": bool(realization.get("conv_layer_found")),
        "int8_conv_realized_int8": bool(realization.get("conv_realized_int8")),
        "int8_conv_layer_names": realization.get("conv_layer_names", []),
        "int8_conv_layer_datatypes": realization.get("conv_layer_datatypes", []),
        "fp16_build": fp16,
        "int8_build": int8,
    }
    _save_json(case_dir / "case_result.json", row)
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "per_group",
        "C_in_total",
        "C_out_total",
        "groups",
        "weight_shape",
        "H",
        "W",
        "fp16_build_success",
        "int8_build_success",
        "int8_conv_realized_int8",
        "int8_could_not_find_implementation",
        "fp16_engine_exists",
        "int8_engine_exists",
        "int8_layer_info_exists",
        "int8_error_summary",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(row[field]) if isinstance(row.get(field), list) else row.get(field) for field in fields})


def summarize_by_per_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for per_group in DEFAULT_PER_GROUP_CHANNELS:
        subset = [r for r in rows if int(r["per_group"]) == per_group]
        if not subset:
            continue
        out[str(per_group)] = {
            "case_count": len(subset),
            "all_fp16_build_success": all(bool(r["fp16_build_success"]) for r in subset),
            "all_int8_build_success": all(bool(r["int8_build_success"]) for r in subset),
            "any_int8_build_success": any(bool(r["int8_build_success"]) for r in subset),
            "all_int8_conv_realized_int8": all(bool(r["int8_conv_realized_int8"]) for r in subset),
            "any_could_not_find_implementation": any(bool(r["int8_could_not_find_implementation"]) for r in subset),
            "per_hw": [
                {
                    "H": r["H"],
                    "W": r["W"],
                    "fp16": r["fp16_build_success"],
                    "int8": r["int8_build_success"],
                    "int8_realized": r["int8_conv_realized_int8"],
                    "could_not_find_implementation": r["int8_could_not_find_implementation"],
                }
                for r in subset
            ],
        }
    failures = [r for r in rows if not r["int8_build_success"]]
    return {
        "by_per_group": out,
        "int8_failures": [
            {
                "per_group": r["per_group"],
                "H": r["H"],
                "W": r["W"],
                "could_not_find_implementation": r["int8_could_not_find_implementation"],
                "error_summary": r["int8_error_summary"],
            }
            for r in failures
        ],
        "failures_all_non_8_multiples": bool(failures) and all(int(r["per_group"]) % 8 != 0 for r in failures),
        "non_8_multiple_failures": [r["per_group"] for r in failures if int(r["per_group"]) % 8 != 0],
        "multiple_of_8_failures": [r["per_group"] for r in failures if int(r["per_group"]) % 8 == 0],
    }


def answer_line(summary: Dict[str, Any], per_group: int) -> str:
    info = summary["by_per_group"].get(str(per_group), {})
    if not info:
        return f"- per_group={per_group}: not tested"
    status = "success" if info.get("all_int8_build_success") else ("partial" if info.get("any_int8_build_success") else "failed")
    realized = "INT8 realized" if info.get("all_int8_conv_realized_int8") else "INT8 realization not proven"
    impl = "could-not-find-implementation seen" if info.get("any_could_not_find_implementation") else "no could-not-find-implementation"
    return f"- per_group={per_group}: INT8 build {status}; {realized}; {impl}"


def write_report(path: Path, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    lines = ["# TensorRT Grouped Conv INT8 Q/DQ Support Experiment", ""]
    lines.append("## Matrix")
    lines.append("|per_group|H|W|C_in|C_out|weight_shape|FP16 build|INT8 build|INT8 realized|Could not find implementation|")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            "|{per_group}|{H}|{W}|{C_in_total}|{C_out_total}|{weight_shape}|{fp16}|{int8}|{realized}|{impl}|".format(
                per_group=row["per_group"],
                H=row["H"],
                W=row["W"],
                C_in_total=row["C_in_total"],
                C_out_total=row["C_out_total"],
                weight_shape=row["weight_shape"],
                fp16=row["fp16_build_success"],
                int8=row["int8_build_success"],
                realized=row["int8_conv_realized_int8"],
                impl=row["int8_could_not_find_implementation"],
            )
        )
    lines.append("")
    lines.append("## Required Answers")
    for per_group in DEFAULT_PER_GROUP_CHANNELS:
        lines.append(answer_line(summary, per_group))
    lines.append("")
    lines.append(
        f"- Failures concentrated in per_group not multiple of 8: {summary['failures_all_non_8_multiples']} "
        f"(multiple-of-8 failures={sorted(set(summary['multiple_of_8_failures']))}, "
        f"non-8-multiple failures={sorted(set(summary['non_8_multiple_failures']))})"
    )
    pg4 = summary["by_per_group"].get("4", {})
    pg24 = summary["by_per_group"].get("24", {})
    lines.append(
        "- per_group=4 special-case implication: "
        + (
            "can be considered as a possible special case because all tested H/W built INT8"
            if pg4.get("all_int8_build_success")
            else "not supported by this experiment because at least one tested H/W failed"
        )
    )
    lines.append(
        "- per_group=24 implication: "
        + (
            "successful, so a power-of-two-only rule would be too strict; multiple-of-8 is the relevant candidate rule"
            if pg24.get("all_int8_build_success")
            else "not successful in this experiment; it does not support a simple multiple-of-8 rule"
        )
    )
    lines.append("")
    lines.append("## INT8 Failure Summaries")
    if summary["int8_failures"]:
        for item in summary["int8_failures"]:
            lines.append(
                f"- per_group={item['per_group']} H={item['H']} W={item['W']} "
                f"could_not_find_implementation={item['could_not_find_implementation']}:"
            )
            lines.append("```text")
            lines.append(str(item["error_summary"]))
            lines.append("```")
    else:
        lines.append("- No INT8 build failures.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_hw(values: Sequence[str]) -> List[Tuple[int, int]]:
    out = []
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
    parser.add_argument("--per-groups", nargs="*", type=int, default=DEFAULT_PER_GROUP_CHANNELS)
    parser.add_argument("--hw", nargs="*", default=[f"{h}x{w}" for h, w in DEFAULT_HW])
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    if not args.trt_root.exists():
        raise SystemExit(f"TRT root does not exist: {args.trt_root}")
    hws = parse_hw(args.hw)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for per_group in args.per_groups:
        if per_group <= 0:
            raise SystemExit(f"invalid per_group: {per_group}")
        for height, width in hws:
            print(f"[case] per_group={per_group} H={height} W={width}", flush=True)
            rows.append(run_case(args, per_group=int(per_group), height=int(height), width=int(width)))

    summary = summarize_by_per_group(rows)
    payload = {
        "trt_root": str(args.trt_root),
        "groups": GROUPS,
        "conv_node_name": CONV_NODE_NAME,
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
