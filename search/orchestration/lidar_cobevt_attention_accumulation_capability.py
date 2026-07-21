"""Audit the locally installed TensorRT operand/accumulator capabilities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_TRT_ROOT = Path(
    "/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matrix_multiply_section(header_text: str) -> str:
    start = header_text.find("class IMatrixMultiplyLayer")
    if start < 0:
        return ""
    next_class = header_text.find("//! \\class ", start + 1)
    return header_text[start : next_class if next_class >= 0 else len(header_text)]


def parse_installed_header_capability(header_text: str) -> dict[str, Any]:
    text = str(header_text)
    section = _matrix_multiply_section(text)
    accumulator_api = bool(
        re.search(
            r"(?:set|get)(?:Accumulator|Accumulation)(?:Precision|Type)|"
            r"accumulator_precision|accumulatorPrecision",
            section,
            re.IGNORECASE,
        )
    )
    return {
        "matrix_multiply_layer_present": bool(section),
        "strongly_typed_rejects_set_precision": bool(
            re.search(
                r"Strongly[- ]typed networks reject calls to (?:method )?setPrecision",
                text,
                re.IGNORECASE,
            )
        ),
        "separate_accumulator_precision_api": accumulator_api,
        "bf16_type_present": "DataType::kBF16" in text,
        "quantize_layer_present": "class IQuantizeLayer" in text,
        "matrix_multiply_section": section,
    }


def build_precision_capability_rows(
    *,
    trt_version: str,
    matrix_multiply_present: bool,
    separate_accumulator_api: bool,
    inspector_accumulator_field: bool,
    bf16_type_present: bool,
    quantize_layer_present: bool,
) -> list[dict[str, Any]]:
    direct_accumulator = bool(separate_accumulator_api or inspector_accumulator_field)
    return [
        {
            "phenotype": "F32A32",
            "operand_type": "FP32",
            "accumulator_type": "FP32",
            "native_request_status": "supported" if matrix_multiply_present else "unsupported",
            "maximum_native_evidence": "A" if matrix_multiply_present else "none",
            "trt_version": trt_version,
        },
        {
            "phenotype": "F16A32",
            "operand_type": "FP16",
            "accumulator_type": "FP32",
            "native_request_status": (
                "separately_expressible" if direct_accumulator else "not_separately_expressible"
            ),
            "maximum_native_evidence": "A" if direct_accumulator else "C",
            "trt_version": trt_version,
        },
        {
            "phenotype": "F16A16",
            "operand_type": "FP16",
            "accumulator_type": "FP16",
            "native_request_status": (
                "separately_expressible" if direct_accumulator else "not_separately_expressible"
            ),
            "maximum_native_evidence": "A" if direct_accumulator else "C",
            "trt_version": trt_version,
        },
        {
            "phenotype": "BF16A32",
            "operand_type": "BF16",
            "accumulator_type": "FP32",
            "native_request_status": (
                "operand_type_available_accumulator_unproven"
                if bf16_type_present
                else "unsupported"
            ),
            "maximum_native_evidence": "C" if bf16_type_present else "none",
            "trt_version": trt_version,
        },
        {
            "phenotype": "I8A32I",
            "operand_type": "INT8",
            "accumulator_type": "INT32",
            "native_request_status": (
                "qdq_pattern_requires_realization_test" if quantize_layer_present else "unsupported"
            ),
            "maximum_native_evidence": "C" if quantize_layer_present else "none",
            "trt_version": trt_version,
        },
    ]


def validate_trt_runtime_paths(
    trt_root: str | Path, *, ld_library_path: str
) -> dict[str, str]:
    root = Path(trt_root).expanduser().resolve()
    candidates = (
        root / "lib/libnvinfer.so.10",
        root / "targets/x86_64-linux-gnu/lib/libnvinfer.so.10",
    )
    library = next((path for path in candidates if path.exists()), None)
    if library is None:
        raise ValueError(f"tensorrt_library_missing:{root}")
    active = {str(Path(value).expanduser().resolve()) for value in ld_library_path.split(":") if value}
    if str(library.parent.resolve()) not in active:
        raise ValueError("tensorrt_library_path_not_active")
    return {"trt_root": str(root), "libnvinfer": str(library.resolve())}


def resolve_engine_inspector_api_name(module_attributes: set[str]) -> str:
    for name in ("EngineInspector", "IEngineInspector"):
        if name in module_attributes:
            return name
    raise ValueError("engine_inspector_api_missing")


def _python_probe(python: Path, env: dict[str, str]) -> dict[str, Any]:
    code = """
import json
import tensorrt as trt
inspector_name = 'EngineInspector' if hasattr(trt, 'EngineInspector') else 'IEngineInspector'
inspector_type = getattr(trt, inspector_name)
payload = {
    'version': trt.__version__,
    'matrix_multiply_attributes': sorted(x for x in dir(trt.IMatrixMultiplyLayer) if 'precision' in x.lower() or 'accum' in x.lower() or 'type' in x.lower()),
    'network_matrix_methods': sorted(x for x in dir(trt.INetworkDefinition) if 'matrix' in x.lower()),
    'network_cast_methods': sorted(x for x in dir(trt.INetworkDefinition) if 'cast' in x.lower()),
    'network_quantize_methods': sorted(x for x in dir(trt.INetworkDefinition) if 'quant' in x.lower()),
    'engine_inspector_api_name': inspector_name,
    'inspector_attributes': sorted(x for x in dir(inspector_type) if 'layer' in x.lower() or 'precision' in x.lower() or 'accum' in x.lower()),
    'bf16_type_present': hasattr(trt.DataType, 'BF16'),
}
print(json.dumps(payload, sort_keys=True))
"""
    completed = subprocess.run(
        [str(python), "-c", code],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"tensorrt_python_probe_failed:{completed.stdout}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_installed_tensorrt(
    *, trt_root: str | Path, python: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    root = Path(trt_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    header = root / "include/NvInfer.h"
    if not header.is_file():
        raise FileNotFoundError(header)
    header_text = header.read_text(encoding="utf-8", errors="replace")
    header_capability = parse_installed_header_capability(header_text)
    library_dirs = [
        root / "lib",
        root / "targets/x86_64-linux-gnu/lib",
    ]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(
        [str(path) for path in library_dirs if path.is_dir()]
        + [env.get("LD_LIBRARY_PATH", "")]
    )
    runtime = validate_trt_runtime_paths(root, ld_library_path=env["LD_LIBRARY_PATH"])
    python_api = _python_probe(Path(python).expanduser().resolve(), env)
    matrix_attributes = tuple(str(value).lower() for value in python_api["matrix_multiply_attributes"])
    inspector_attributes = tuple(str(value).lower() for value in python_api["inspector_attributes"])
    python_accumulator = any("accum" in value for value in matrix_attributes)
    inspector_accumulator = any("accum" in value for value in inspector_attributes)
    rows = build_precision_capability_rows(
        trt_version=str(python_api["version"]),
        matrix_multiply_present=header_capability["matrix_multiply_layer_present"],
        separate_accumulator_api=bool(
            header_capability["separate_accumulator_precision_api"] or python_accumulator
        ),
        inspector_accumulator_field=inspector_accumulator,
        bf16_type_present=bool(
            header_capability["bf16_type_present"] or python_api["bf16_type_present"]
        ),
        quantize_layer_present=header_capability["quantize_layer_present"],
    )
    inventory = {
        "trt_root": str(root),
        "python": str(Path(python).expanduser().resolve()),
        "runtime": runtime,
        "header": str(header),
        "header_sha256": _sha256(header),
        "libnvinfer_sha256": _sha256(Path(runtime["libnvinfer"])),
        "header_capability": {
            key: value
            for key, value in header_capability.items()
            if key != "matrix_multiply_section"
        },
        "python_api": python_api,
        "inspector_exposes_accumulator": inspector_accumulator,
        "separate_accumulator_precision_api": bool(
            header_capability["separate_accumulator_precision_api"] or python_accumulator
        ),
    }
    (output / "installed_tensorrt_api_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "installed_header_evidence.txt").write_text(
        header_capability["matrix_multiply_section"], encoding="utf-8"
    )
    _write_csv(output / "precision_capability_matrix.csv", rows)
    conclusion = [
        "# Installed TensorRT precision capability conclusion",
        "",
        f"- TensorRT: `{python_api['version']}`",
        f"- Strongly typed rejects setPrecision: `{header_capability['strongly_typed_rejects_set_precision']}`",
        f"- Separate MatrixMultiply accumulator API: `{inventory['separate_accumulator_precision_api']}`",
        f"- EngineInspector accumulator field: `{inspector_accumulator}`",
        "- Native F16A32/F16A16 cannot receive Level-A evidence from these APIs when both fields above are false.",
        "- cuBLASLt/CUTLASS descriptors or an explicit plugin contract are required for direct accumulator evidence.",
    ]
    (output / "precision_capability_conclusion.md").write_text(
        "\n".join(conclusion) + "\n", encoding="utf-8"
    )
    return inventory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trt-root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    audit_installed_tensorrt(
        trt_root=args.trt_root, python=args.python, output_dir=args.output_dir
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
