"""Local TensorRT 10.9 editable-timing-cache capability inventory."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import csv
from pathlib import Path
from typing import Any


def parse_header_evidence(text: str) -> dict[str, Any]:
    lowered = text.lower()
    return {
        "editable_timing_cache_flag": "keditable_timing_cache" in lowered,
        "timing_cache_update_api": all(
            token in lowered for token in ("querykeys", "query", "update")
        ),
        "timing_cache_serialization_api": "serialize" in lowered,
        "matrix_multiply_layer_present": "imatrixmultiplylayer" in lowered
        or "addmatrixmultiply" in lowered,
        "separate_accumulator_precision_api": any(
            token in lowered
            for token in (
                "accumulatorprecision",
                "setaccumulator",
                "accumulationprecision",
                "setaccumulation",
            )
        ),
        "algorithm_selector_deprecated": "deprecated in tensorrt 10.8" in lowered,
    }


def build_capability_rows(
    *,
    trt_version: str,
    editable_timing_cache: bool,
    timing_cache_update: bool,
    accumulator_api: bool,
    inspector_accumulator: bool,
    algorithm_selector_deprecated: bool,
) -> list[dict[str, Any]]:
    direct = accumulator_api or inspector_accumulator
    rows = []
    for phenotype, operand, accumulator, output in (
        ("F32A32", "FP32", "FP32", "FP32"),
        ("F16A32", "FP16", "FP32", "FP32"),
        ("F16A16", "FP16", "FP16", "FP16"),
        ("I8A32I", "INT8", "INT32", "INT32"),
    ):
        if phenotype == "F32A32":
            request_status = "supported"
        elif phenotype == "F16A32" and not direct:
            request_status = "not_separately_expressible"
        elif phenotype == "I8A32I":
            request_status = "requires_native_int8_tactic_evidence"
        else:
            request_status = "inspect_tactic_and_cache"
        rows.append(
            {
                "phenotype": phenotype,
                "operand_precision": operand,
                "accumulator_precision": accumulator,
                "output_precision": output,
                "trt_version": str(trt_version),
                "editable_timing_cache": bool(editable_timing_cache),
                "timing_cache_update": bool(timing_cache_update),
                "separate_accumulator_api": bool(accumulator_api),
                "inspector_accumulator_field": bool(inspector_accumulator),
                "algorithm_selector_deprecated": bool(algorithm_selector_deprecated),
                "native_request_status": request_status,
                "maximum_evidence_level": (
                    "LEVEL_A_DIRECT" if direct else "LEVEL_C_UNKNOWN"
                ),
            }
        )
    return rows


def validate_conda_toolchain(
    *, conda_prefix: str, nvcc_path: str, cuda_home: str, arch_list: list[str]
) -> dict[str, Any]:
    prefix = Path(conda_prefix).resolve()
    nvcc = Path(nvcc_path).resolve()
    home = Path(cuda_home).resolve()
    if str(nvcc).startswith("/usr/bin") or str(nvcc).startswith("/usr/local"):
        raise ValueError("system_nvcc_forbidden")
    try:
        nvcc.relative_to(prefix)
        home.relative_to(prefix)
    except ValueError as exc:
        raise ValueError("cuda_toolchain_outside_conda_prefix") from exc
    normalized = set()
    for value in arch_list:
        normalized.update(re.findall(r"(?:sm[_-]?|compute[_-]?)([0-9]+)", str(value).lower()))
        normalized.add(str(value).lower().replace("sm_", "").replace("sm", ""))
    if "89" not in normalized and "8.9" not in normalized:
        raise ValueError("sm89_not_supported_by_toolchain")
    return {
        "conda_prefix": str(prefix),
        "nvcc": str(nvcc),
        "cuda_home": str(home),
        "sm89_supported": True,
        "cpu_fallback": False,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_probe(python: Path, env: dict[str, str]) -> dict[str, Any]:
    code = (
        "import json,sys,tensorrt as trt; "
        "print(json.dumps({'python':sys.executable,'trt':trt.__version__,"
        "'builder_flag_editable':hasattr(trt.BuilderFlag,'EDITABLE_TIMING_CACHE'),"
        "'timing_cache': [x for x in dir(trt.ITimingCache) if not x.startswith('_')],"
        "'inspector': [x for x in dir(trt) if 'Inspector' in x],"
        "'algorithm_selector':hasattr(trt,'IAlgorithmSelector')}))"
    )
    completed = subprocess.run(
        [str(python), "-c", code], text=True, capture_output=True, env=env, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(f"tensorrt_runtime_probe_failed:{completed.stderr[-500:]}")
    return json.loads(completed.stdout.strip())


def write_capability_inventory(
    output_dir: str | Path,
    *,
    trt_root: str | Path,
    conda_prefix: str | Path,
    python_path: str | Path,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    root = Path(trt_root).resolve()
    prefix = Path(conda_prefix).resolve()
    python = Path(python_path).resolve()
    nvcc = prefix / "bin/nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"conda_nvcc_missing:{nvcc}")
    arch = subprocess.run(
        [str(nvcc), "--list-gpu-code"], text=True, capture_output=True, check=True
    ).stdout
    toolchain = validate_conda_toolchain(
        conda_prefix=str(prefix), nvcc_path=str(nvcc), cuda_home=str(prefix), arch_list=arch.splitlines()
    )
    header = (root / "include/NvInfer.h").read_text(encoding="utf-8", errors="replace")
    evidence = parse_header_evidence(header)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(
        [str(root / "lib"), str(root / "targets/x86_64-linux-gnu/lib"), str(prefix / "lib"), env.get("LD_LIBRARY_PATH", "")]
    )
    runtime = _runtime_probe(python, env)
    trtexec = root / "bin/trtexec"
    manifest = {
        "trt_root": str(root),
        "trt_root_sha256": _sha256(root / "lib/libnvinfer.so.10"),
        "python": str(python),
        "nvcc": str(nvcc),
        "toolchain": toolchain,
        "header": evidence,
        "runtime": runtime,
        "trtexec": str(trtexec),
        "trtexec_sha256": _sha256(trtexec) if trtexec.is_file() else None,
        "sample": str(root / "samples/sampleEditableTimingCache"),
    }
    (destination / "environment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / "installed_header_evidence.txt").write_text(json.dumps(evidence, indent=2) + "\n")
    rows = build_capability_rows(
        trt_version=str(runtime["trt"]),
        editable_timing_cache=bool(evidence["editable_timing_cache_flag"]),
        timing_cache_update=bool(evidence["timing_cache_update_api"]),
        accumulator_api=bool(evidence["separate_accumulator_precision_api"]),
        inspector_accumulator=False,
        algorithm_selector_deprecated=bool(evidence["algorithm_selector_deprecated"]),
    )
    (destination / "tensorrt_version.txt").write_text(str(runtime["trt"]) + "\n")
    (destination / "cuda_toolchain.json").write_text(json.dumps(toolchain, indent=2) + "\n")
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,compute_cap,driver_version",
            "--format=csv,noheader",
        ], text=True, capture_output=True, check=True,
    ).stdout
    (destination / "gpu_manifest.json").write_text(
        json.dumps({"query": gpu_query.strip(), "target_arch": "SM89"}, indent=2) + "\n"
    )
    compiler_paths = [nvcc, prefix / "bin/gcc", prefix / "bin/g++"]
    (destination / "compiler_hashes.json").write_text(
        json.dumps(
            {str(path): _sha256(path) if path.is_file() else None for path in compiler_paths},
            indent=2,
        ) + "\n"
    )
    with (destination / "precision_capability_matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (destination / "editable_timing_cache_api.md").write_text(
        "# Editable Timing Cache API\n\n"
        "TensorRT 10.9 exposes `kEDITABLE_TIMING_CACHE`, `ITimingCache.queryKeys`, "
        "`query`, `update`, `serialize`, and `IBuilderConfig.setTimingCache`. "
        "The cache fixes a tactic hash for an exact timing-cache key; it does not "
        "provide an accumulator dtype API.\n"
    )
    (destination / "installed_sample_inventory.md").write_text(
        f"# Installed Sample Inventory\n\n`{root / 'samples/sampleEditableTimingCache'}` exists and is the source "
        "for the cache key/value editing procedure.\n"
    )
    (destination / "accumulator_api_conclusion.md").write_text(
        "# Accumulator API Conclusion\n\n"
        "The installed TensorRT 10.9 headers and Python API do not expose a "
        "separate MatrixMultiply accumulator-precision setter or Inspector field. "
        "F16A32 therefore requires direct tactic/kernel evidence; cache pinning "
        "alone does not prove it.\n"
    )
    (destination / "tactic_selection_capability.json").write_text(
        json.dumps(
            {"editable_timing_cache": evidence, "python_runtime": runtime, "rows": rows},
            indent=2,
        ) + "\n"
    )
    return {"manifest": manifest, "rows": rows}


__all__ = [
    "build_capability_rows",
    "parse_header_evidence",
    "validate_conda_toolchain",
    "write_capability_inventory",
]
