"""Build a native post-scatter TensorRT engine from a searched Pyramid ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from deploy.post_scatter import (
    audit_post_scatter_onnx as audit_formal_post_scatter_onnx,
    post_scatter_shape_profiles as formal_post_scatter_shape_profiles,
)

from .external_scatter_onnx import externalize_scatter


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trtexec(tensorrt_root: Path) -> Path:
    for relative in (
        "bin/trtexec",
        "targets/x86_64-linux-gnu/bin/trtexec",
    ):
        candidate = tensorrt_root / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"trtexec not found below {tensorrt_root}")


def post_scatter_shape_profiles(
    input_names: Sequence[str] = ("spatial_features", "pairwise_t_matrix"),
) -> Mapping[str, Mapping[str, tuple[int, ...]]]:
    return formal_post_scatter_shape_profiles(input_names)


def build_command(
    *,
    trtexec: Path,
    onnx_path: Path,
    engine_path: Path,
    layer_info_path: Path,
    input_names: Sequence[str] = ("spatial_features", "pairwise_t_matrix"),
) -> list[str]:
    profiles = post_scatter_shape_profiles(input_names)
    command = [
        str(trtexec),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--stronglyTyped",
        "--noTF32",
        "--skipInference",
        "--profilingVerbosity=detailed",
        "--memPoolSize=workspace:4096",
        f"--exportLayerInfo={layer_info_path}",
    ]
    if profiles:
        for kind in ("min", "opt", "max"):
            values = ",".join(
                f"{name}:{'x'.join(str(value) for value in profiles[name][kind])}"
                for name in sorted(profiles)
            )
            command.append(f"--{kind}Shapes={values}")
    return command


def _onnx_audit(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path), load_external_data=True)
    report = dict(audit_formal_post_scatter_onnx(path))
    qdq_count = sum(
        node.op_type in {"QuantizeLinear", "DequantizeLinear"}
        for node in model.graph.node
    )
    report["qdq_node_count"] = qdq_count
    return report


def build_post_scatter_engine(
    source_onnx: Path,
    output_dir: Path,
    *,
    tensorrt_root: Path,
    physical_gpu: int,
) -> dict[str, Any]:
    source = source_onnx.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    root = tensorrt_root.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    post_scatter_onnx = destination / "post_scatter_qdq.onnx"
    engine = destination / "post_scatter.plan"
    layer_info = destination / "engine_layer_info.json"
    log = destination / "trtexec.log"
    boundary = externalize_scatter(source, post_scatter_onnx)
    onnx_audit = _onnx_audit(post_scatter_onnx)
    if not onnx_audit["passed"]:
        raise RuntimeError(f"post_scatter_onnx_audit_failed:{onnx_audit['issues']}")
    executable = _trtexec(root)
    command = build_command(
        trtexec=executable,
        onnx_path=post_scatter_onnx,
        engine_path=engine,
        layer_info_path=layer_info,
        input_names=onnx_audit["input_names"],
    )
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(int(physical_gpu))
    library_paths = [
        root / "targets/x86_64-linux-gnu/lib",
        root / "lib",
    ]
    env["LD_LIBRARY_PATH"] = ":".join(
        [str(path) for path in library_paths if path.is_dir()]
        + [env.get("LD_LIBRARY_PATH", "")]
    )
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )
    log.write_text(completed.stdout or "", encoding="utf-8")
    success = completed.returncode == 0 and engine.is_file() and layer_info.is_file()
    report = {
        "schema_version": "heal-post-scatter-deployment-v1",
        "success": success,
        "source_onnx": str(source),
        "source_onnx_sha256": _sha256(source),
        "post_scatter_onnx": str(post_scatter_onnx),
        "post_scatter_onnx_sha256": _sha256(post_scatter_onnx),
        "engine": str(engine),
        "engine_sha256": _sha256(engine) if engine.is_file() else "",
        "layer_info": str(layer_info),
        "trtexec_log": str(log),
        "trtexec_returncode": completed.returncode,
        "physical_gpu": int(physical_gpu),
        "shape_profiles": post_scatter_shape_profiles(onnx_audit["input_names"]),
        "strongly_typed": True,
        "plugin_required": False,
        "boundary_rewrite": boundary,
        "onnx_audit": onnx_audit,
    }
    report_path = destination / "deployment_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not success:
        raise RuntimeError(
            f"post_scatter_trtexec_failed:rc={completed.returncode}:log={log}"
        )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_onnx", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tensorrt-root", required=True, type=Path)
    parser.add_argument("--physical-gpu", required=True, type=int)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = build_post_scatter_engine(
        arguments.source_onnx,
        arguments.output_dir,
        tensorrt_root=arguments.tensorrt_root,
        physical_gpu=arguments.physical_gpu,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
