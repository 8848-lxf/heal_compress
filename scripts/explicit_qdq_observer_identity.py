#!/usr/bin/env python3
"""Numerically verify calibration observer tensors against actual ONNX Q boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
for entry in (REPO, REPO.parent, Path("../../HEAL")):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

CHECKPOINT = Path("${MODEL_ROOT}/lidar_pyramid/net_epoch_bestval_at17.pth")
CONFIG = Path("${MODEL_ROOT}/lidar_pyramid/config.yaml")
HEAL_ROOT = Path("../../HEAL")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for item in value.values():
            try:
                return first_tensor(item)
            except RuntimeError:
                pass
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return first_tensor(item)
            except RuntimeError:
                pass
    raise RuntimeError("hook_value_has_no_tensor")


class Metrics:
    def __init__(self) -> None:
        self.count = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.max_abs = 0.0
        self.dot = 0.0
        self.ref_sq = 0.0
        self.cand_sq = 0.0
        self.shape_matches = True

    def update(self, reference: np.ndarray, candidate: np.ndarray) -> None:
        self.shape_matches &= reference.shape == candidate.shape
        if reference.shape != candidate.shape:
            return
        ref = np.asarray(reference, dtype=np.float64).reshape(-1)
        cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
        delta = cand - ref
        self.count += int(delta.size)
        self.sum_abs += float(np.abs(delta).sum())
        self.sum_sq += float(np.square(delta).sum())
        self.max_abs = max(self.max_abs, float(np.abs(delta).max(initial=0.0)))
        self.dot += float(np.dot(ref, cand))
        self.ref_sq += float(np.dot(ref, ref))
        self.cand_sq += float(np.dot(cand, cand))

    def result(self) -> dict[str, Any]:
        denominator = max(self.count, 1)
        cosine_denominator = np.sqrt(self.ref_sq * self.cand_sq)
        return {
            "shape_matches": self.shape_matches,
            "element_count": self.count,
            "mae": self.sum_abs / denominator,
            "rmse": np.sqrt(self.sum_sq / denominator),
            "max_error": self.max_abs,
            "cosine": self.dot / cosine_denominator if cosine_denominator > 0 else None,
        }


def make_augmented_onnx(base_onnx: Path, mapping: dict[str, Any], destination: Path) -> dict[str, dict[str, str]]:
    import onnx
    from onnx import TensorProto, helper
    from scripts.int8_equivalence_tensor_parity import replace_scatter_plugin_for_ort

    temporary = destination.with_name(destination.stem + "_scatter.onnx")
    replace_scatter_plugin_for_ort(base_onnx, temporary)
    model = onnx.load(str(temporary))
    value_info = {
        row.name: row
        for row in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    }
    existing = {row.name for row in model.graph.output}
    boundaries: dict[str, dict[str, str]] = {}
    node_by_name = {str(node.name): node for node in model.graph.node}
    for entry in mapping.get("entries", []):
        if str(entry.get("realized_request_precision", "")).lower() != "int8":
            continue
        module = str(entry["module_path"])
        node = node_by_name[str(entry["canonical_node_name"])]
        boundaries[module] = {"input": str(node.input[0]), "output": str(node.output[0])}
        for name in boundaries[module].values():
            if name in existing:
                continue
            model.graph.output.append(
                value_info.get(name, helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
            )
            existing.add(name)
    onnx.save(model, str(destination))
    temporary.unlink()
    return boundaries


def capture(args: argparse.Namespace) -> int:
    if os.environ.get("CONDA_DEFAULT_ENV", "") != "univ2x-opt":
        raise RuntimeError("observer_identity_capture_requires_univ2x-opt")
    from heal_compress.adapters.heal_lidar_adapter import HEALLiDARAdapter
    from quantization.config import OnnxExportConfig
    from search.integration.calibration_provider import _paired_batchnorm_path
    from search.integration.trt_compatible_export import build_search_trt_compatible_export_module

    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"force_rebuild_destination_exists:{output}")
    output.mkdir(parents=True)
    source = args.source_audit_root.resolve()
    artifacts = source / "search_maximal_legal_int8_force_rebuild/artifacts"
    work = source / "tensor_parity_work_v2"
    input_manifest = read_json(work / "ten_frame_inputs.json")
    frames = input_manifest.get("frames", [])[: int(args.frames)]
    if len(frames) != int(args.frames):
        raise RuntimeError(f"observer_identity_input_count_mismatch:{len(frames)}")
    mapping = read_json(artifacts / "canonical_layer_map.json")
    augmented = output / "observer_boundaries_fp32.onnx"
    boundaries = make_augmented_onnx(artifacts / "pruned_fp32.onnx", mapping, augmented)

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    adapter = HEALLiDARAdapter(heal_repo=HEAL_ROOT, config={"model": {"hypes_yaml": str(CONFIG)}})
    model = adapter.build_model(CONFIG, CHECKPOINT).to(device).eval()
    export_config = OnnxExportConfig(fixed_k=29696, min_agents=1, opt_agents=2, max_agents=2)
    wrapper = build_search_trt_compatible_export_module(
        model,
        output_names=export_config.output_names,
        fixed_k=export_config.fixed_k,
        modality="m1",
    ).to(device).eval()
    modules = dict(wrapper.named_modules())
    captures: dict[str, torch.Tensor] = {}
    handles = []

    def input_hook(module_path: str):
        def hook(_module: Any, inputs: Any) -> None:
            captures[f"{module_path}::input"] = first_tensor(inputs).detach().clone()
        return hook

    def output_hook(module_path: str):
        def hook(_module: Any, _inputs: Any, value: Any) -> None:
            # Clone inside the hook: many following ReLUs are in-place and would
            # otherwise mutate a detached view into a post-activation tensor.
            captures[f"{module_path}::output"] = first_tensor(value).detach().clone()
        return hook

    output_modules: dict[str, str] = {}
    for module_path in boundaries:
        paired = _paired_batchnorm_path(model, module_path) or module_path
        input_name = f"model.{module_path}"
        output_name = f"model.{paired}"
        if input_name not in modules or output_name not in modules:
            raise RuntimeError(f"observer_identity_hook_module_missing:{module_path}:{input_name}:{output_name}")
        output_modules[module_path] = paired
        handles.append(modules[input_name].register_forward_pre_hook(input_hook(module_path)))
        handles.append(modules[output_name].register_forward_hook(output_hook(module_path)))

    capture_rows: list[dict[str, Any]] = []
    try:
        for index, frame in enumerate(frames):
            values = np.load(frame["path"])
            feeds = {name: values[name] for name in values.files}
            captures.clear()
            with torch.inference_mode():
                wrapper(**{name: torch.as_tensor(value, device=device) for name, value in feeds.items()})
            payload = {f"feed::{name}": value for name, value in feeds.items()}
            payload.update(
                {
                    f"capture::{key}": value.float().cpu().numpy()
                    for key, value in captures.items()
                }
            )
            path = output / f"frame_{index:02d}_{frame['frame_id']}.npz"
            np.savez_compressed(path, **payload)
            capture_rows.append(
                {
                    "frame_id": frame["frame_id"],
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "capture_count": len(captures),
                }
            )
    finally:
        for handle in handles:
            handle.remove()
    write_json(
        output / "capture_manifest.json",
        {
            "frame_count": len(capture_rows),
            "boundary_count": len(boundaries) * 2,
            "base_onnx": str(artifacts / "pruned_fp32.onnx"),
            "base_onnx_sha256": sha256_file(artifacts / "pruned_fp32.onnx"),
            "augmented_onnx": str(augmented),
            "augmented_onnx_sha256": sha256_file(augmented),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "boundaries": boundaries,
            "output_observer_modules": output_modules,
            "captures": capture_rows,
        },
    )
    return 0


def compare(args: argparse.Namespace) -> int:
    if os.environ.get("CONDA_DEFAULT_ENV", "") != "modelopt":
        raise RuntimeError("observer_identity_compare_requires_modelopt")
    import onnxruntime as ort

    output = args.output.resolve()
    report_path = output / "observer_q_boundary_identity.json"
    if report_path.exists():
        raise RuntimeError(f"force_rebuild_report_exists:{report_path}")
    manifest = read_json(output / "capture_manifest.json")
    boundaries = manifest["boundaries"]
    session = ort.InferenceSession(str(manifest["augmented_onnx"]), providers=["CPUExecutionProvider"])
    metrics = {
        f"{module}::{role}": Metrics()
        for module in boundaries
        for role in ("input", "output")
    }
    per_frame: list[dict[str, Any]] = []
    for frame in manifest["captures"]:
        values = np.load(frame["path"])
        feeds = {
            name.removeprefix("feed::"): values[name]
            for name in values.files
            if name.startswith("feed::")
        }
        ort_values = session.run(None, feeds)
        ort_outputs = {row.name: value for row, value in zip(session.get_outputs(), ort_values)}
        frame_row: dict[str, Any] = {"frame_id": frame["frame_id"], "boundaries": {}}
        for module, roles in boundaries.items():
            module_row: dict[str, Any] = {}
            for role, tensor_name in roles.items():
                key = f"{module}::{role}"
                capture_key = f"capture::{key}"
                if capture_key not in values.files or tensor_name not in ort_outputs:
                    raise RuntimeError(f"observer_identity_tensor_missing:{key}:{tensor_name}")
                pytorch_value = values[capture_key]
                onnx_value = np.asarray(ort_outputs[tensor_name])
                metrics[key].update(onnx_value, pytorch_value)
                module_row[role] = {
                    "observer_shape": list(pytorch_value.shape),
                    "onnx_shape": list(onnx_value.shape),
                    "onnx_tensor": tensor_name,
                }
            frame_row["boundaries"][module] = module_row
        per_frame.append(frame_row)
    results = {key: value.result() for key, value in metrics.items()}
    shape_passed = all(row["shape_matches"] for row in results.values())
    numerical_passed = all(
        row["cosine"] is not None and row["cosine"] >= 0.9999
        for row in results.values()
    )
    report = {
        "status": "verified" if shape_passed and numerical_passed else "mismatch",
        "criterion": {"shape_exact": True, "cosine_minimum": 0.9999},
        "frame_count": len(per_frame),
        "boundary_count": len(results),
        "shape_passed": shape_passed,
        "numerical_passed": numerical_passed,
        "base_onnx": manifest["base_onnx"],
        "base_onnx_sha256": manifest["base_onnx_sha256"],
        "augmented_onnx_sha256": manifest["augmented_onnx_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "boundaries": boundaries,
        "output_observer_modules": manifest["output_observer_modules"],
        "metrics": results,
        "frames": per_frame,
    }
    write_json(output / "observer_q_boundary_identity.json", report)
    print(json.dumps({key: report[key] for key in ("status", "frame_count", "boundary_count")}, sort_keys=True))
    return 0 if report["status"] == "verified" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("capture", "compare"), required=True)
    parser.add_argument("--source-audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--frames", type=int, default=1)
    args = parser.parse_args()
    return capture(args) if args.action == "capture" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
