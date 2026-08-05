"""Aggregate same-protocol HEAL CARLA evaluations across model families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _named_paths(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name.strip() or not path.strip():
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        if name in result:
            raise ValueError(f"duplicate model name: {name}")
        result[name] = Path(path).expanduser().resolve()
    return result


def _named_ints(values: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        name, separator, index = value.partition("=")
        if not separator:
            raise ValueError(f"expected NAME=INDEX, got {value!r}")
        result[name] = int(index)
    return result


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _search_metrics(path: Path, index: int | None) -> Mapping[str, Any]:
    payload = _read_json(path)
    if isinstance(payload, list):
        if index is None:
            raise ValueError(f"search result {path} is an array and requires an index")
        payload = payload[index]
    return {
        "map": float(payload["mAP"]),
        "ap30": float(payload["AP@0.3"]),
        "ap50": float(payload["AP@0.5"]),
        "ap70": float(payload["AP@0.7"]),
        "forward_p50_ms": float(payload["forward_p50_ms"]),
        "evaluated_frames": int(payload["num_evaluated_frames"]),
        "candidate_hash": payload.get("candidate_hash"),
        "source": str(path),
        "source_index": index,
    }


def _accuracy(payload: Mapping[str, Any], backend: str, protocol: str) -> Mapping[str, Any]:
    return payload["models"][backend]["accuracy"][protocol]


def aggregate(
    evaluations: Mapping[str, Path],
    deployments: Mapping[str, Path],
    search_results: Mapping[str, Path],
    search_indices: Mapping[str, int],
) -> Mapping[str, Any]:
    names = set(evaluations)
    if names != set(deployments) or names != set(search_results):
        raise ValueError("evaluation, deployment, and search-result model names must match")
    loaded = {name: _read_json(path) for name, path in evaluations.items()}
    reference_name = next(iter(evaluations))
    reference = loaded[reference_name]
    common_fields = (
        "data_manifest_sha256",
        "scene_count",
        "frame_count",
        "score_floor",
        "intensity_mode",
        "metric_protocols",
        "postprocess_timing_order",
        "voxelization_backend",
    )
    for name, payload in loaded.items():
        mismatches = [
            field for field in common_fields if payload.get(field) != reference.get(field)
        ]
        if mismatches:
            raise ValueError(f"protocol mismatch for {name}: {mismatches}")
        if payload.get("legacy_fixed_k_used") is not False:
            raise ValueError(f"CARLA evaluation for {name} used legacy fixed K")

    models: dict[str, Any] = {}
    for name in evaluations:
        payload = loaded[name]
        deployment = _read_json(deployments[name])
        candidate = payload["models"]["candidate"]
        fp32 = payload["models"]["fp32_pytorch"]
        strict_trt = payload["models"]["baseline"]
        frontend = payload["frontend"]
        models[name] = {
            "carla": {
                "candidate": {
                    "range_all": _accuracy(payload, "candidate", "range_all"),
                    "visible_5plus": _accuracy(payload, "candidate", "visible_5plus"),
                    "forward_gpu_ms": candidate["forward_gpu_ms"],
                    "model_path_gpu_ms": candidate["model_path_gpu_ms"],
                    "postprocess_ms": candidate["postprocess_ms"],
                    "composed_end_to_end_ms": candidate["composed_end_to_end_ms"],
                },
                "unpruned_fp32_pytorch": {
                    "range_all": _accuracy(payload, "fp32_pytorch", "range_all"),
                    "visible_5plus": _accuracy(payload, "fp32_pytorch", "visible_5plus"),
                    "forward_gpu_ms": fp32["forward_gpu_ms"],
                    "composed_end_to_end_ms": fp32["composed_end_to_end_ms"],
                },
                "strict_fp32_tensorrt": {
                    "range_all": _accuracy(payload, "baseline", "range_all"),
                    "visible_5plus": _accuracy(payload, "baseline", "visible_5plus"),
                    "forward_gpu_ms": strict_trt["forward_gpu_ms"],
                    "composed_end_to_end_ms": strict_trt["composed_end_to_end_ms"],
                },
                "frontend": frontend,
                "speedup": payload["speedup"],
                "candidate_minus_fp32_map": {
                    protocol: float(_accuracy(payload, "candidate", protocol)["map"])
                    - float(_accuracy(payload, "fp32_pytorch", protocol)["map"])
                    for protocol in ("range_all", "visible_5plus")
                },
                "evaluation_source": str(evaluations[name]),
            },
            "deployment": {
                "engine_sha256": deployment["engine_sha256"],
                "inputs": deployment["onnx_audit"]["input_names"],
                "qdq_node_count": deployment["onnx_audit"]["qdq_node_count"],
                "scatter_node_count": deployment["onnx_audit"]["scatter_node_count"],
                "fixed_k_inputs": deployment["onnx_audit"]["fixed_k_inputs"],
                "plugin_required": deployment["plugin_required"],
                "runtime_max_k_dependency": deployment["onnx_audit"][
                    "runtime_max_k_dependency"
                ],
                "deployment_source": str(deployments[name]),
            },
            "dair_open_loop": _search_metrics(
                search_results[name], search_indices.get(name)
            ),
        }
    return {
        "schema_version": "heal-carla-four-model-comparison-v2",
        "protocol": {
            field: reference.get(field) for field in common_fields
        }
        | {
            "point_frontend": reference["point_frontend"],
            "legacy_fixed_k_used": False,
            "scope": "online_carla_capture_offline_same_frame_perception_replay",
            "closed_loop_planning_control": False,
            "engine_partition_contract": {
                "search_stage2": "pre_scatter_fixed_k_29696_with_pfn_and_scatter",
                "carla_deployment": "post_scatter_without_fixed_k_pfn_or_scatter",
                "separate_engine_artifacts": True,
                "accuracy_compatibility_requirement": (
                    "identical preprocessing, PFN weights, voxel geometry, feature "
                    "normalization, scatter semantics, and cut-tensor layout"
                ),
                "latency_scope_is_identical": False,
            },
        },
        "models": models,
    }


def _f(value: float) -> str:
    return f"{float(value):.6f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# HEAL 0.1-BOPS four-model CARLA comparison",
        "",
        "## Protocol",
        "",
        f"- Scenes: {report['protocol']['scene_count']}; frames: {report['protocol']['frame_count']}.",
        f"- Score floor: {report['protocol']['score_floor']}; intensity: {report['protocol']['intensity_mode']}.",
        "- CARLA was used for synchronized online sensor capture, followed by same-frame offline perception replay.",
        "- This is not a planner/controller/safety closed-loop evaluation.",
        f"- Voxelization backend: {report['protocol']['voxelization_backend']}; dynamic voxelization, PFN, and scatter are outside every CARLA TensorRT engine.",
        "",
        "## Engine partition contract",
        "",
        "The DAIR Stage-2 audit engine and the CARLA engine are two separately built artifacts derived from the same physical candidate. Stage 2 uses a pre-scatter fixed-K=29696 ABI and includes PFN/scatter so every candidate is audited under one deterministic DAIR contract. CARLA cuts the graph at the dense BEV tensor and moves dynamic voxelization, PFN, and scatter outside TensorRT.",
        "",
        "The two artifacts are accuracy-compatible only when preprocessing, PFN weights, voxel geometry, feature normalization, scatter behavior, and the cut-tensor layout are identical. Their raw forward latency scopes are not identical: DAIR Stage 2 includes PFN/scatter, while CARLA engine forward starts after scatter. A future fully deployment-aligned search should use the post-scatter boundary in Stage 2 or score frontend and post-scatter latency separately.",
        "",
        "## CARLA accuracy",
        "",
        "| Model | Candidate range mAP | FP32 range mAP | Delta | Candidate visible5 mAP | FP32 visible5 mAP | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["models"].items():
        carla = row["carla"]
        candidate = carla["candidate"]
        fp32 = carla["unpruned_fp32_pytorch"]
        delta = carla["candidate_minus_fp32_map"]
        lines.append(
            f"| {name} | {_f(candidate['range_all']['map'])} | {_f(fp32['range_all']['map'])} | {_f(delta['range_all'])} | "
            f"{_f(candidate['visible_5plus']['map'])} | {_f(fp32['visible_5plus']['map'])} | {_f(delta['visible_5plus'])} |"
        )
    lines.extend(
        [
            "",
            "## Latency",
            "",
            "All values are means in milliseconds. Composed latency is CPU point preparation + H2D + GPU voxelization + GPU PFN/scatter + forward + postprocess; it is not a wall-clock control-loop latency.",
            "",
            "| Model | CPU point prep | H2D | GPU voxel | PFN/scatter | Candidate forward | Candidate post | Candidate composed | FP32 PyTorch forward | FP32 composed | Forward speedup | Composed speedup |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in report["models"].items():
        carla = row["carla"]
        candidate = carla["candidate"]
        fp32 = carla["unpruned_fp32_pytorch"]
        frontend = carla["frontend"]
        speedup = carla["speedup"]
        lines.append(
            f"| {name} | {_f(frontend['point_preprocess_cpu_ms']['mean'])} | {_f(frontend['host_to_device_ms']['mean'])} | "
            f"{_f(frontend['voxelize_gpu_ms']['mean'])} | {_f(frontend['pfn_scatter_gpu_ms']['mean'])} | "
            f"{_f(candidate['forward_gpu_ms']['mean'])} | {_f(candidate['postprocess_ms']['mean'])} | "
            f"{_f(candidate['composed_end_to_end_ms']['mean'])} | {_f(fp32['forward_gpu_ms']['mean'])} | "
            f"{_f(fp32['composed_end_to_end_ms']['mean'])} | {_f(speedup['candidate_vs_unpruned_fp32_pytorch_forward_mean'])}x | "
            f"{_f(speedup['candidate_vs_unpruned_fp32_pytorch_composed_mean'])}x |"
        )
    lines.extend(
        [
            "",
            "## Deployment boundary",
            "",
            "| Model | Inputs | Q/DQ nodes | Scatter nodes | Fixed-K inputs | Plugin | Runtime MaxK |",
            "|---|---|---:|---:|---|---|---|",
        ]
    )
    for name, row in report["models"].items():
        deployment = row["deployment"]
        lines.append(
            f"| {name} | {', '.join(deployment['inputs'])} | {deployment['qdq_node_count']} | "
            f"{deployment['scatter_node_count']} | {deployment['fixed_k_inputs']} | "
            f"{deployment['plugin_required']} | {deployment['runtime_max_k_dependency']} |"
        )
    lines.extend(
        [
            "",
            "## DAIR open-loop provenance",
            "",
            "These values are included for search-artifact provenance, not as a direct comparison to CARLA actor ground truth.",
            "",
            "| Model | mAP | AP30 | AP50 | AP70 | Forward p50 ms | Frames |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in report["models"].items():
        dair = row["dair_open_loop"]
        lines.append(
            f"| {name} | {_f(dair['map'])} | {_f(dair['ap30'])} | {_f(dair['ap50'])} | "
            f"{_f(dair['ap70'])} | {_f(dair['forward_p50_ms'])} | {dair['evaluated_frames']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The compressed engines preserve same-frame CARLA accuracy within a few thousandths of their unpruned FP32 PyTorch references. GPU voxelization removes the former CPU hard-voxel bottleneck, while point preparation, transfer, PFN/scatter, and postprocessing remain shared costs. The postprocessing order is rotated per frame to balance first-call overhead.",
            "",
            "F-Cooper retains zero explicit Q/DQ nodes after the post-scatter cut. Its CARLA artifact therefore does not demonstrate post-scatter INT8 execution, even though it is the selected 0.1-BOPS search artifact. This boundary-specific precision realization must remain explicit in any publication claim.",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--deployment", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--search-result", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--search-index", action="append", default=[], metavar="NAME=INDEX")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = aggregate(
        _named_paths(arguments.evaluation),
        _named_paths(arguments.deployment),
        _named_paths(arguments.search_result),
        _named_ints(arguments.search_index),
    )
    json_path = arguments.output_json.expanduser().resolve()
    markdown_path = arguments.output_markdown.expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"success": True, "models": list(report["models"])}, indent=2))


if __name__ == "__main__":
    main()
