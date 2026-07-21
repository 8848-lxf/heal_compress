"""Export, build, and evaluate one strict-FP32 HEAL LiDAR baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import torch
from torch.utils.data import DataLoader


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _move(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value


def _strict_model(config: Path, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils

    hypes = yaml_utils.load_yaml(str(config))
    model = train_utils.create_model(hypes)
    raw = torch.load(checkpoint, map_location="cpu")
    state = raw.get("model", raw) if isinstance(raw, dict) else raw
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _parity(
    reference: dict[str, torch.Tensor],
    actual: tuple[torch.Tensor, ...],
    names: tuple[str, ...],
) -> dict[str, Any]:
    rows = {}
    for name, observed in zip(names, actual):
        expected = reference[name]
        difference = (expected.float() - observed.float()).abs()
        max_abs = float(difference.max().item())
        mean_abs = float(difference.mean().item())
        rows[name] = {
            "expected_shape": list(expected.shape),
            "actual_shape": list(observed.shape),
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "allclose": bool(
                expected.shape == observed.shape
                and torch.allclose(expected.float(), observed.float(), atol=2.0e-3, rtol=1.0e-4)
                and max_abs <= 2.0e-3
                and mean_abs <= 1.0e-5
            ),
        }
    return {"passed": all(bool(row["allclose"]) for row in rows.values()), "outputs": rows}


def _mapping(origin: Any) -> Any:
    from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult

    entries = [
        CanonicalPrecisionEntry(
            module_path=str(row.module_path),
            canonical_node_name=str(row.canonical_node_name),
            precision_group=f"strict_fp32::{row.module_path}",
            requested_precision="fp32",
            realized_request_precision="fp32",
            realized_output_precision="fp32",
            original_node_name=str(row.original_node_name),
            weight_initializer=str(row.weight_initializer),
            onnx_op_type=str(row.onnx_op_type),
            call_index=int(row.call_index),
        )
        for row in origin.entries
    ]
    return CanonicalPrecisionMappingResult(
        entries=entries,
        profile_id="strict_fp32",
        origin_map_hash=str(origin.origin_map_hash),
        policy_version="heal-dair-lidar-strict-fp32-strongly-typed-v1",
    )


def _build_wrapper(
    model_name: str,
    model: torch.nn.Module,
    ego: dict[str, Any],
    fixed_k: int,
) -> tuple[torch.nn.Module, dict[str, torch.Tensor], tuple[str, ...], str, dict[str, Any]]:
    outputs = ("cls_preds", "reg_preds", "dir_preds")
    if model_name == "lidar_pyramid":
        from quantization.config import OnnxExportConfig
        from quantization.export import build_heal_signal_maxk_export_module, prepare_signal_maxk_inputs

        config = OnnxExportConfig(
            fixed_k=int(fixed_k),
            min_agents=1,
            opt_agents=2,
            max_agents=2,
        )
        return (
            build_heal_signal_maxk_export_module(model, config=config).eval(),
            prepare_signal_maxk_inputs(ego, config=config),
            outputs,
            "heal_lidar_pyramid_signal_maxk",
            {"dynamic_agent_dimension": True},
        )
    if model_name == "lidar_v2xvit":
        from search.model_family.export import (
            HealV2XViTExportPolicy,
            build_heal_v2xvit_export_module,
            prepare_v2xvit_fixed_k_inputs,
        )

        policy = HealV2XViTExportPolicy(fixed_k=int(fixed_k), max_agents=2)
        return (
            build_heal_v2xvit_export_module(model, policy=policy).eval(),
            prepare_v2xvit_fixed_k_inputs(ego, policy=policy),
            outputs,
            "heal_v2xvit_fixed_k",
            {"dynamic_agent_dimension": False},
        )
    from search.model_family.export import (
        HealLidarBaselineExportPolicy,
        build_heal_lidar_baseline_export_module,
        prepare_heal_lidar_baseline_inputs,
    )

    policy = HealLidarBaselineExportPolicy(fixed_k=int(fixed_k), max_agents=2)
    return (
        build_heal_lidar_baseline_export_module(model, policy=policy).eval(),
        prepare_heal_lidar_baseline_inputs(ego, policy=policy),
        outputs,
        "heal_lidar_baseline_fixed_k",
        {"dynamic_agent_dimension": False},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    output = Path(request["output_dir"])
    result_path = output / "deployment_result.json"
    try:
        if os.environ.get("CONDA_DEFAULT_ENV") != "univ2x-opt":
            raise RuntimeError("fp32_deployment_worker_requires_univ2x_opt")
        repo_root = Path(__file__).resolve().parents[2]
        for path in (
            "/home/lixingfeng/UniAD_examine",
            "/home/lixingfeng/UniAD_examine/HEAL",
            str(repo_root),
        ):
            if path not in sys.path:
                sys.path.insert(0, path)
        from opencood.data_utils.datasets import build_dataset
        from opencood.hypes_yaml import yaml_utils
        from quantization.config import TensorRTBuildConfig
        from quantization.export.origin_mapping import apply_canonical_node_names, build_onnx_origin_map
        from quantization.export.signal_maxk import capture_weighted_module_calls
        from search.baselines.original_engines import validate_baseline_layer_precisions
        from search.model_family.deployment import build_physical_structure_snapshot_v2
        from search.stage2.trt_modelopt import build_engine_modelopt

        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        config_path = Path(request["config_path"]).resolve()
        checkpoint = Path(request["checkpoint_path"]).resolve()
        model_name = str(request["model_name"])
        fixed_k = int(request["fixed_k"])
        model = _strict_model(config_path, checkpoint, device)
        hypes = yaml_utils.load_yaml(str(config_path))
        heal_root = Path(request["heal_root"]).resolve()
        for key in ("data_dir", "root_dir", "validate_dir", "test_dir"):
            value = hypes.get(key)
            if isinstance(value, str) and not Path(value).is_absolute():
                hypes[key] = str(heal_root / value)
        dataset = build_dataset(hypes, visualize=True, train=False)
        raw_batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_batch_test)))
        batch = _move(raw_batch, device)
        ego = batch["ego"]
        wrapper, prepared, output_names, input_contract, export_contract = _build_wrapper(
            model_name, model, ego, fixed_k
        )
        input_names = tuple(prepared)
        tensors = tuple(prepared[name] for name in input_names)
        with torch.inference_mode():
            reference = model(ego)
            actual = wrapper(*tensors)
        parity = _parity(reference, actual, output_names)
        _write_json(output / "real_wrapper_parity.json", parity)
        if not parity["passed"]:
            raise RuntimeError(f"strict_fp32_wrapper_parity_failed:{parity}")

        onnx_path = output / "strict_fp32.onnx"
        dynamic_axes = None
        if bool(export_contract["dynamic_agent_dimension"]):
            dynamic_axes = {"pairwise_t_matrix": {1: "num_agents", 2: "num_agents"}}
        with capture_weighted_module_calls(wrapper) as calls:
            torch.onnx.export(
                wrapper,
                tensors,
                str(onnx_path),
                export_params=True,
                opset_version=int(request.get("opset", 17)),
                do_constant_folding=True,
                input_names=list(input_names),
                output_names=list(output_names),
                dynamic_axes=dynamic_axes,
                custom_opsets={"trt": 1},
            )
        import onnx

        onnx.checker.check_model(onnx.load(str(onnx_path), load_external_data=False))
        origin = build_onnx_origin_map(onnx_path, calls)
        apply_canonical_node_names(onnx_path, origin, output_path=onnx_path, allow_custom_ops=True)
        mapping = _mapping(origin)
        snapshot = build_physical_structure_snapshot_v2(model, model_family=model_name)
        _write_json(output / "canonical_origin_map.json", origin.to_dict())
        _write_json(output / "canonical_precision_mapping.json", mapping.to_dict())
        _write_json(output / "physical_structure_snapshot_v2.json", snapshot)
        _write_json(
            output / "onnx_audit.json",
            {
                "checker_passed": True,
                "onnx_path": str(onnx_path),
                "onnx_sha256": _sha256(onnx_path),
                "input_names": list(input_names),
                "output_names": list(output_names),
                "weighted_call_count": len(calls),
                "canonical_weighted_count": len(origin.entries),
                "functional_compute_group_count": len(origin.functional_compute_groups),
                "fixed_k": fixed_k,
                "input_contract": input_contract,
            },
        )

        del actual, reference, prepared, tensors, wrapper, ego, batch, raw_batch, model
        torch.cuda.empty_cache()
        trt_root = Path(request["tensorrt_root"]).resolve()
        trtexec = trt_root / "bin/trtexec"
        if not trtexec.is_file():
            trtexec = trt_root / "targets/x86_64-linux-gnu/bin/trtexec"
        shape_profiles = {}
        if bool(export_contract["dynamic_agent_dimension"]):
            shape_profiles = {
                "pairwise_t_matrix": {
                    "min": (1, 1, 1, 4, 4),
                    "opt": (1, 2, 2, 4, 4),
                    "max": (1, 2, 2, 4, 4),
                }
            }
        build_config = TensorRTBuildConfig(
            trtexec_path=trtexec,
            plugin_path=Path(request["plugin_path"]),
            workspace_mib=int(request.get("workspace_mib", 4096)),
            shape_profiles=shape_profiles,
            timeout_seconds=int(request.get("build_timeout_seconds", 3600)),
            enable_fp16=False,
            enable_int8=False,
            no_tf32=True,
            skip_inference=True,
            export_layer_info=True,
            strongly_typed=True,
            policy_version="heal-dair-lidar-strict-fp32-strongly-typed-no-tf32-v1",
        )
        engine_path = output / "strict_fp32.plan"
        build = build_engine_modelopt(
            qdq_onnx=onnx_path,
            engine_path=engine_path,
            precision_mapping=mapping,
            build_config=build_config,
            physical_snapshot=snapshot,
            output_dir=output / "engine_build",
            tensorrt_root=trt_root,
            conda_env="modelopt",
            gpu_id=int(request["physical_gpu"]),
        )
        _write_json(output / "engine_build_acceptance.json", build)
        layer_info = output / "engine_build/engine_layer_info.json"
        precision_audit = (
            validate_baseline_layer_precisions("strict_fp32", layer_info)
            if layer_info.is_file()
            else {"passed": False, "status": "layer_info_missing"}
        )
        _write_json(output / "strict_fp32_precision_audit.json", precision_audit)
        if build.get("status") != "ok":
            raise RuntimeError(
                f"strict_fp32_engine_build_or_canonical_validation_failed:{build.get('status')}:"
                f"{build.get('failure_reason', '')}"
            )
        if not precision_audit.get("passed", False):
            raise RuntimeError(f"strict_fp32_engine_precision_failed:{precision_audit}")
        if not engine_path.is_file() or engine_path.stat().st_size <= 0:
            raise RuntimeError("strict_fp32_engine_missing")

        eval_frames = int(request.get("num_frames", 1789))
        warmup_frames = int(request.get("warmup_frames", 200))
        if model_name == "lidar_pyramid":
            from search.integration.evaluation_provider import evaluate_engine_modelopt

            evaluation = evaluate_engine_modelopt(
                engine_path=engine_path,
                checkpoint=checkpoint,
                model_config=config_path,
                heal_root=request["heal_root"],
                device=f"cuda:{int(request['physical_gpu'])}",
                output_dir=output / "evaluation",
                tensorrt_root=trt_root,
                plugin_path=request["plugin_path"],
                num_frames=eval_frames,
                warmup_frames=warmup_frames,
                fixed_k=fixed_k,
                latency_rounds=int(request.get("latency_rounds", 1)),
                eval_manifest_path=request["eval_manifest_path"],
                dataloader_num_workers=int(request.get("dataloader_num_workers", 8)),
            )
        else:
            from search.model_family.evaluation import evaluate_v2xvit_engine_modelopt

            evaluation = evaluate_v2xvit_engine_modelopt(
                engine_path=engine_path,
                model_config=config_path,
                heal_root=request["heal_root"],
                output_dir=output / "evaluation",
                tensorrt_root=trt_root,
                plugin_path=request["plugin_path"],
                eval_manifest_path=request["eval_manifest_path"],
                physical_gpu_id=int(request["physical_gpu"]),
                fixed_k=fixed_k,
                max_agents=2,
                num_frames=eval_frames,
                warmup_frames=warmup_frames,
                latency_rounds=int(request.get("latency_rounds", 1)),
                dataloader_num_workers=int(request.get("dataloader_num_workers", 8)),
                input_contract=input_contract,
            )
        _write_json(output / "evaluation_acceptance.json", evaluation)
        if evaluation.get("status") != "ok":
            raise RuntimeError(f"strict_fp32_engine_evaluation_failed:{evaluation.get('failure_reason', '')}")
        result = {
            "status": "ok",
            "model_name": model_name,
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint),
            "fixed_k": fixed_k,
            "input_contract": input_contract,
            "wrapper_parity_passed": True,
            "onnx_checker_passed": True,
            "strongly_typed": True,
            "no_tf32": True,
            "precision_audit": precision_audit,
            "engine_sha256": _sha256(engine_path),
            "engine_size": engine_path.stat().st_size,
            "evaluation": evaluation,
        }
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "failed",
            "model_name": str(request.get("model_name", "")),
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    _write_json(result_path, result)
    print(json.dumps({"status": result["status"], "model": result["model_name"]}, sort_keys=True))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
