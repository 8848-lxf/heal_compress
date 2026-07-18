"""Build and measure real-activation CoBEVT Attention subgraphs."""

from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from search.model_families.lidar_cobevt.attention_dim_pruning import (
    PrunableCobevtAttention,
    materialize_attention_bottleneck,
    uniform_attention_masks,
)
from search.model_families.lidar_cobevt.attention_microbenchmark import (
    ExplicitQDQAttentionGraph,
    attention_microbenchmark_specs,
)


DEFAULT_OUTPUT = Path(
    "/data/lxf/heal_data/outputs/cobevt_attention_dim_pruning_20260718_085644"
)
DEFAULT_CHECKPOINT = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/"
    "LiDAROnly/lidar_cobevt/net_epoch_bestval_at19.pth"
)
DEFAULT_CONFIG = DEFAULT_CHECKPOINT.with_name("config.yaml")
DEFAULT_HEAL_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL")
DEFAULT_TRT_ROOT = DEFAULT_HEAL_ROOT / "prune_model/TensorRT-10.9_x86_cu118"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(row.get(key), sort_keys=True)
                        if isinstance(row.get(key), (dict, list, tuple))
                        else row.get(key)
                    )
                    for key in keys
                }
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_bundle(
    checkpoint: Path, config: Path, heal_root: Path, device: torch.device
) -> Any:
    from search.model_families.lidar_cobevt.model_capability import (
        CobevtModelCapability,
    )

    return CobevtModelCapability(checkpoint, config, heal_root).load(device=device)


def capture_real_attention_inputs(
    *, bundle: Any, config: Path, device: torch.device
) -> tuple[str, torch.Tensor, torch.Tensor, torch.nn.Module]:
    from search.integration.data_provider import build_dataset_and_loader, move_batch_to_device

    target = next(
        (name, module)
        for name, module in bundle.model.named_modules()
        if name.startswith("fusion_net")
        and module.__class__.__name__ == "Attention"
        and hasattr(module, "to_qkv")
    )
    module_name, module = target
    stock_module = copy.deepcopy(module).cpu().eval()
    captured: dict[str, torch.Tensor] = {}

    def pre_hook(
        _module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> None:
        captured["activation"] = args[0].detach().clone()
        mask = kwargs.get("mask", args[1] if len(args) > 1 else None)
        if not torch.is_tensor(mask):
            raise RuntimeError("cobevt_attention_real_mask_missing")
        captured["mask"] = mask.detach().clone()

    hook = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        _dataset, loader = build_dataset_and_loader(
            bundle.adapter,
            config,
            split="val",
            num_workers=8,
            visualize=False,
        )
        batch = move_batch_to_device(next(iter(loader)), device)
        with torch.no_grad():
            bundle.adapter.forward_for_task(bundle.model, batch)
    finally:
        hook.remove()
    if set(captured) != {"activation", "mask"}:
        raise RuntimeError("cobevt_attention_real_input_capture_incomplete")
    return module_name, captured["activation"], captured["mask"], stock_module


def _source_modules_by_spec(
    *,
    bundle: Any,
    stock_module: torch.nn.Module,
    module_name: str,
    gradients_path: Path,
) -> dict[str, PrunableCobevtAttention]:
    from search.model_families.lidar_cobevt.attention_taylor import (
        attention_masks_from_mean_gradients,
    )

    identity = uniform_attention_masks(bundle.model, d_qk=32, d_v=32)
    report = materialize_attention_bottleneck(bundle.model, identity)
    if not report.passed:
        raise RuntimeError(f"explicit_attention_baseline_invalid:{report.issues}")
    payload = torch.load(gradients_path, map_location="cpu")
    gradients = payload["gradients"]
    result: dict[str, PrunableCobevtAttention] = {}
    for spec in attention_microbenchmark_specs():
        masks, _audit = attention_masks_from_mean_gradients(
            bundle.model,
            gradients,
            d_qk=int(spec.d_qk),
            d_v=int(spec.d_v),
        )
        mask = masks[module_name]
        result[spec.spec_id] = PrunableCobevtAttention.from_stock_attention(
            stock_module,
            qk_keep_by_head=mask.qk_keep_by_head,
            vo_keep_by_head=mask.vo_keep_by_head,
        ).eval()
    return result


def _export_graph(
    *,
    source: PrunableCobevtAttention,
    activation: torch.Tensor,
    attention_mask: torch.Tensor,
    use_mask_rpe: bool,
    precision: str,
    destination: Path,
) -> tuple[torch.nn.Module, dict[str, torch.Tensor], torch.Tensor, int, int]:
    graph = ExplicitQDQAttentionGraph.from_attention(
        source, use_mask_rpe=use_mask_rpe
    ).to(activation.device)
    mask = attention_mask.to(activation.device)
    normalized = str(precision).upper()
    if normalized == "INT8":
        graph = graph.float().eval()
        value = activation.float()
        graph.calibrate(value, mask)
    elif normalized == "FP16":
        graph = graph.half().eval()
        value = activation.half()
    else:
        raise ValueError(f"unsupported_attention_microbench_precision:{precision}")
    with torch.no_grad():
        reference = graph(value, mask)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        graph,
        (value, mask),
        destination,
        input_names=("activation", "attention_mask"),
        output_names=("output",),
        opset_version=17,
        do_constant_folding=True,
    )
    import onnx

    model = onnx.load(str(destination))
    input_names = {str(row.name) for row in model.graph.input}
    inputs = {"activation": value}
    if "attention_mask" in input_names:
        inputs["attention_mask"] = mask
    node_types = [str(node.op_type) for node in model.graph.node]
    return (
        graph,
        inputs,
        reference,
        node_types.count("QuantizeLinear"),
        node_types.count("DequantizeLinear"),
    )


def _trt_environment(trt_root: Path, physical_gpu: int) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(int(physical_gpu))
    libraries = (
        trt_root / "targets/x86_64-linux-gnu/lib",
        trt_root / "lib",
        Path("/home/lixingfeng/anaconda3/envs/modelopt/lib"),
    )
    env["LD_LIBRARY_PATH"] = ":".join(
        [*(str(path) for path in libraries), env.get("LD_LIBRARY_PATH", "")]
    )
    return env


def load_tensorrt_runtime(trt_root: Path) -> None:
    library_root = Path(trt_root) / "targets/x86_64-linux-gnu/lib"
    libraries = (
        library_root / "libnvinfer.so.10",
        library_root / "libnvinfer_plugin.so.10",
        library_root / "libnvonnxparser.so.10",
    )
    for library in libraries:
        if not library.is_file():
            raise FileNotFoundError(str(library))
        ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)


def _build_engine(
    *,
    onnx_path: Path,
    engine_path: Path,
    layer_info_path: Path,
    log_path: Path,
    trt_root: Path,
    physical_gpu: int,
) -> dict[str, Any]:
    trtexec = trt_root / "targets/x86_64-linux-gnu/bin/trtexec"
    command = [
        str(trtexec),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--profilingVerbosity=detailed",
        f"--exportLayerInfo={layer_info_path}",
        "--memPoolSize=workspace:512",
        "--skipInference",
        "--noTF32",
        "--stronglyTyped",
    ]
    started = time.monotonic()
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=_trt_environment(trt_root, physical_gpu),
        check=False,
        timeout=600,
    )
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    success = (
        completed.returncode == 0
        and engine_path.is_file()
        and layer_info_path.is_file()
    )
    return {
        "build_elapsed_seconds": time.monotonic() - started,
        "build_returncode": int(completed.returncode),
        "build_success": bool(success),
        "builder_command": command,
        "engine_path": str(engine_path),
        "engine_sha256": _sha256(engine_path) if engine_path.is_file() else "",
        "failure_reason": "" if success else "trtexec_strongly_typed_build_failed",
        "layer_info_path": str(layer_info_path),
        "onnx_path": str(onnx_path),
    }


def _formats(layer: Mapping[str, Any]) -> list[str]:
    tensors = [*layer.get("Inputs", []), *layer.get("Outputs", [])]
    return sorted(
        {
            str(row.get("Format/Datatype", ""))
            for row in tensors
            if str(row.get("Format/Datatype", ""))
        }
    )


def audit_attention_engine(layer_info_path: Path) -> dict[str, Any]:
    payload = json.loads(layer_info_path.read_text(encoding="utf-8"))
    layers = list(payload.get("Layers", []))
    targets = {
        "q_precision": "q_proj",
        "k_precision": "k_proj",
        "v_precision": "v_proj",
        "qk_matmul_precision": "qk_matmul",
        "softmax_precision": "softmax",
        "av_matmul_precision": "av_matmul",
        "out_proj_precision": "out_proj",
    }
    result: dict[str, Any] = {}

    def searchable(row: Mapping[str, Any]) -> str:
        return " ".join(
            str(row.get(key, ""))
            for key in ("Name", "LayerType", "TacticName", "Metadata")
        ).lower()

    for field, token in targets.items():
        matched = [row for row in layers if token in searchable(row)]
        result[field] = sorted({value for row in matched for value in _formats(row)})
        result[f"{field}_layer_count"] = len(matched)
    result["mha_fused"] = any(
        "mha" in searchable(row) or "multiheadattention" in searchable(row)
        for row in layers
    )
    result["reformat_count"] = sum(
        str(row.get("Name", "")).startswith("Reformatting") for row in layers
    )
    result["layer_count"] = len(layers)
    result["int8_tensor_count"] = sum(
        "Int8" in value for row in layers for value in _formats(row)
    )
    return result


def _measure_engine(
    *,
    engine_path: Path,
    inputs: Mapping[str, torch.Tensor],
    reference: torch.Tensor,
    device: torch.device,
    warmup: int,
    iterations: int,
    trt_root: Path,
) -> dict[str, Any]:
    torch.cuda.set_device(device)
    load_tensorrt_runtime(trt_root)
    from tests.quant_deploy.deployment_equivalence import TensorRTEngineRunner

    runner = TensorRTEngineRunner(str(engine_path), device)
    for _ in range(int(warmup)):
        runner.run_profiled(inputs)
    timings = []
    output = None
    for _ in range(int(iterations)):
        outputs, profile = runner.run_profiled(inputs)
        output = outputs["output"].float()
        timings.append(float(profile["total_runner_ms"]))
    if output is None:
        raise RuntimeError("attention_microbenchmark_no_output")
    expected = reference.float()
    flat_output = output.reshape(-1)
    flat_expected = expected.reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(
        flat_output.unsqueeze(0), flat_expected.unsqueeze(0)
    ).item()
    difference = (output - expected).abs()
    values = np.asarray(timings, dtype=np.float64)
    return {
        "finite": bool(torch.isfinite(output).all()),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
        "cosine_similarity": float(cosine),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p99_ms": float(np.percentile(values, 99)),
        "warmup_iterations": int(warmup),
        "measured_iterations": int(iterations),
    }


def run_attention_microbenchmarks(
    *,
    output_dir: Path,
    checkpoint: Path,
    config: Path,
    heal_root: Path,
    trt_root: Path,
    device: torch.device,
    physical_gpu: int,
    only_spec: str = "",
    only_graph: str = "",
    only_precision: str = "",
) -> dict[str, Any]:
    bundle = _load_bundle(checkpoint, config, heal_root, device)
    module_name, activation, mask, stock_module = capture_real_attention_inputs(
        bundle=bundle, config=config, device=device
    )
    gradients_path = output_dir / "attention_mean_gradients.pt"
    sources = _source_modules_by_spec(
        bundle=bundle,
        stock_module=stock_module,
        module_name=module_name,
        gradients_path=gradients_path,
    )
    activation_record = {
        "activation_shape": list(activation.shape),
        "activation_dtype": str(activation.dtype),
        "activation_absmax": float(activation.abs().max().item()),
        "attention_module": module_name,
        "mask_shape": list(mask.shape),
        "mask_true_ratio": float(mask.float().mean().item()),
        "source": "real CoBEVT validation activation",
    }
    _write_json(output_dir / "attention_microbenchmark_activation.json", activation_record)
    root = output_dir / "attention_microbenchmark"
    rows: list[dict[str, Any]] = []
    for spec in attention_microbenchmark_specs():
        if only_spec and spec.spec_id != only_spec:
            continue
        for graph_kind, use_mask_rpe in (
            ("pure_attention", False),
            ("cobevt_mask_rpe", True),
        ):
            if only_graph and graph_kind != only_graph:
                continue
            for precision in ("FP16", "INT8"):
                if only_precision and precision != only_precision.upper():
                    continue
                destination = root / graph_kind / spec.spec_id / precision.lower()
                onnx_path = destination / "attention.onnx"
                engine_path = destination / "engine.plan"
                layer_info_path = destination / "engine_layer_info.json"
                log_path = destination / "trtexec.log"
                destination.mkdir(parents=True, exist_ok=True)
                try:
                    _graph, inputs, reference, q_count, dq_count = _export_graph(
                        source=sources[spec.spec_id],
                        activation=activation,
                        attention_mask=mask,
                        use_mask_rpe=use_mask_rpe,
                        precision=precision,
                        destination=onnx_path,
                    )
                    build = _build_engine(
                        onnx_path=onnx_path,
                        engine_path=engine_path,
                        layer_info_path=layer_info_path,
                        log_path=log_path,
                        trt_root=trt_root,
                        physical_gpu=physical_gpu,
                    )
                    if not build["build_success"]:
                        raise RuntimeError(str(build["failure_reason"]))
                    realization = audit_attention_engine(layer_info_path)
                    measurement = _measure_engine(
                        engine_path=engine_path,
                        inputs=inputs,
                        reference=reference,
                        device=device,
                        warmup=20,
                        iterations=100,
                        trt_root=trt_root,
                    )
                    row = {
                        "status": "ok",
                        "variant": spec.variant,
                        "spec_id": spec.spec_id,
                        "graph_kind": graph_kind,
                        "d_qk": spec.d_qk,
                        "d_v": spec.d_v,
                        "heads": 8,
                        "requested_precision": precision,
                        "qdq_count": int(q_count + dq_count),
                        "q_count": int(q_count),
                        "dq_count": int(dq_count),
                        **build,
                        **realization,
                        **measurement,
                    }
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "status": "failed",
                        "variant": spec.variant,
                        "spec_id": spec.spec_id,
                        "graph_kind": graph_kind,
                        "d_qk": spec.d_qk,
                        "d_v": spec.d_v,
                        "heads": 8,
                        "requested_precision": precision,
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    }
                _write_json(destination / "result.json", row)
                rows.append(row)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    _write_json(output_dir / "attention_microbenchmark_results.json", rows)
    _write_csv(output_dir / "int8_subgraph_build.csv", rows)
    _write_csv(output_dir / "int8_precision_realization.csv", rows)
    _write_csv(output_dir / "int8_subgraph_latency.csv", rows)
    return {
        "result_count": len(rows),
        "successful_count": sum(row["status"] == "ok" for row in rows),
        "int8_successful_count": sum(
            row["status"] == "ok" and row["requested_precision"] == "INT8"
            for row in rows
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--heal-root", default=str(DEFAULT_HEAL_ROOT))
    parser.add_argument("--trt-root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--physical-gpu", type=int, default=2)
    parser.add_argument("--only-spec", default="")
    parser.add_argument("--only-graph", choices=("", "pure_attention", "cobevt_mask_rpe"), default="")
    parser.add_argument("--only-precision", choices=("", "FP16", "INT8"), default="")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    result = run_attention_microbenchmarks(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        checkpoint=Path(args.checkpoint).expanduser().resolve(),
        config=Path(args.config).expanduser().resolve(),
        heal_root=Path(args.heal_root).expanduser().resolve(),
        trt_root=Path(args.trt_root).expanduser().resolve(),
        device=torch.device(args.device),
        physical_gpu=int(args.physical_gpu),
        only_spec=str(args.only_spec),
        only_graph=str(args.only_graph),
        only_precision=str(args.only_precision),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["successful_count"] == result["result_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
