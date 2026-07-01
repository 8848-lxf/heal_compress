from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import FIXED_K, LatencyLUTKey


class SubgraphExportError(RuntimeError):
    pass


def _scalar(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    if isinstance(value, (list, tuple)):
        return int(value[0]) if value else int(default)
    return int(value)


def _write_default_scale_cache() -> None:
    cache = Path("outputs/latency_lut/activation_scale_cache.json")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        cache.write_text(
            json.dumps(
                {
                    "scale_source": "default_synthetic",
                    "activation_scale": 0.03125,
                    "weight_scale": 0.02,
                    "note": "Synthetic scales are used only to benchmark Q/DQ subgraphs before D_cal activation ranges are collected.",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def _np_rng(key: LatencyLUTKey):
    import numpy as np

    seed = int(key.stable_hash()[:8], 16)
    return np.random.default_rng(seed)


def _qdq_pair(
    nodes: list[Any],
    initializers: list[Any],
    tensor_name: str,
    *,
    scale: float = 0.03125,
    prefix: str,
    axis: int | None = None,
    channels: int | None = None,
) -> str:
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    if channels is not None and channels > 1 and axis is not None:
        scale_name = f"{prefix}_scale"
        zero_name = f"{prefix}_zero"
        initializers.append(numpy_helper.from_array(np.full((channels,), float(scale), dtype=np.float32), scale_name))
        initializers.append(numpy_helper.from_array(np.zeros((channels,), dtype=np.int8), zero_name))
        attrs = {"axis": int(axis)}
    else:
        scale_name = f"{prefix}_scale"
        zero_name = f"{prefix}_zero"
        initializers.append(helper.make_tensor(scale_name, TensorProto.FLOAT, [], [float(scale)]))
        initializers.append(helper.make_tensor(zero_name, TensorProto.INT8, [], [0]))
        attrs = {}
    q_name = f"{prefix}_q"
    dq_name = f"{prefix}_dq"
    nodes.append(helper.make_node("QuantizeLinear", [tensor_name, scale_name, zero_name], [q_name], name=f"{prefix}_QuantizeLinear", **attrs))
    nodes.append(helper.make_node("DequantizeLinear", [q_name, scale_name, zero_name], [dq_name], name=f"{prefix}_DequantizeLinear", **attrs))
    return dq_name


def _weight_initializer(initializers: list[Any], key: LatencyLUTKey, name: str, shape: tuple[int, ...], *, per_channel_axis: int | None = 0) -> str:
    import numpy as np
    from onnx import numpy_helper

    rng = _np_rng(key)
    weight = rng.normal(0.0, 0.05, size=shape).astype(np.float32)
    initializers.append(numpy_helper.from_array(weight, name))
    nodes: list[Any] = []
    qdq_name = _qdq_pair(
        nodes,
        initializers,
        name,
        scale=0.02,
        prefix=f"{name}_qdq",
        axis=per_channel_axis,
        channels=shape[0] if per_channel_axis == 0 and shape else None,
    )
    return qdq_name, nodes


def _conv_qdq(
    nodes: list[Any],
    initializers: list[Any],
    key: LatencyLUTKey,
    input_name: str,
    *,
    prefix: str,
    c_in: int,
    c_out: int,
    kernel: int,
    stride: int = 1,
    padding: int = 0,
    relu: bool = True,
) -> str:
    from onnx import helper

    activation = _qdq_pair(nodes, initializers, input_name, prefix=f"{prefix}_act")
    weight_name, weight_nodes = _weight_initializer(initializers, key, f"{prefix}_weight", (c_out, c_in, kernel, kernel), per_channel_axis=0)
    nodes.extend(weight_nodes)
    conv_out = f"{prefix}_conv"
    nodes.append(
        helper.make_node(
            "Conv",
            [activation, weight_name],
            [conv_out],
            name=f"{prefix}_Conv",
            kernel_shape=[kernel, kernel],
            strides=[stride, stride],
            pads=[padding, padding, padding, padding],
        )
    )
    out = _qdq_pair(nodes, initializers, conv_out, prefix=f"{prefix}_out")
    if relu:
        relu_out = f"{prefix}_relu"
        nodes.append(helper.make_node("Relu", [out], [relu_out], name=f"{prefix}_Relu"))
        out = _qdq_pair(nodes, initializers, relu_out, prefix=f"{prefix}_relu_out")
    return out


def _export_qdq_subgraph(key: LatencyLUTKey, onnx_path: Path) -> dict[str, Any]:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    _write_default_scale_cache()
    nodes: list[Any] = []
    initializers: list[Any] = []
    inputs: list[Any] = []
    outputs: list[Any] = []
    c_in = int(key.C_in or key.C_out or 1)
    c_out = int(key.C_out or key.C_in or 1)
    c_mid = int(key.C_mid or c_out)
    h, w = int(key.H or 16), int(key.W or 16)
    kernel = _scalar(key.kernel_size, 1)
    stride = _scalar(key.stride, 1)
    padding = _scalar(key.padding, kernel // 2 if kernel > 1 else 0)

    metadata = {
        "scale_source": "default_synthetic",
        "activation_quantization": "per-tensor symmetric int8 zero_point=0",
        "weight_quantization": "per-channel symmetric int8 zero_point=0",
    }

    if key.block_type == "conv_block":
        inputs.append(helper.make_tensor_value_info("input", TensorProto.FLOAT, [int(key.batch_size), c_in, h, w]))
        out = _conv_qdq(nodes, initializers, key, "input", prefix="conv_block", c_in=c_in, c_out=c_out, kernel=kernel, stride=stride, padding=padding)
        outputs.append(helper.make_tensor_value_info("output", TensorProto.FLOAT, [int(key.batch_size), c_out, max(1, h // max(1, stride)), max(1, w // max(1, stride))]))
    elif key.block_type == "compression_1x1":
        inputs.append(helper.make_tensor_value_info("input", TensorProto.FLOAT, [int(key.batch_size), c_in, h, w]))
        out = _conv_qdq(nodes, initializers, key, "input", prefix="compression", c_in=c_in, c_out=c_out, kernel=1, stride=stride, padding=0)
        outputs.append(helper.make_tensor_value_info("output", TensorProto.FLOAT, [int(key.batch_size), c_out, max(1, h // max(1, stride)), max(1, w // max(1, stride))]))
    elif key.block_type == "head_branch":
        inputs.append(helper.make_tensor_value_info("input", TensorProto.FLOAT, [int(key.batch_size), c_in, h, w]))
        out = _conv_qdq(nodes, initializers, key, "input", prefix="head", c_in=c_in, c_out=c_out, kernel=1, stride=1, padding=0, relu=False)
        outputs.append(helper.make_tensor_value_info("output", TensorProto.FLOAT, [int(key.batch_size), c_out, h, w]))
    elif key.block_type == "residual_block":
        inputs.append(helper.make_tensor_value_info("input", TensorProto.FLOAT, [int(key.batch_size), c_in, h, w]))
        conv1 = _conv_qdq(nodes, initializers, key, "input", prefix="res_conv1", c_in=c_in, c_out=c_mid, kernel=kernel, stride=stride, padding=padding)
        conv2 = _conv_qdq(nodes, initializers, key, conv1, prefix="res_conv2", c_in=c_mid, c_out=c_out, kernel=kernel, stride=1, padding=padding, relu=False)
        identity = "input"
        if c_in != c_out or stride != 1:
            identity = _conv_qdq(nodes, initializers, key, "input", prefix="res_proj", c_in=c_in, c_out=c_out, kernel=1, stride=stride, padding=0, relu=False)
        identity = _qdq_pair(nodes, initializers, identity, prefix="res_identity")
        add_out = "res_add"
        nodes.append(helper.make_node("Add", [conv2, identity], [add_out], name="Residual_Add"))
        add_qdq = _qdq_pair(nodes, initializers, add_out, prefix="res_add_out")
        relu_out = "res_relu"
        nodes.append(helper.make_node("Relu", [add_qdq], [relu_out], name="Residual_Relu"))
        out = _qdq_pair(nodes, initializers, relu_out, prefix="res_output")
        outputs.append(helper.make_tensor_value_info("output", TensorProto.FLOAT, [int(key.batch_size), c_out, max(1, h // max(1, stride)), max(1, w // max(1, stride))]))
    elif key.block_type == "fusion_block":
        c_ego = int(key.metadata.get("C_ego") or key.C_mid or key.C_out or max(1, c_in // 2))
        c_infra = int(key.metadata.get("C_infra") or max(1, c_in - c_ego))
        c_fused = int(key.metadata.get("C_fused") or key.C_out or key.C_mid or max(c_ego, c_infra))
        inputs.append(helper.make_tensor_value_info("ego", TensorProto.FLOAT, [int(key.batch_size), c_ego, h, w]))
        inputs.append(helper.make_tensor_value_info("infrastructure", TensorProto.FLOAT, [int(key.batch_size), c_infra, h, w]))
        ego = _qdq_pair(nodes, initializers, "ego", prefix="fusion_ego")
        infra = _qdq_pair(nodes, initializers, "infrastructure", prefix="fusion_infra")
        concat = "fusion_concat"
        nodes.append(helper.make_node("Concat", [ego, infra], [concat], name="Fusion_Concat", axis=1))
        concat_qdq = _qdq_pair(nodes, initializers, concat, prefix="fusion_concat_out")
        compressed = _conv_qdq(nodes, initializers, key, concat_qdq, prefix="fusion_compress", c_in=c_ego + c_infra, c_out=c_fused, kernel=1, padding=0)
        ego_proj = _conv_qdq(nodes, initializers, key, ego, prefix="fusion_ego_proj", c_in=c_ego, c_out=c_fused, kernel=1, padding=0, relu=False)
        add = "fusion_add"
        nodes.append(helper.make_node("Add", [compressed, ego_proj], [add], name="Fusion_Add"))
        add_qdq = _qdq_pair(nodes, initializers, add, prefix="fusion_add_out")
        out = _conv_qdq(nodes, initializers, key, add_qdq, prefix="fusion_conv", c_in=c_fused, c_out=c_fused, kernel=3, padding=1)
        outputs.append(helper.make_tensor_value_info("output", TensorProto.FLOAT, [int(key.batch_size), c_fused, h, w]))
    elif key.block_type == "pfn_block":
        point_feature_dim = int(key.metadata.get("point_feature_dim") or key.C_in or key.C_out or 1)
        inputs.append(helper.make_tensor_value_info("points", TensorProto.FLOAT, [int(key.batch_size), int(key.fixed_K), point_feature_dim]))
        points = _qdq_pair(nodes, initializers, "points", prefix="pfn_points")
        rng = _np_rng(key)
        weight = rng.normal(0.0, 0.05, size=(point_feature_dim, c_out)).astype(np.float32)
        initializers.append(numpy_helper.from_array(weight, "pfn_weight"))
        weight_dq = _qdq_pair(nodes, initializers, "pfn_weight", scale=0.02, prefix="pfn_weight_qdq")
        matmul = "pfn_matmul"
        nodes.append(helper.make_node("MatMul", [points, weight_dq], [matmul], name="PFN_MatMul"))
        matmul_qdq = _qdq_pair(nodes, initializers, matmul, prefix="pfn_matmul_out")
        relu = "pfn_relu"
        nodes.append(helper.make_node("Relu", [matmul_qdq], [relu], name="PFN_Relu"))
        out = "pfn_reduce_max"
        nodes.append(helper.make_node("ReduceMax", [relu], [out], name="PFN_ReduceMax", axes=[1], keepdims=0))
        outputs.append(helper.make_tensor_value_info("output", TensorProto.FLOAT, [int(key.batch_size), c_out]))
        metadata["weight_quantization"] = "per-tensor symmetric int8 zero_point=0"
    else:
        raise SubgraphExportError(f"Q/DQ subgraph export is not implemented for block_type={key.block_type}")

    nodes.append(helper.make_node("Identity", [out], ["output"], name="Output_Identity"))
    graph = helper.make_graph(nodes, f"{key.block_type}_qdq", inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)], producer_name="heal_compress_latency_lut")
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))
    return metadata


def _export_boundary_subgraph(key: LatencyLUTKey, onnx_path: Path) -> dict[str, Any]:
    import onnx
    from onnx import TensorProto, helper

    nodes: list[Any] = []
    initializers: list[Any] = []
    c = int(key.C_in or key.C_out or 1)
    h, w = int(key.H or 16), int(key.W or 16)
    src = key.src_precision or key.tensor_dtype_before or "TRT_FP16"
    dst = key.dst_precision or key.tensor_dtype_after or "TRT_FP16"
    uses_qdq = "INT8" in src or "INT8" in dst
    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, [int(key.batch_size), c, h, w])]
    current = "input"
    if "FP16" in src:
        nodes.append(helper.make_node("Cast", [current], ["src_fp16"], name="Boundary_Source_FP16", to=TensorProto.FLOAT16))
        nodes.append(helper.make_node("Cast", ["src_fp16"], ["src_fp32"], name="Boundary_Source_Back_To_FP32", to=TensorProto.FLOAT))
        current = "src_fp32"
    if uses_qdq:
        current = _qdq_pair(nodes, initializers, current, prefix="boundary_int8")
    if "FP16" in dst:
        nodes.append(helper.make_node("Cast", [current], ["dst_fp16"], name="Boundary_Destination_FP16", to=TensorProto.FLOAT16))
        nodes.append(helper.make_node("Cast", ["dst_fp16"], ["dst_fp32"], name="Boundary_Destination_Back_To_FP32", to=TensorProto.FLOAT))
        current = "dst_fp32"
    nodes.append(helper.make_node("Identity", [current], ["output"], name="Boundary_Output"))
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, [int(key.batch_size), c, h, w])]
    graph = helper.make_graph(nodes, "precision_boundary", inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)], producer_name="heal_compress_latency_lut")
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))
    return {
        "boundary_type": f"{src}_to_{dst}",
        "uses_qdq": uses_qdq,
        "src_precision": src,
        "dst_precision": dst,
        "scale_source": "default_synthetic" if uses_qdq else None,
    }


def _export_pointpillar_scatter_plugin_subgraph(key: LatencyLUTKey, onnx_path: Path) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        raise SubgraphExportError(f"torch is required for PointPillarScatterTRT ONNX export: {exc}") from exc

    class PointPillarScatterTRTFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, pillar_features, voxel_coords, valid_voxel_mask, num_agents: int, height: int, width: int):
            return pillar_features.new_zeros((int(num_agents), int(pillar_features.shape[1]), int(height), int(width)))

        @staticmethod
        def symbolic(g, pillar_features, voxel_coords, valid_voxel_mask, num_agents: int, height: int, width: int):
            return g.op(
                "PointPillarScatterTRT",
                pillar_features,
                voxel_coords,
                valid_voxel_mask,
                num_agents_i=int(num_agents),
                height_i=int(height),
                width_i=int(width),
            )

    class PointPillarScatterTRTModule(torch.nn.Module):
        def __init__(self, num_agents: int, height: int, width: int) -> None:
            super().__init__()
            self.num_agents = int(num_agents)
            self.height = int(height)
            self.width = int(width)

        def forward(self, pillar_features, voxel_coords, valid_voxel_mask):
            return PointPillarScatterTRTFn.apply(pillar_features, voxel_coords, valid_voxel_mask, self.num_agents, self.height, self.width)

    check_onnx_proto = getattr(torch._C, "_check_onnx_proto", None)
    c = int(key.C_in or key.C_out or key.metadata.get("C_bev") or 64)
    k = int(key.fixed_K or FIXED_K)
    h, w = int(key.H or key.metadata.get("H_bev") or 200), int(key.W or key.metadata.get("W_bev") or 704)
    num_agents = int(key.metadata.get("num_agents") or 2)
    dtype = torch.float16 if key.precision_profile == "TRT_FP16" else torch.float32
    module = PointPillarScatterTRTModule(num_agents, h, w).eval()
    pillar = torch.zeros((k, c), dtype=dtype)
    coords = torch.zeros((k, 4), dtype=torch.int32)
    mask = torch.zeros((k,), dtype=dtype)
    if check_onnx_proto is not None:
        torch._C._check_onnx_proto = lambda proto: None
    try:
        torch.onnx.export(
            module,
            (pillar, coords, mask),
            str(onnx_path),
            opset_version=17,
            input_names=["pillar_features", "voxel_coords", "valid_voxel_mask"],
            output_names=["spatial_features"],
            dynamic_axes={},
            do_constant_folding=True,
        )
    finally:
        if check_onnx_proto is not None:
            torch._C._check_onnx_proto = check_onnx_proto
    return {
        "plugin_name": "PointPillarScatterTRT",
        "plugin_version": key.plugin_version or "1",
        "H_bev": h,
        "W_bev": w,
        "C_bev": c,
        "fixed_K": k,
        "num_agents": num_agents,
    }


def export_minimal_subgraph(key: LatencyLUTKey, output_dir: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Export or describe a minimal ONNX subgraph for a LUT key.

    Dry-run writes a JSON spec only. Real ONNX export is implemented for simple
    conv-like blocks when torch/onnx are available. Plugin benchmarking remains
    an explicit placeholder because it needs the built PointPillarScatterTRT
    library and plugin wrapper.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec_path = out / f"{key.stable_hash()}.subgraph.json"
    onnx_path = out / f"{key.stable_hash()}.onnx"
    spec = {
        "key_hash": key.stable_hash(),
        "key": key.to_dict(),
        "onnx_path": str(onnx_path),
        "implemented": key.block_type in {"conv_block", "residual_block", "compression_1x1", "head_branch", "fusion_block", "pfn_block", "precision_boundary", "plugin"},
        "dry_run": bool(dry_run),
    }
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    if dry_run:
        return {**spec, "status": "dry_run"}
    if key.block_type == "plugin":
        if key.precision_profile == "TRT_INT8_QDQ":
            raise SubgraphExportError("PointPillarScatterTRT INT8 IO benchmark is not supported by the current plugin")
        metadata = _export_pointpillar_scatter_plugin_subgraph(key, onnx_path)
        return {**spec, "status": "exported", "onnx_path": str(onnx_path), "metadata": metadata}
    if key.precision_profile == "TRT_INT8_QDQ":
        metadata = _export_qdq_subgraph(key, onnx_path)
        return {**spec, "status": "exported", "onnx_path": str(onnx_path), "metadata": metadata}
    if key.block_type == "precision_boundary":
        metadata = _export_boundary_subgraph(key, onnx_path)
        return {**spec, "status": "exported", "onnx_path": str(onnx_path), "metadata": metadata}
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise SubgraphExportError(f"torch is required for ONNX subgraph export: {exc}") from exc

    if key.block_type not in {"conv_block", "residual_block", "compression_1x1", "head_branch", "fusion_block", "pfn_block"}:
        raise SubgraphExportError(f"subgraph export is not implemented for block_type={key.block_type}")

    c_in = int(key.C_in or key.C_out or 1)
    c_out = int(key.C_out or key.C_in or 1)
    h, w = int(key.H or 16), int(key.W or 16)
    kernel = _scalar(key.kernel_size, 1)
    stride = _scalar(key.stride, 1)
    padding = _scalar(key.padding, kernel // 2 if kernel > 1 else 0)
    dilation = _scalar(key.dilation, 1)
    groups = int(key.groups or 1)

    class ConvBNReLU(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, k: int, s: int, p: int, d: int, g: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, k: int, s: int, p: int, d: int) -> None:
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, k, stride=s, padding=p, dilation=d, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=False),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(hidden_channels, out_channels, k, stride=1, padding=p, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels or s != 1:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=s, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.proj = nn.Identity()
            self.relu = nn.ReLU(inplace=False)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            identity = self.proj(x)
            out = self.conv2(self.conv1(x))
            return self.relu(out + identity)

    class HeadBranch(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=True)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.conv(x)

    class FusionSurrogate(nn.Module):
        def __init__(self, c_ego: int, c_infra: int, c_fused: int) -> None:
            super().__init__()
            self.compress = nn.Sequential(
                nn.Conv2d(c_ego + c_infra, c_fused, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(c_fused),
                nn.ReLU(inplace=False),
            )
            self.ego_proj = nn.Sequential(
                nn.Conv2d(c_ego, c_fused, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(c_fused),
            )
            self.fuse = nn.Sequential(
                nn.Conv2d(c_fused, c_fused, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(c_fused),
                nn.ReLU(inplace=False),
            )

        def forward(self, ego: "torch.Tensor", infrastructure: "torch.Tensor") -> "torch.Tensor":
            merged = torch.cat([ego, infrastructure], dim=1)
            compressed = self.compress(merged)
            fused = compressed + self.ego_proj(ego)
            return self.fuse(fused)

    class PFNSurrogate(nn.Module):
        def __init__(self, point_feature_dim: int, out_channels: int) -> None:
            super().__init__()
            self.linear = nn.Linear(point_feature_dim, out_channels, bias=False)
            self.bn = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=False)

        def forward(self, points: "torch.Tensor") -> "torch.Tensor":
            features = self.linear(points)
            features = features.transpose(1, 2)
            features = self.bn(features)
            features = features.transpose(1, 2)
            features = self.relu(features)
            return torch.max(features, dim=1).values

    if key.block_type == "residual_block":
        module = ResidualBlock(c_in, int(key.C_mid or c_out), c_out, kernel, stride, padding, dilation).eval()
    elif key.block_type == "compression_1x1":
        module = ConvBNReLU(c_in, c_out, 1, stride, 0, 1, groups).eval()
    elif key.block_type == "head_branch":
        module = HeadBranch(c_in, c_out).eval()
    elif key.block_type == "fusion_block":
        c_ego = int(key.metadata.get("C_ego") or key.C_mid or key.C_out or max(1, c_in // 2))
        c_infra = int(key.metadata.get("C_infra") or max(1, c_in - c_ego))
        c_fused = int(key.metadata.get("C_fused") or key.C_out or key.C_mid or max(c_ego, c_infra))
        module = FusionSurrogate(c_ego, c_infra, c_fused).eval()
        ego = torch.randn(int(key.batch_size), c_ego, h, w)
        infrastructure = torch.randn(int(key.batch_size), c_infra, h, w)
        torch.onnx.export(
            module,
            (ego, infrastructure),
            str(onnx_path),
            input_names=["ego", "infrastructure"],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
        )
        return {**spec, "status": "exported", "onnx_path": str(onnx_path)}
    elif key.block_type == "pfn_block":
        point_feature_dim = int(key.metadata.get("point_feature_dim") or key.C_in or key.C_out or 1)
        module = PFNSurrogate(point_feature_dim, c_out).eval()
        dummy = torch.randn(int(key.batch_size), int(key.fixed_K), point_feature_dim)
        torch.onnx.export(
            module,
            dummy,
            str(onnx_path),
            input_names=["points"],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
        )
        return {**spec, "status": "exported", "onnx_path": str(onnx_path)}
    else:
        module = ConvBNReLU(c_in, c_out, kernel, stride, padding, dilation, groups).eval()
    dummy = torch.randn(int(key.batch_size), c_in, h, w)
    torch.onnx.export(
        module,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
    )
    return {**spec, "status": "exported", "onnx_path": str(onnx_path)}
