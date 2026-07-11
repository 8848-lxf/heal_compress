"""Signal-maxK ONNX export using an export-ready PyTorch module."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ..artifacts.io import atomic_write_json
from ..config import CanonicalNamingConfig, OnnxExportConfig
from ..exceptions import OnnxExportError
from ..types import OnnxExportResult
from .origin_mapping import apply_canonical_node_names, build_onnx_origin_map


REQUIRED_SIGNAL_MAXK_INPUTS = (
    "voxel_features",
    "voxel_coords",
    "voxel_num_points",
    "pairwise_t_matrix",
    "valid_voxel_mask",
)
EXPECTED_SIGNAL_MAXK_OUTPUTS = ("cls_preds", "reg_preds", "dir_preds")


def signal_maxk_dynamic_axes(output_names: Sequence[str]) -> dict[str, dict[int, str]]:
    """Return fixed-K/dynamic-agent axes used by the deployed graph."""

    axes: dict[str, dict[int, str]] = {"pairwise_t_matrix": {1: "num_agents", 2: "num_agents"}}
    axes.update({str(name): {0: "batch"} for name in output_names})
    return axes


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in getattr(value, "shape", ()))


def _validate_example_inputs(inputs: Sequence[Any], config: OnnxExportConfig) -> None:
    if len(inputs) != len(config.input_names):
        raise OnnxExportError(f"expected {len(config.input_names)} signal-maxK inputs, got {len(inputs)}")
    by_name = dict(zip(config.input_names, inputs))
    for name in ("voxel_features", "voxel_coords", "voxel_num_points", "valid_voxel_mask"):
        shape = _shape(by_name[name])
        if not shape or shape[0] != int(config.fixed_k):
            raise OnnxExportError(f"{name} must have fixed first dimension K={config.fixed_k}, got {shape}")
    pairwise = _shape(by_name["pairwise_t_matrix"])
    if len(pairwise) != 5 or pairwise[1] != pairwise[2]:
        raise OnnxExportError(f"pairwise_t_matrix must be [1,N,N,4,4], got {pairwise}")
    if not (config.min_agents <= pairwise[1] <= config.max_agents):
        raise OnnxExportError(f"example agent count {pairwise[1]} is outside configured profile")


@contextmanager
def capture_weighted_module_calls(model: Any) -> Iterator[list[dict[str, Any]]]:
    """Capture weighted calls without patching process-global exporter state."""

    import torch

    records: list[dict[str, Any]] = []
    handles = []
    counter = 0

    def register(module_path: str, module: Any) -> None:
        def hook(_module: Any, _inputs: tuple[Any, ...], _output: Any) -> None:
            nonlocal counter
            mapped = "ConvTranspose" if isinstance(module, torch.nn.ConvTranspose2d) else "Conv" if isinstance(module, torch.nn.Conv2d) else "MatMul"
            weight = getattr(module, "weight", None)
            records.append(
                {
                    "module_path": module_path.removeprefix("model."),
                    "module_type": type(module).__name__,
                    "call_index": counter,
                    "mapped_onnx_op_type": mapped,
                    "weight_shape": list(weight.shape) if weight is not None else [],
                    "groups": int(getattr(module, "groups", 1) or 1),
                }
            )
            counter += 1

        handles.append(module.register_forward_hook(hook))

    for name, module in model.named_modules():
        if name and isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
            register(str(name), module)
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


def inspect_signal_maxk_onnx(path: str | Path, *, validate: bool = True, allow_custom_ops: bool = True) -> dict[str, Any]:
    """Inspect bindings, output contract, custom ops, and checker status."""

    import onnx

    model = onnx.load(str(path))
    inputs = [str(value.name) for value in model.graph.input]
    outputs = [str(value.name) for value in model.graph.output]
    custom_ops = sorted({f"{node.domain}::{node.op_type}" for node in model.graph.node if str(node.domain)})
    checker_passed = True
    checker_error = ""
    if validate:
        try:
            onnx.checker.check_model(model)
        except Exception as exc:
            checker_passed = bool(allow_custom_ops and custom_ops)
            checker_error = f"{type(exc).__name__}: {exc}"
    return {
        "input_names": inputs,
        "output_names": outputs,
        "required_inputs_present": set(REQUIRED_SIGNAL_MAXK_INPUTS) <= set(inputs),
        "expected_outputs_present": set(EXPECTED_SIGNAL_MAXK_OUTPUTS) <= set(outputs),
        "custom_ops": custom_ops,
        "checker_passed": checker_passed,
        "checker_error": checker_error,
    }


def export_signal_maxk_onnx(
    model: Any,
    example_inputs: Sequence[Any] | Mapping[str, Any],
    output_path: str | Path,
    *,
    config: OnnxExportConfig | None = None,
    naming_config: CanonicalNamingConfig | None = None,
    report_path: str | Path | None = None,
) -> OnnxExportResult:
    """Export, origin-map, and canonically name a signal-maxK graph.

    The model must already expose the five tensor arguments in configured
    order. Model construction and data selection are explicit adapter inputs.
    """

    import torch

    policy = config or OnnxExportConfig()
    if policy.custom_op_domain != "trt":
        raise OnnxExportError(
            f"PointPillarScatterTRT requires custom_op_domain='trt', got {policy.custom_op_domain!r}"
        )
    tensors = tuple(example_inputs[name] for name in policy.input_names) if isinstance(example_inputs, Mapping) else tuple(example_inputs)
    _validate_example_inputs(tensors, policy)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = signal_maxk_dynamic_axes(policy.output_names) if policy.dynamic_agent_dimension else None
    try:
        with capture_weighted_module_calls(model) as calls:
            torch.onnx.export(
                model,
                tensors,
                str(destination),
                input_names=list(policy.input_names),
                output_names=list(policy.output_names),
                dynamic_axes=dynamic_axes,
                opset_version=int(policy.opset_version),
                custom_opsets={str(policy.custom_op_domain): int(policy.custom_opset_version)},
                do_constant_folding=bool(policy.do_constant_folding),
            )
        origin = build_onnx_origin_map(destination, calls, naming_config=naming_config)
        rename = apply_canonical_node_names(destination, origin, output_path=destination, allow_custom_ops=policy.allow_custom_ops)
        audit = inspect_signal_maxk_onnx(destination, validate=policy.validate_onnx, allow_custom_ops=policy.allow_custom_ops)
    except Exception as exc:
        if isinstance(exc, OnnxExportError):
            raise
        raise OnnxExportError(f"signal-maxK ONNX export failed: {type(exc).__name__}: {exc}") from exc
    if not audit["required_inputs_present"] or not audit["expected_outputs_present"] or not audit["checker_passed"]:
        raise OnnxExportError(f"signal-maxK ONNX validation failed: {audit}")
    result = OnnxExportResult(
        onnx_path=str(destination),
        input_names=list(audit["input_names"]),
        output_names=list(audit["output_names"]),
        fixed_k=int(policy.fixed_k),
        dynamic_agent_dimension=bool(policy.dynamic_agent_dimension),
        checker_passed=bool(audit["checker_passed"]),
        origin_map=origin,
        canonical_rename=rename,
    )
    if report_path is not None:
        atomic_write_json(report_path, result.to_dict())
    return result
