"""Pure TensorRT command generation."""

from __future__ import annotations

from pathlib import Path

from ..config import TensorRTBuildConfig
from ..exceptions import TensorRTConfigurationError
from ..types import CanonicalPrecisionMappingResult, TensorRTCommandResult


def _shape_flags(profiles: dict[str, dict[str, tuple[int, ...]]]) -> list[str]:
    flags = []
    for kind in ("min", "opt", "max"):
        values = []
        for name in sorted(profiles):
            profile = profiles[name]
            if kind not in profile:
                raise TensorRTConfigurationError(f"shape profile {name} is missing {kind}")
            values.append(f"{name}:{'x'.join(str(int(dim)) for dim in profile[kind])}")
        if values:
            flags.append(f"--{kind}Shapes={','.join(values)}")
    return flags


def build_trt_command(
    onnx_path: str | Path,
    engine_path: str | Path,
    precision_mapping: CanonicalPrecisionMappingResult,
    *,
    config: TensorRTBuildConfig | None = None,
    layer_info_path: str | Path | None = None,
) -> TensorRTCommandResult:
    """Generate a canonical layer-constrained trtexec command without running it."""

    policy = config or TensorRTBuildConfig()
    if policy.production_mode and not policy.strongly_typed:
        raise TensorRTConfigurationError("production_requires_strongly_typed")
    if policy.strongly_typed:
        if policy.production_mode and (policy.enable_fp16 or policy.enable_int8):
            raise TensorRTConfigurationError(
                "strongly_typed_forbids_implicit_precision_flags"
            )
        if policy.production_mode and str(policy.precision_constraints).lower() not in {"", "none"}:
            raise TensorRTConfigurationError(
                "strongly_typed_forbids_precision_constraints"
            )
        if policy.production_mode and str(policy.plugin_boundary_dtype).lower() not in {
            "fp16",
            "fp32",
        }:
            raise TensorRTConfigurationError(
                "strongly_typed_plugin_boundary_must_be_fp16_or_fp32"
            )
    if not precision_mapping.entries:
        raise TensorRTConfigurationError("canonical precision mapping is empty")
    canonical_names = [row.canonical_node_name for row in precision_mapping.entries]
    if len(canonical_names) != len(set(canonical_names)) or any(not name.startswith("__canonical__") for name in canonical_names):
        raise TensorRTConfigurationError("layer precision constraints require unique canonical node names")
    invalid = [
        row.realized_request_precision
        for row in precision_mapping.entries
        if row.realized_request_precision not in {"fp32", "fp16", "bf16", "fp8", "int8"}
    ]
    if invalid:
        raise TensorRTConfigurationError(f"unsupported realized precisions: {sorted(set(invalid))}")
    invalid_outputs = [
        row.realized_output_precision
        for row in precision_mapping.entries
        if row.realized_output_precision and row.realized_output_precision not in {"fp32", "fp16", "bf16", "fp8", "int8"}
    ]
    if invalid_outputs:
        raise TensorRTConfigurationError(f"unsupported realized output precisions: {sorted(set(invalid_outputs))}")
    source = Path(onnx_path)
    engine = Path(engine_path)
    layer_info = Path(layer_info_path) if layer_info_path is not None else engine.with_suffix(".layerinfo.json")
    executable = str(policy.trtexec_path) if policy.trtexec_path is not None else "trtexec"
    command = [
        executable,
        f"--onnx={source}",
        f"--saveEngine={engine}",
        "--profilingVerbosity=detailed",
        f"--memPoolSize=workspace:{int(policy.workspace_mib)}",
    ]
    if policy.export_layer_info:
        command.append(f"--exportLayerInfo={layer_info}")
    if policy.skip_inference:
        command.append("--skipInference")
    if policy.no_tf32:
        command.append("--noTF32")
    if policy.strongly_typed:
        # A strongly typed TensorRT network takes its compute and tensor types
        # exclusively from the ONNX graph.  Builder precision flags and layer
        # constraints are weak-typing hints and are forbidden in this mode.
        command.append("--stronglyTyped")
    else:
        if policy.enable_fp16:
            command.append("--fp16")
        if policy.enable_int8 and any(row.realized_request_precision == "int8" for row in precision_mapping.entries):
            command.append("--int8")
    if policy.plugin_path is not None:
        command.append(f"--staticPlugins={policy.plugin_path}")
    command.extend(_shape_flags(policy.shape_profiles))
    constraint_rows = [
        (node_name, row)
        for row in sorted(precision_mapping.entries, key=lambda item: item.canonical_node_name)
        for node_name in (row.constraint_node_names or (row.canonical_node_name,))
    ]
    constraint_names = [name for name, _ in constraint_rows]
    if len(constraint_names) != len(set(constraint_names)):
        raise TensorRTConfigurationError("expanded layer precision constraint names are not unique")
    auxiliary_precisions = dict(precision_mapping.auxiliary_layer_precisions)
    auxiliary_outputs = dict(precision_mapping.auxiliary_layer_output_types)
    if set(constraint_names) & set(auxiliary_precisions):
        raise TensorRTConfigurationError("auxiliary precision constraints overlap canonical constraints")
    invalid_auxiliary = sorted(
        {str(value) for value in [*auxiliary_precisions.values(), *auxiliary_outputs.values()]}
        - {"fp32", "fp16", "bf16", "fp8", "int8"}
    )
    if invalid_auxiliary:
        raise TensorRTConfigurationError(f"unsupported auxiliary precisions: {invalid_auxiliary}")
    compute_specs = ",".join(
        [
            *(
                f"{node_name}:{row.realized_request_precision}"
                for node_name, row in constraint_rows
            ),
            *(
                f"{name}:{precision}"
                for name, precision in sorted(auxiliary_precisions.items())
            ),
        ]
    )
    output_specs = ",".join(
        [
            *(
                f"{node_name}:{row.realized_output_precision or row.realized_request_precision}"
                for node_name, row in constraint_rows
            ),
            *(
                f"{name}:{precision}"
                for name, precision in sorted(auxiliary_outputs.items())
            ),
        ]
    )
    if not policy.strongly_typed:
        command.extend(
            [
                f"--precisionConstraints={policy.precision_constraints}",
                f"--layerPrecisions={compute_specs}",
                f"--layerOutputTypes={output_specs}",
            ]
        )
    return TensorRTCommandResult(
        command=command,
        onnx_path=str(source),
        engine_path=str(engine),
        layer_info_path=str(layer_info),
        policy_version=policy.policy_version,
    )
