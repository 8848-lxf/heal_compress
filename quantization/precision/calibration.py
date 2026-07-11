"""Calibration metadata validation (collection is caller/integration owned)."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Mapping

from ..config import CalibrationConfig
from ..exceptions import QDQInsertionError
from ..types import CalibrationResult, CalibrationScaleRecord


def _first_tensor(value: Any) -> Any | None:
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def collect_calibration_scales(
    model: Any,
    batches: Iterable[Any],
    *,
    module_paths: Sequence[str],
    forward_fn: Callable[[Any, Any], Any] | None = None,
    config: CalibrationConfig | None = None,
) -> CalibrationResult:
    """Collect symmetric per-tensor activation and weight scales.

    Dataset selection and input preparation remain caller-owned. Hooks retain
    only detached scalar tensors, so no activation graph is kept between
    frames. Every requested weighted module must be observed at least once.
    """

    import torch

    policy = config or CalibrationConfig()
    if policy.split != "train":
        raise QDQInsertionError("formal INT8 calibration split must be train")
    modules = dict(model.named_modules())
    requested = [str(value) for value in module_paths]
    if len(requested) != len(set(requested)):
        raise QDQInsertionError("calibration module paths are not unique")
    missing = [name for name in requested if name not in modules or getattr(modules[name], "weight", None) is None]
    if missing:
        raise QDQInsertionError(f"calibration weighted modules are missing: {missing}")
    state: dict[str, dict[str, Any]] = {
        name: {"input": None, "output": None, "count": 0} for name in requested
    }
    handles = []

    def register(name: str) -> None:
        def hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            input_tensor = _first_tensor(inputs)
            output_tensor = _first_tensor(output)
            if input_tensor is None or output_tensor is None:
                raise QDQInsertionError(f"weighted module {name} has no tensor input/output")
            input_amax = input_tensor.detach().abs().amax()
            output_amax = output_tensor.detach().abs().amax()
            row = state[name]
            row["input"] = input_amax if row["input"] is None else torch.maximum(row["input"], input_amax)
            row["output"] = output_amax if row["output"] is None else torch.maximum(row["output"], output_amax)
            row["count"] += 1

        handles.append(modules[name].register_forward_hook(hook))

    for name in requested:
        register(name)
    was_training = bool(model.training)
    model.eval()
    frame_count = 0
    try:
        with torch.inference_mode():
            for batch in batches:
                forward_fn(model, batch) if forward_fn is not None else model(batch)
                frame_count += 1
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    if frame_count <= 0:
        raise QDQInsertionError("calibration received no frames")
    if policy.require_observed_scales and frame_count != int(policy.frame_count):
        raise QDQInsertionError(
            f"calibration observed {frame_count} frames, expected exactly {policy.frame_count}"
        )
    records: list[CalibrationScaleRecord] = []
    for name in requested:
        row = state[name]
        if int(row["count"]) <= 0 or row["input"] is None or row["output"] is None:
            raise QDQInsertionError(f"calibration module was never observed: {name}")
        input_amax = float(row["input"].item())
        output_amax = float(row["output"].item())
        weight_amax = float(modules[name].weight.detach().abs().amax().item())
        if min(input_amax, output_amax, weight_amax) <= 0.0:
            raise QDQInsertionError(f"zero calibration amax is invalid for {name}")
        records.append(
            CalibrationScaleRecord(
                module_path=name,
                activation_input_scale=input_amax / 127.0,
                weight_scale=weight_amax / 127.0,
                activation_output_scale=output_amax / 127.0,
                activation_input_amax=input_amax,
                weight_amax=weight_amax,
                activation_output_amax=output_amax,
                observation_count=int(row["count"]),
            )
        )
    return CalibrationResult(
        records=records,
        frame_count=frame_count,
        module_count=len(records),
        split=policy.split,
        activation_granularity=policy.activation_granularity,
        weight_granularity=policy.weight_granularity,
    )


def validate_calibration_scales(scales: Mapping[str, Any], *, config: CalibrationConfig | None = None) -> dict[str, Any]:
    """Validate positive finite scales and record calibration provenance."""

    policy = config or CalibrationConfig()
    if policy.split != "train":
        raise QDQInsertionError("formal INT8 calibration split must be train")
    invalid = []
    for name, raw in scales.items():
        values = (
            [raw.get("activation_input_scale"), raw.get("weight_scale"), raw.get("activation_output_scale")]
            if isinstance(raw, Mapping) and "activation_input_scale" in raw
            else [raw.get("scale") if isinstance(raw, Mapping) else raw]
        )
        try:
            valid = all(math.isfinite(float(value)) and float(value) > 0.0 for value in values)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            invalid.append(str(name))
    if invalid:
        raise QDQInsertionError(f"invalid calibration scales: {invalid}")
    return {
        "calibration_schema_version": policy.schema_version,
        "split": policy.split,
        "frame_count": int(policy.frame_count),
        "scale_count": len(scales),
        "activation_granularity": policy.activation_granularity,
        "weight_granularity": policy.weight_granularity,
    }


__all__ = ["collect_calibration_scales", "validate_calibration_scales"]
