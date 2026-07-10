from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch


PREFERRED_OUTPUT_NAMES = ("cls_preds", "reg_preds", "dir_preds")
DEFAULT_INPUT_NAMES = ("voxel_features", "voxel_coords", "voxel_num_points", "pairwise_t_matrix", "valid_voxel_mask")


def to_device(data: Any, device: torch.device) -> Any:
    if isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [to_device(value, device) for value in data]
    if hasattr(data, "to") and not isinstance(data, (str, bytes)):
        return data.to(device, non_blocking=True)
    return data


def infer_modality(model: Any) -> str:
    names = getattr(model, "modality_name_list", None)
    if names:
        return str(names[0])
    return str(getattr(model, "ego_modality", "m1"))


def _record_len_value(sample: dict[str, Any]) -> int:
    record_len = sample["record_len"]
    if torch.is_tensor(record_len):
        return int(record_len.detach().sum().item())
    return int(record_len)


def _pad_first_dim(tensor: torch.Tensor, fixed_k: int, *, fill: float = 0.0) -> torch.Tensor:
    current = int(tensor.shape[0])
    if current > int(fixed_k):
        raise ValueError(f"num_voxels={current} exceeds fixed_k={int(fixed_k)}")
    if current == int(fixed_k):
        return tensor.contiguous()
    shape = (int(fixed_k), *tuple(int(v) for v in tensor.shape[1:]))
    out = torch.full(shape, fill, dtype=tensor.dtype, device=tensor.device)
    out[:current].copy_(tensor)
    return out.contiguous()


def _valid_voxel_mask(num_voxels: int, fixed_k: int, *, device: torch.device) -> torch.Tensor:
    if int(num_voxels) > int(fixed_k):
        raise ValueError(f"num_voxels={num_voxels} exceeds fixed_k={int(fixed_k)}")
    mask = torch.zeros((int(fixed_k),), dtype=torch.float32, device=device)
    mask[: int(num_voxels)] = 1.0
    return mask


def extract_lidar_tensors(sample: dict[str, Any], modality: str, *, fixed_k: int | None = None) -> tuple[dict[str, torch.Tensor], list[str]]:
    input_key = f"inputs_{modality}"
    if input_key not in sample:
        candidates = [key for key in sample if str(key).startswith("inputs_")]
        if not candidates:
            raise KeyError(f"sample does not contain {input_key} or any inputs_<modality> key")
        input_key = candidates[0]
        modality = input_key.replace("inputs_", "", 1)
    lidar_inputs = sample[input_key]
    voxel_features = lidar_inputs["voxel_features"].float()
    voxel_coords = lidar_inputs["voxel_coords"].to(torch.int32)
    voxel_num_points = lidar_inputs["voxel_num_points"].to(torch.int32)
    n_agents = _record_len_value(sample)
    pairwise_t_matrix = sample["pairwise_t_matrix"][:, :n_agents, :n_agents, :, :].float()
    tensors = {
        "voxel_features": voxel_features,
        "voxel_coords": voxel_coords,
        "voxel_num_points": voxel_num_points,
        "record_len": sample["record_len"],
        "pairwise_t_matrix": pairwise_t_matrix,
    }
    if fixed_k is not None:
        original_k = int(voxel_features.shape[0])
        valid_mask = _valid_voxel_mask(original_k, int(fixed_k), device=voxel_features.device)
        padded_points = _pad_first_dim(voxel_num_points, int(fixed_k), fill=1)
        padded_points = torch.where(valid_mask > 0.5, padded_points, torch.ones_like(padded_points)).to(torch.int32)
        tensors.update(
            {
                "voxel_features": _pad_first_dim(voxel_features, int(fixed_k), fill=0.0),
                "voxel_coords": _pad_first_dim(voxel_coords, int(fixed_k), fill=0).to(torch.int32),
                "voxel_num_points": padded_points,
                "valid_voxel_mask": valid_mask,
            }
        )
    agent_modality_list = sample.get("agent_modality_list")
    if agent_modality_list is None:
        agent_num = _record_len_value(sample)
        agent_modality_list = [modality for _ in range(agent_num)]
    return tensors, list(agent_modality_list)


def bind_inputs_for_engine(engine_input_names: list[str], sample: dict[str, Any], modality: str, *, fixed_k: int | None = None) -> dict[str, torch.Tensor]:
    needs_fixed_k = "valid_voxel_mask" in set(engine_input_names)
    tensors, _agent_modalities = extract_lidar_tensors(sample, modality, fixed_k=fixed_k if needs_fixed_k else None)
    missing = [name for name in engine_input_names if name not in tensors]
    if missing:
        raise KeyError(f"HEAL TensorRT input adapter cannot provide bindings {missing}; available={sorted(tensors)}")
    return {name: tensors[name] for name in engine_input_names}


def tensor_output_names(raw_output: dict[str, Any]) -> list[str]:
    names = [name for name in PREFERRED_OUTPUT_NAMES if name in raw_output and torch.is_tensor(raw_output[name])]
    if names:
        return names
    names = [name for name, value in raw_output.items() if torch.is_tensor(value)]
    if not names:
        raise RuntimeError("model forward produced no tensor outputs suitable for HEAL TensorRT output adapter")
    return names


def adapt_outputs_for_postprocess(output_names: list[str], trt_outputs: dict[str, torch.Tensor]) -> OrderedDict:
    missing = [name for name in output_names if name not in trt_outputs]
    if missing:
        raise KeyError(f"TensorRT outputs missing HEAL postprocess tensors {missing}; engine_outputs={sorted(trt_outputs)}")
    output = {name: trt_outputs[name].float() for name in output_names}
    wrapped = OrderedDict()
    wrapped["ego"] = output
    return wrapped
