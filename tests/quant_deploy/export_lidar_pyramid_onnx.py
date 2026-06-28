from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant_deploy_utils import (
    DEFAULT_CHECKPOINT,
    DEFAULT_HEAL_REPO,
    DEFAULT_HYPES_YAML,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRT_ROOT,
    collect_env_report,
    create_quant_deploy_run_dirs,
    detect_special_ops_in_onnx,
    dirs_for_summary,
    ensure_quant_deploy_run_dirs,
    onnx_graph_info,
    save_json,
    write_debug_reports,
)
from exportable_bev_warp import check_exportable_bev_warp_equivalence, warp_affine_simple_exportable
from exportable_lidar_pyramid import (
    FixedLidarPyramidExportWrapper,
    SOURCE_FIXED_WRAPPER,
    check_fixed_lidar_pyramid_wrapper_equivalence,
)
from exportable_lidar_pyramid_dynamic_agent import (
    ExportableLidarPyramidDynamicAgent,
    SOURCE_DYNAMIC_AGENT_WRAPPER,
    check_dynamic_agent_wrapper_equivalence,
    dynamic_agent_dynamic_axes,
    dynamic_agent_input_names,
)
from exportable_lidar_pyramid_padded_agent import (
    ExportableLidarPyramidPaddedAgent,
    SOURCE_PADDED_WRAPPER,
    check_padded_agent_wrapper_equivalence,
    make_valid_agent_mask,
    padded_agent_dynamic_axes,
    padded_agent_input_names,
)
from exportable_lidar_pyramid_fixed_k_scatter_plugin import (
    ExportableLidarPyramidFixedKScatterPlugin,
    SOURCE_FIXED_K_SCATTER_PLUGIN_WRAPPER,
    check_fixed_k_scatter_plugin_wrapper_equivalence,
    fixed_k_scatter_plugin_dynamic_axes,
    fixed_k_scatter_plugin_input_names,
)
from exportable_lidar_pyramid_dynamic_fixed_k_scatter_plugin import (
    ExportableLidarPyramidDynamicFixedKScatterPlugin,
    SOURCE_DYNAMIC_FIXED_K_SCATTER_PLUGIN_WRAPPER,
    check_dynamic_fixed_k_scatter_plugin_wrapper_equivalence,
    dynamic_fixed_k_scatter_plugin_dynamic_axes,
    dynamic_fixed_k_scatter_plugin_input_names,
)
from exportable_pillar_vfe import (
    check_pillar_vfe_export_fix_equivalence,
    pillar_vfe_forward_explicit_squeeze,
)


INPUT_NAMES = ["voxel_features", "voxel_coords", "voxel_num_points", "record_len", "pairwise_t_matrix"]
PYRAMID_FORWARD_EXPORT_MODES = (
    "original",
    "fixed_static",
    "dynamic_agent_dim",
    "padded_agent_static",
    "fixed_k_scatter_plugin",
    "dynamic_agent_dim_fixed_k_scatter_plugin",
)


class LidarPyramidDeployWrapper(nn.Module):
    def __init__(self, model: nn.Module, modality_name: str, agent_modality_list: list[str], output_names: list[str]):
        super().__init__()
        self.model = model
        self.modality_name = modality_name
        self.agent_modality_list = list(agent_modality_list)
        self.output_names = list(output_names)

    def forward(
        self,
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        voxel_num_points: torch.Tensor,
        record_len: torch.Tensor,
        pairwise_t_matrix: torch.Tensor,
    ):
        data_dict = {
            "agent_modality_list": self.agent_modality_list,
            "record_len": record_len,
            "pairwise_t_matrix": pairwise_t_matrix,
            f"inputs_{self.modality_name}": {
                "voxel_features": voxel_features,
                "voxel_coords": voxel_coords,
                "voxel_num_points": voxel_num_points,
            },
        }
        outputs = self.model(data_dict)
        return tuple(outputs[name] for name in self.output_names)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export HEAL LiDAROnly lidar_pyramid FP32 dynamic ONNX.")
    parser.add_argument("--hypes_yaml", default=str(DEFAULT_HYPES_YAML))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal_repo", default=str(DEFAULT_HEAL_REPO))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output_root", default=None, help="Existing run root. Used by the one-key runner.")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--max_cav", type=int, default=2)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow_synthetic_fallback", action="store_true")
    parser.add_argument("--trt_root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--trtexec_path", default=None)
    parser.add_argument("--bev_warp_export_mode", default="exportable_grid", choices=["original", "exportable_grid"])
    parser.add_argument("--pillar_vfe_export_fix", default="explicit_squeeze", choices=["none", "explicit_squeeze"])
    parser.add_argument("--pyramid_forward_export_mode", default="fixed_static", choices=list(PYRAMID_FORWARD_EXPORT_MODES))
    parser.add_argument("--fixed_k", type=int, default=None, help="Fixed voxel count for fixed_k_scatter_plugin exports.")
    parser.add_argument("--fixed_num_agents", type=int, default=None, help="Fixed N for dynamic_agent_dim_fixed_k_scatter_plugin fallback exports.")
    return parser.parse_args(argv)


def _add_paths(heal_repo: str | Path) -> None:
    root = Path(__file__).resolve().parents[2]
    parent = root.parent
    for item in (str(parent), str(root), str(Path(heal_repo).expanduser().resolve())):
        if item not in sys.path:
            sys.path.insert(0, item)


def _load_hypes(hypes_yaml: str | Path, heal_repo: str | Path) -> dict[str, Any]:
    from opencood.hypes_yaml import yaml_utils

    hypes = yaml_utils.load_yaml(str(Path(hypes_yaml).expanduser()))
    heal_root = Path(heal_repo).expanduser()
    for key in ("data_dir", "root_dir", "validate_dir", "test_dir"):
        value = hypes.get(key)
        if isinstance(value, str) and value and not Path(value).is_absolute():
            hypes[key] = str(heal_root / value)
    return hypes


def _load_model(hypes: dict[str, Any], checkpoint: str | Path, device: torch.device) -> nn.Module:
    from opencood.tools import train_utils

    model = train_utils.create_model(hypes)
    ckpt_path = Path(checkpoint).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {ckpt_path}")
    state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"load_state_dict missing keys ({len(missing)}): {missing[:20]}")
    if unexpected:
        print(f"load_state_dict unexpected keys ({len(unexpected)}): {unexpected[:20]}")
    model.to(device)
    model.eval()
    return model


def _to_device(data: Any, device: torch.device) -> Any:
    if isinstance(data, dict):
        return {key: _to_device(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [_to_device(value, device) for value in data]
    if hasattr(data, "to") and not isinstance(data, (str, bytes)):
        return data.to(device, non_blocking=True)
    return data


def _first_real_sample(hypes: dict[str, Any], device: torch.device) -> dict[str, Any]:
    from opencood.data_utils.datasets import build_dataset
    from torch.utils.data import DataLoader

    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_batch_test)
    for batch in loader:
        if batch is None:
            continue
        ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
        return _to_device(ego, device)
    raise RuntimeError("No valid validation/test sample was produced by the HEAL dataloader.")


def _real_sample_for_export_mode(
    hypes: dict[str, Any],
    device: torch.device,
    *,
    export_mode: str,
    max_cav: int,
    max_scan_frames: int = 64,
) -> tuple[dict[str, Any], str]:
    if export_mode not in {"dynamic_agent_dim", "padded_agent_static", "dynamic_agent_dim_fixed_k_scatter_plugin"}:
        return _first_real_sample(hypes, device), "real_validation_or_test_sample"

    from opencood.data_utils.datasets import build_dataset
    from torch.utils.data import DataLoader

    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_batch_test)
    fallback: dict[str, Any] | None = None
    for frame_idx, batch in enumerate(loader):
        if frame_idx >= int(max_scan_frames):
            break
        if batch is None:
            continue
        ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
        ego = _to_device(ego, device)
        if fallback is None:
            fallback = ego
        record_len = ego.get("record_len")
        target_agents = int(max_cav)
        if record_len is not None and _record_len_agent_count(record_len) == target_agents:
            return ego, f"real_validation_or_test_sample_record_len_{target_agents}"
    if fallback is not None:
        return fallback, "real_validation_or_test_sample_fallback_no_max_cav_match"
    raise RuntimeError("No valid validation/test sample was produced by the HEAL dataloader.")


def _synthetic_sample(model: nn.Module, device: torch.device, max_cav: int) -> dict[str, Any]:
    modality = _infer_modality(model)
    agent_num = max(1, int(max_cav))
    voxels_per_agent = 24
    max_points = 32
    encoder = getattr(model, f"encoder_{modality}", None)
    scatter = getattr(encoder, "scatter", None)
    nx = int(getattr(scatter, "nx", 16))
    ny = int(getattr(scatter, "ny", 16))
    coords = []
    for agent_idx in range(agent_num):
        for voxel_idx in range(voxels_per_agent):
            coords.append([agent_idx, 0, (voxel_idx * 5 + agent_idx) % max(ny, 1), (voxel_idx * 7 + agent_idx) % max(nx, 1)])
    voxel_features = torch.zeros((agent_num * voxels_per_agent, max_points, 4), dtype=torch.float32, device=device)
    voxel_features[..., 3] = 1.0
    pairwise_t_matrix = torch.eye(4, dtype=torch.float32, device=device).view(1, 1, 1, 4, 4).repeat(1, agent_num, agent_num, 1, 1)
    return {
        "agent_modality_list": [modality for _ in range(agent_num)],
        "record_len": torch.tensor([agent_num], dtype=torch.long, device=device),
        "pairwise_t_matrix": pairwise_t_matrix,
        f"inputs_{modality}": {
            "voxel_features": voxel_features,
            "voxel_coords": torch.tensor(coords, dtype=torch.int32, device=device),
            "voxel_num_points": torch.full((agent_num * voxels_per_agent,), max_points, dtype=torch.int32, device=device),
        },
    }


def _infer_modality(model: nn.Module) -> str:
    names = getattr(model, "modality_name_list", None)
    if names:
        return str(names[0])
    return str(getattr(model, "ego_modality", "m1"))


def _extract_inputs(sample: dict[str, Any], modality: str) -> tuple[tuple[torch.Tensor, ...], list[str]]:
    input_key = f"inputs_{modality}"
    if input_key not in sample:
        candidate_keys = [key for key in sample if key.startswith("inputs_")]
        if not candidate_keys:
            raise KeyError(f"Sample does not contain inputs_{modality} or any inputs_<modality> tensor dict.")
        input_key = candidate_keys[0]
        modality = input_key.replace("inputs_", "", 1)
    lidar_inputs = sample[input_key]
    tensors = (
        lidar_inputs["voxel_features"],
        lidar_inputs["voxel_coords"],
        lidar_inputs["voxel_num_points"],
        sample["record_len"],
        sample["pairwise_t_matrix"],
    )
    agent_modality_list = sample.get("agent_modality_list")
    if agent_modality_list is None:
        agent_num = int(sample["record_len"].sum().item())
        agent_modality_list = [modality for _ in range(agent_num)]
    return tensors, list(agent_modality_list)


def _record_len_agent_count(record_len: torch.Tensor) -> int:
    return int(record_len.detach().sum().item()) if torch.is_tensor(record_len) else int(record_len)


def _input_names_for_export_mode(mode: str) -> list[str]:
    if mode == "dynamic_agent_dim":
        return dynamic_agent_input_names()
    if mode == "padded_agent_static":
        return padded_agent_input_names()
    if mode == "fixed_k_scatter_plugin":
        return fixed_k_scatter_plugin_input_names()
    if mode == "dynamic_agent_dim_fixed_k_scatter_plugin":
        return dynamic_fixed_k_scatter_plugin_input_names()
    return list(INPUT_NAMES)


def _prepare_export_tensors(
    tensors: tuple[torch.Tensor, ...],
    *,
    export_mode: str,
    max_cav: int,
) -> tuple[torch.Tensor, ...]:
    voxel_features, voxel_coords, voxel_num_points, record_len, pairwise_t_matrix = tensors
    if export_mode == "dynamic_agent_dim":
        agent_count = _record_len_agent_count(record_len)
        return (
            voxel_features,
            voxel_coords,
            voxel_num_points,
            pairwise_t_matrix[:, :agent_count, :agent_count, :, :],
        )
    if export_mode == "padded_agent_static":
        valid_agent_mask = make_valid_agent_mask(record_len, max_cav=max_cav, dtype=voxel_features.dtype)
        return (
            voxel_features,
            voxel_coords,
            voxel_num_points,
            valid_agent_mask,
            pairwise_t_matrix[:, :max_cav, :max_cav, :, :],
        )
    if export_mode == "fixed_k_scatter_plugin":
        valid_agent_mask = make_valid_agent_mask(record_len, max_cav=max_cav, dtype=voxel_features.dtype)
        valid_voxel_mask = torch.ones((int(voxel_features.shape[0]),), dtype=voxel_features.dtype, device=voxel_features.device)
        return (
            voxel_features,
            voxel_coords,
            voxel_num_points,
            valid_agent_mask,
            pairwise_t_matrix[:, :max_cav, :max_cav, :, :],
            valid_voxel_mask,
        )
    if export_mode == "dynamic_agent_dim_fixed_k_scatter_plugin":
        agent_count = _record_len_agent_count(record_len)
        valid_voxel_mask = torch.ones((int(voxel_features.shape[0]),), dtype=voxel_features.dtype, device=voxel_features.device)
        return (
            voxel_features,
            voxel_coords,
            voxel_num_points,
            pairwise_t_matrix[:, :agent_count, :agent_count, :, :],
            valid_voxel_mask,
        )
    return tensors


def _pad_fixed_k_export_tensors(tensors: tuple[torch.Tensor, ...], fixed_k: int) -> tuple[tuple[torch.Tensor, ...], int, int]:
    fixed_k = int(fixed_k)
    if len(tensors) != 6:
        raise ValueError(f"fixed_k_scatter_plugin expects 6 tensors, got {len(tensors)}")
    voxel_features, voxel_coords, voxel_num_points, valid_agent_mask, pairwise_t_matrix, valid_voxel_mask = tensors
    original_num_voxels = int(voxel_features.shape[0])
    if original_num_voxels > fixed_k:
        raise ValueError(f"num_voxels={original_num_voxels} exceeds fixed_k={fixed_k}")

    def _pad_first_dim(tensor: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
        shape = list(tensor.shape)
        shape[0] = fixed_k
        out = torch.full(tuple(shape), fill_value, dtype=tensor.dtype, device=tensor.device)
        if original_num_voxels:
            out[:original_num_voxels].copy_(tensor)
        return out

    padded_valid = torch.zeros((fixed_k,), dtype=valid_voxel_mask.dtype, device=valid_voxel_mask.device)
    if original_num_voxels:
        padded_valid[:original_num_voxels] = 1
    return (
        (
            _pad_first_dim(voxel_features),
            _pad_first_dim(voxel_coords),
            _pad_first_dim(voxel_num_points),
            valid_agent_mask,
            pairwise_t_matrix,
            padded_valid,
        ),
        original_num_voxels,
        fixed_k,
    )


def _pad_dynamic_fixed_k_export_tensors(tensors: tuple[torch.Tensor, ...], fixed_k: int) -> tuple[tuple[torch.Tensor, ...], int, int]:
    fixed_k = int(fixed_k)
    if len(tensors) != 5:
        raise ValueError(f"dynamic_agent_dim_fixed_k_scatter_plugin expects 5 tensors, got {len(tensors)}")
    voxel_features, voxel_coords, voxel_num_points, pairwise_t_matrix, valid_voxel_mask = tensors
    original_num_voxels = int(voxel_features.shape[0])
    if original_num_voxels > fixed_k:
        raise ValueError(f"num_voxels={original_num_voxels} exceeds fixed_k={fixed_k}")

    def _pad_first_dim(tensor: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
        shape = list(tensor.shape)
        shape[0] = fixed_k
        out = torch.full(tuple(shape), fill_value, dtype=tensor.dtype, device=tensor.device)
        if original_num_voxels:
            out[:original_num_voxels].copy_(tensor)
        return out

    padded_valid = torch.zeros((fixed_k,), dtype=valid_voxel_mask.dtype, device=valid_voxel_mask.device)
    if original_num_voxels:
        padded_valid[:original_num_voxels] = 1
    return (
        (
            _pad_first_dim(voxel_features),
            _pad_first_dim(voxel_coords),
            _pad_first_dim(voxel_num_points),
            pairwise_t_matrix,
            padded_valid,
        ),
        original_num_voxels,
        fixed_k,
    )


def _tensor_output_names(outputs: dict[str, Any]) -> list[str]:
    preferred = ["cls_preds", "reg_preds", "dir_preds"]
    names = [name for name in preferred if name in outputs and torch.is_tensor(outputs[name])]
    if names:
        return names
    names = [name for name, value in outputs.items() if torch.is_tensor(value)]
    if not names:
        raise RuntimeError("Model forward produced no tensor outputs suitable for ONNX export.")
    return names


def _dynamic_axes(output_names: list[str], export_mode: str = "fixed_static") -> dict[str, dict[int, str]]:
    if export_mode == "dynamic_agent_dim":
        return dynamic_agent_dynamic_axes(output_names)
    if export_mode == "padded_agent_static":
        return padded_agent_dynamic_axes(output_names)
    if export_mode == "fixed_k_scatter_plugin":
        return fixed_k_scatter_plugin_dynamic_axes(output_names)
    if export_mode == "dynamic_agent_dim_fixed_k_scatter_plugin":
        return dynamic_fixed_k_scatter_plugin_dynamic_axes(output_names)
    axes = {
        "voxel_features": {0: "num_voxels"},
        "voxel_coords": {0: "num_voxels"},
        "voxel_num_points": {0: "num_voxels"},
        "record_len": {0: "batch"},
        "pairwise_t_matrix": {0: "batch", 1: "max_cav", 2: "max_cav"},
    }
    for name in output_names:
        axes[name] = {0: "batch"}
    return axes


def _profile_shapes(tensors: tuple[torch.Tensor, ...], max_cav: int, hypes: dict[str, Any]) -> dict[str, Any]:
    return _profile_shapes_for_inputs(INPUT_NAMES, tensors, max_cav, hypes)


def _profile_shapes_for_inputs(
    input_names: list[str],
    tensors: tuple[torch.Tensor, ...],
    max_cav: int,
    hypes: dict[str, Any],
) -> dict[str, Any]:
    shapes = {name: list(tensor.shape) for name, tensor in zip(input_names, tensors)}
    max_voxels = int(
        hypes.get("heter", {})
        .get("modality_setting", {})
        .get("m1", {})
        .get("preprocess", {})
        .get("args", {})
        .get("max_voxel_test", max(shapes.get("voxel_features", [1])[0], 1))
    )
    profiles: dict[str, Any] = {}
    for name, shape in shapes.items():
        min_shape = list(shape)
        opt_shape = list(shape)
        max_shape = list(shape)
        if name in {"voxel_features", "voxel_coords", "voxel_num_points"}:
            min_shape[0] = 1
            max_shape[0] = max(int(shape[0]), max_voxels * max(1, max_cav))
        if name == "record_len":
            min_shape[0] = 1
            opt_shape[0] = 1
            max_shape[0] = 1
        if name == "valid_agent_mask":
            min_shape = [1, max_cav]
            opt_shape = [1, max_cav]
            max_shape = [1, max_cav]
        if name == "pairwise_t_matrix":
            min_shape[0] = 1
            opt_shape[0] = 1
            max_shape[0] = 1
            min_shape[1] = min(min_shape[1], 1)
            min_shape[2] = min(min_shape[2], 1)
            max_shape[1] = max(max_shape[1], max_cav)
            max_shape[2] = max(max_shape[2], max_cav)
        profiles[name] = {"min": min_shape, "opt": opt_shape, "max": max_shape}
    return profiles


def _onnx_filename_for_export_mode(mode: str) -> str:
    if mode == "dynamic_agent_dim":
        return "lidar_pyramid_dynamic_agent_dim_fp32_dynamic.onnx"
    if mode == "padded_agent_static":
        return "lidar_pyramid_padded_agent_static_fp32_dynamic.onnx"
    if mode == "fixed_k_scatter_plugin":
        return "lidar_pyramid_fixed_k_scatter_plugin_fp32_dynamic.onnx"
    if mode == "dynamic_agent_dim_fixed_k_scatter_plugin":
        return "lidar_pyramid_dynamic_agent_dim_fixed_k_scatter_plugin_fp32_dynamic.onnx"
    if mode == "original":
        return "lidar_pyramid_original_fp32_dynamic.onnx"
    return "lidar_pyramid_fp32_dynamic.onnx"


def _wrapper_source_for_export_mode(mode: str) -> str | None:
    if mode == "fixed_static":
        return SOURCE_FIXED_WRAPPER
    if mode == "dynamic_agent_dim":
        return SOURCE_DYNAMIC_AGENT_WRAPPER
    if mode == "padded_agent_static":
        return SOURCE_PADDED_WRAPPER
    if mode == "fixed_k_scatter_plugin":
        return SOURCE_FIXED_K_SCATTER_PLUGIN_WRAPPER
    if mode == "dynamic_agent_dim_fixed_k_scatter_plugin":
        return SOURCE_DYNAMIC_FIXED_K_SCATTER_PLUGIN_WRAPPER
    return None


def _special_ops_from_export_error(error_text: str) -> dict[str, Any]:
    report = {
        "GridSample": [],
        "AffineGrid": [],
        "Scatter": [],
        "Gather": [],
        "NonZero": [],
        "Inverse": [],
        "aten_ops": [],
        "org_pytorch_ops": [],
        "unsupported_ops": [],
    }
    lowered = error_text.lower()
    if "affine_grid" in lowered or "affine_grid_generator" in lowered:
        report["AffineGrid"].append({"name": "aten::affine_grid_generator", "source": "torch.onnx.export_error"})
    if "grid_sampler" in lowered or "grid_sample" in lowered:
        report["GridSample"].append({"name": "aten::grid_sampler", "source": "torch.onnx.export_error"})
    if "scatter" in lowered:
        report["Scatter"].append({"name": "scatter", "source": "torch.onnx.export_error"})
    if "gather" in lowered:
        report["Gather"].append({"name": "gather", "source": "torch.onnx.export_error"})
    if "nonzero" in lowered:
        report["NonZero"].append({"name": "nonzero", "source": "torch.onnx.export_error"})
    if "inverse" in lowered or "solve" in lowered:
        report["Inverse"].append({"name": "inverse_or_solve", "source": "torch.onnx.export_error"})
    if "unsupported" in lowered or "not supported" in lowered:
        report["unsupported_ops"].append({"source": "torch.onnx.export", "line": error_text.splitlines()[0] if error_text else ""})
    if "aten::" in lowered:
        report["aten_ops"].append({"source": "torch.onnx.export_error", "line": error_text.splitlines()[0] if error_text else ""})
    if "org.pytorch" in lowered:
        report["org_pytorch_ops"].append({"source": "torch.onnx.export_error", "line": error_text.splitlines()[0] if error_text else ""})
    return report


@contextmanager
def _patch_default_domain_plugin_checker(enabled: bool):
    check_onnx_proto = getattr(torch._C, "_check_onnx_proto", None)
    if not enabled or check_onnx_proto is None:
        yield {"patched": False}
        return
    torch._C._check_onnx_proto = lambda proto: None
    try:
        yield {"patched": True}
    finally:
        torch._C._check_onnx_proto = check_onnx_proto


@contextmanager
def _patch_bev_warp_for_export(mode: str):
    if mode != "exportable_grid":
        yield {"patched": False, "mode": mode}
        return
    import opencood.models.fuse_modules.pyramid_fuse as pyramid_fuse
    import opencood.models.sub_modules.torch_transformation_utils as ttu

    original_ttu = ttu.warp_affine_simple
    original_pyramid = pyramid_fuse.warp_affine_simple
    ttu.warp_affine_simple = warp_affine_simple_exportable
    pyramid_fuse.warp_affine_simple = warp_affine_simple_exportable
    try:
        yield {"patched": True, "mode": mode}
    finally:
        ttu.warp_affine_simple = original_ttu
        pyramid_fuse.warp_affine_simple = original_pyramid


@contextmanager
def _patch_pillar_vfe_for_export(mode: str):
    if mode != "explicit_squeeze":
        yield {"patched": False, "mode": mode}
        return
    from opencood.models.sub_modules.pillar_vfe import PillarVFE

    original_forward = PillarVFE.forward
    PillarVFE.forward = pillar_vfe_forward_explicit_squeeze
    try:
        yield {"patched": True, "mode": mode}
    finally:
        PillarVFE.forward = original_forward


def export_lidar_pyramid_onnx(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root) if args.output_root else create_quant_deploy_run_dirs(args.output_dir, args.run_name, args.overwrite)
    log_path = dirs["logs_export"] / "export_onnx.log"
    export_mode = args.pyramid_forward_export_mode
    onnx_path = dirs["onnx_fp32"] / _onnx_filename_for_export_mode(export_mode)
    compatibility_onnx_path = dirs["onnx_fp32"] / "lidar_pyramid_fp32_dynamic.onnx"
    hypes_yaml = Path(args.hypes_yaml).expanduser()
    checkpoint = Path(args.checkpoint).expanduser()
    summary: dict[str, Any] = {
        "success": False,
        "error": None,
        "export_boundary": "full_model",
        "opset": args.opset,
        "onnx_path": str(onnx_path),
        "log_path": str(log_path),
    }
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        try:
            _add_paths(args.heal_repo)
            env_report = collect_env_report(trt_root=getattr(args, "trt_root", None), explicit_trtexec=getattr(args, "trtexec_path", None))
            save_json(env_report, dirs["debug"] / "env_report.json")
            print(f"hypes_yaml={hypes_yaml}")
            print(f"checkpoint={checkpoint}")
            print(f"device={args.device}")
            print(f"bev_warp_export_mode={args.bev_warp_export_mode}")
            print(f"pillar_vfe_export_fix={args.pillar_vfe_export_fix}")
            print(f"pyramid_forward_export_mode={export_mode}")
            if not hypes_yaml.exists():
                raise FileNotFoundError(f"hypes_yaml is required and does not exist: {hypes_yaml}")
            hypes = _load_hypes(hypes_yaml, args.heal_repo)
            device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
            model = _load_model(hypes, checkpoint, device)
            modality = _infer_modality(model)
            num_pyramid_scales = int(getattr(getattr(model, "pyramid_backbone", None), "num_levels", 0) or 0)
            try:
                sample_max_cav = int(args.fixed_num_agents or args.max_cav) if export_mode == "dynamic_agent_dim_fixed_k_scatter_plugin" else int(args.max_cav)
                sample, sample_source = _real_sample_for_export_mode(hypes, device, export_mode=export_mode, max_cav=sample_max_cav)
                summary["sample_source"] = sample_source
            except Exception:
                if not args.allow_synthetic_fallback:
                    raise
                print("Real sample loading failed; using synthetic fallback because --allow_synthetic_fallback was set.")
                print(traceback.format_exc())
                sample = _synthetic_sample(model, device, args.max_cav)
                summary["sample_source"] = "synthetic_fallback"
            original_tensors, agent_modality_list = _extract_inputs(sample, modality)
            input_names = _input_names_for_export_mode(export_mode)
            tensors = _prepare_export_tensors(original_tensors, export_mode=export_mode, max_cav=args.max_cav)
            fixed_k_padding_report = None
            if export_mode == "fixed_k_scatter_plugin":
                fixed_k = int(args.fixed_k or int(tensors[0].shape[0]))
                tensors, original_num_voxels, fixed_k = _pad_fixed_k_export_tensors(tensors, fixed_k)
                fixed_k_padding_report = {
                    "original_num_voxels": original_num_voxels,
                    "fixed_K": fixed_k,
                    "valid_voxel_count": int(tensors[-1].sum().item()),
                    "valid_voxel_mask_shape": list(tensors[-1].shape),
                    "ap_valid": True,
                    "whether_valid_mask_used": True,
                }
                save_json(fixed_k_padding_report, dirs["debug"] / "fixed_k_scatter_plugin_export_padding_report.json")
            elif export_mode == "dynamic_agent_dim_fixed_k_scatter_plugin":
                fixed_k = int(args.fixed_k or int(tensors[0].shape[0]))
                tensors, original_num_voxels, fixed_k = _pad_dynamic_fixed_k_export_tensors(tensors, fixed_k)
                fixed_k_padding_report = {
                    "original_num_voxels": original_num_voxels,
                    "fixed_K": fixed_k,
                    "fixed_num_agents": int(args.fixed_num_agents or tensors[3].shape[1]),
                    "valid_voxel_count": int(tensors[-1].sum().item()),
                    "valid_voxel_mask_shape": list(tensors[-1].shape),
                    "ap_valid": True,
                    "whether_valid_mask_used": True,
                    "true_dynamic_agent_dim": False,
                    "per_N_engine_routing": True,
                }
                save_json(fixed_k_padding_report, dirs["debug"] / "dynamic_agent_dim_fixed_k_scatter_plugin_export_padding_report.json")
            with torch.no_grad():
                raw_output = model(sample)
            output_names = _tensor_output_names(raw_output)
            if export_mode == "fixed_static":
                wrapper = FixedLidarPyramidExportWrapper(model, modality, output_names).to(device).eval()
                fixed_wrapper_report = check_fixed_lidar_pyramid_wrapper_equivalence(
                    wrapper,
                    original_tensors,
                    raw_output,
                    output_names,
                )
                save_json(fixed_wrapper_report, dirs["debug"] / "fixed_pyramid_forward_report.json")
                if not fixed_wrapper_report["same_shape"]:
                    raise RuntimeError(f"Fixed pyramid export wrapper changed output shape: {fixed_wrapper_report}")
                max_error = max(
                    (
                        item.get("max_abs_error") or 0.0
                        for item in fixed_wrapper_report.get("outputs", {}).values()
                    ),
                    default=0.0,
                )
                if max_error > 1.0e-3:
                    raise RuntimeError(f"Fixed pyramid export wrapper equivalence check failed: {fixed_wrapper_report}")
            elif export_mode == "dynamic_agent_dim":
                wrapper = ExportableLidarPyramidDynamicAgent(model, modality, output_names).to(device).eval()
                fixed_wrapper_report = check_dynamic_agent_wrapper_equivalence(wrapper, tensors, raw_output, output_names)
                save_json(fixed_wrapper_report, dirs["debug"] / "dynamic_agent_dim_wrapper_equivalence.json")
                if not fixed_wrapper_report["same_shape"]:
                    raise RuntimeError(f"Dynamic-agent export wrapper changed output shape: {fixed_wrapper_report}")
                max_error = max((item.get("max_abs_error") or 0.0 for item in fixed_wrapper_report.get("outputs", {}).values()), default=0.0)
                if max_error > 1.0e-3:
                    raise RuntimeError(f"Dynamic-agent export wrapper equivalence check failed: {fixed_wrapper_report}")
            elif export_mode == "padded_agent_static":
                wrapper = ExportableLidarPyramidPaddedAgent(model, modality, output_names, max_cav=args.max_cav).to(device).eval()
                fixed_wrapper_report = check_padded_agent_wrapper_equivalence(wrapper, tensors, raw_output, output_names)
                save_json(fixed_wrapper_report, dirs["debug"] / "padded_agent_static_wrapper_equivalence.json")
                if not fixed_wrapper_report["same_shape"]:
                    raise RuntimeError(f"Padded-agent export wrapper changed output shape: {fixed_wrapper_report}")
                max_error = max((item.get("max_abs_error") or 0.0 for item in fixed_wrapper_report.get("outputs", {}).values()), default=0.0)
                if max_error > 1.0e-3:
                    raise RuntimeError(f"Padded-agent export wrapper equivalence check failed: {fixed_wrapper_report}")
            elif export_mode == "fixed_k_scatter_plugin":
                wrapper = ExportableLidarPyramidFixedKScatterPlugin(model, modality, output_names, max_cav=args.max_cav).to(device).eval()
                fixed_wrapper_report = check_fixed_k_scatter_plugin_wrapper_equivalence(wrapper, tensors, raw_output, output_names)
                save_json(fixed_wrapper_report, dirs["debug"] / "fixed_k_scatter_plugin_wrapper_equivalence.json")
                if not fixed_wrapper_report["same_shape"]:
                    raise RuntimeError(f"Fixed-K scatter plugin export wrapper changed output shape: {fixed_wrapper_report}")
                max_error = max((item.get("max_abs_error") or 0.0 for item in fixed_wrapper_report.get("outputs", {}).values()), default=0.0)
                if max_error > 2.0e-2:
                    raise RuntimeError(f"Fixed-K scatter plugin export wrapper equivalence check failed: {fixed_wrapper_report}")
            elif export_mode == "dynamic_agent_dim_fixed_k_scatter_plugin":
                fixed_num_agents = int(args.fixed_num_agents or tensors[3].shape[1])
                wrapper = ExportableLidarPyramidDynamicFixedKScatterPlugin(
                    model,
                    modality,
                    output_names,
                    fixed_num_agents=fixed_num_agents,
                ).to(device).eval()
                fixed_wrapper_report = check_dynamic_fixed_k_scatter_plugin_wrapper_equivalence(wrapper, tensors, raw_output, output_names)
                save_json(
                    fixed_wrapper_report,
                    dirs["debug"] / f"dynamic_agent_dim_N{fixed_num_agents}_fixed_k_scatter_plugin_wrapper_equivalence.json",
                )
                if not fixed_wrapper_report["same_shape"]:
                    raise RuntimeError(f"Dynamic fixed-K scatter plugin export wrapper changed output shape: {fixed_wrapper_report}")
                max_error = max((item.get("max_abs_error") or 0.0 for item in fixed_wrapper_report.get("outputs", {}).values()), default=0.0)
                if max_error > 2.0e-2:
                    raise RuntimeError(f"Dynamic fixed-K scatter plugin export wrapper equivalence check failed: {fixed_wrapper_report}")
            else:
                wrapper = LidarPyramidDeployWrapper(model, modality, agent_modality_list, output_names).to(device).eval()
                fixed_wrapper_report = None
            pillar_vfe_report = None
            if args.pillar_vfe_export_fix == "explicit_squeeze":
                pillar_vfe = getattr(getattr(model, f"encoder_{modality}", None), "pillar_vfe", None)
                if pillar_vfe is None:
                    raise RuntimeError(f"Could not locate encoder_{modality}.pillar_vfe for export fix check.")
                lidar_inputs = sample[f"inputs_{modality}"]
                pillar_vfe_report = check_pillar_vfe_export_fix_equivalence(
                    pillar_vfe,
                    {
                        "voxel_features": lidar_inputs["voxel_features"],
                        "voxel_coords": lidar_inputs["voxel_coords"],
                        "voxel_num_points": lidar_inputs["voxel_num_points"],
                    },
                    original_forward=pillar_vfe.forward,
                )
                save_json(pillar_vfe_report, dirs["debug"] / "pillar_vfe_export_fix_report.json")
                if not pillar_vfe_report["same_shape"]:
                    raise RuntimeError(f"PillarVFE export fix changed output shape: {pillar_vfe_report}")
                if pillar_vfe_report["max_abs_error"] is not None and pillar_vfe_report["max_abs_error"] > 1e-5:
                    raise RuntimeError(f"PillarVFE export fix equivalence check failed: {pillar_vfe_report}")
            bev_warp_report = None
            if args.bev_warp_export_mode == "exportable_grid":
                import opencood.models.sub_modules.torch_transformation_utils as ttu

                check_device = device
                check_src = torch.randn(2, 3, 6, 8, device=check_device)
                check_theta = torch.tensor(
                    [
                        [[1.0, 0.0, 0.1], [0.0, 1.0, -0.2]],
                        [[0.9, 0.2, 0.0], [-0.2, 0.9, 0.1]],
                    ],
                    dtype=check_src.dtype,
                    device=check_device,
                )
                bev_warp_report = check_exportable_bev_warp_equivalence(
                    src=check_src,
                    theta=check_theta,
                    dsize=(6, 8),
                    align_corners=False,
                    original_warp=ttu.warp_affine_simple,
                )
                save_json(bev_warp_report, dirs["debug"] / "bev_warp_equivalence_report.json")
                if bev_warp_report["max_abs_error"] > 1e-4 or bev_warp_report["relative_error"] > 1e-4:
                    raise RuntimeError(f"exportable BEV warp equivalence check failed: {bev_warp_report}")
            dynamic_axes = _dynamic_axes(output_names, export_mode)
            profiles = _profile_shapes_for_inputs(input_names, tensors, args.max_cav, hypes)
            save_json(dynamic_axes, dirs["configs"] / "dynamic_axes.json")
            save_json(profiles, dirs["configs"] / "profile_shapes.json")
            save_json(
                {
                    "input_names": input_names,
                    "output_names": output_names,
                    "agent_modality_list_length": len(agent_modality_list),
                    "modality_name": modality,
                    "pyramid_forward_export_mode": export_mode,
                },
                dirs["onnx_fp32"] / "input_output_names.json",
            )
            save_json(
                {
                    "hypes_yaml": str(hypes_yaml),
                    "checkpoint": str(checkpoint),
                    "opset": args.opset,
                    "max_cav": args.max_cav,
                    "dynamic_axes": dynamic_axes,
                    "export_boundary": "full_model",
                    "sample_source": summary.get("sample_source"),
                    "bev_warp_export_mode": args.bev_warp_export_mode,
                    "pillar_vfe_export_fix": args.pillar_vfe_export_fix,
                    "pyramid_forward_export_mode": export_mode,
                    "export_forward_mode": export_mode,
                    "is_export_specialized_wrapper": export_mode in {"fixed_static", "dynamic_agent_dim", "padded_agent_static", "fixed_k_scatter_plugin", "dynamic_agent_dim_fixed_k_scatter_plugin"},
                    "is_original_forward": export_mode == "original",
                    "num_pyramid_scales": num_pyramid_scales,
                    "fixed_pyramid_wrapper_source": SOURCE_FIXED_WRAPPER if export_mode == "fixed_static" else None,
                    "export_wrapper_source": _wrapper_source_for_export_mode(export_mode),
                    "valid_for_multi_agent_deployment": export_mode in {"dynamic_agent_dim", "padded_agent_static", "fixed_k_scatter_plugin", "dynamic_agent_dim_fixed_k_scatter_plugin"},
                    "fixed_k_scatter_plugin": export_mode == "fixed_k_scatter_plugin",
                    "dynamic_agent_dim_fixed_k_scatter_plugin": export_mode == "dynamic_agent_dim_fixed_k_scatter_plugin",
                    "fixed_k_padding": fixed_k_padding_report,
                },
                dirs["configs"] / "deploy_config.json",
            )
            save_json(
                {
                    "hypes_yaml": str(hypes_yaml),
                    "checkpoint": str(checkpoint),
                    "output_root": str(dirs["output_root"]),
                    "device": args.device,
                    "opset": args.opset,
                    "max_cav": args.max_cav,
                    "bev_warp_export_mode": args.bev_warp_export_mode,
                    "pillar_vfe_export_fix": args.pillar_vfe_export_fix,
                    "pyramid_forward_export_mode": export_mode,
                    "export_forward_mode": export_mode,
                    "is_export_specialized_wrapper": export_mode in {"fixed_static", "dynamic_agent_dim", "padded_agent_static", "fixed_k_scatter_plugin", "dynamic_agent_dim_fixed_k_scatter_plugin"},
                    "is_original_forward": export_mode == "original",
                    "num_pyramid_scales": num_pyramid_scales,
                    "fixed_pyramid_wrapper_source": SOURCE_FIXED_WRAPPER if export_mode == "fixed_static" else None,
                    "export_wrapper_source": _wrapper_source_for_export_mode(export_mode),
                    "valid_for_multi_agent_deployment": export_mode in {"dynamic_agent_dim", "padded_agent_static", "fixed_k_scatter_plugin", "dynamic_agent_dim_fixed_k_scatter_plugin"},
                    "fixed_k_scatter_plugin": export_mode == "fixed_k_scatter_plugin",
                    "dynamic_agent_dim_fixed_k_scatter_plugin": export_mode == "dynamic_agent_dim_fixed_k_scatter_plugin",
                    "fixed_k_padding": fixed_k_padding_report,
                    "trt_root": getattr(args, "trt_root", None),
                    "trtexec_path": getattr(args, "trtexec_path", None),
                },
                dirs["configs"] / "run_config.json",
            )
            if hypes_yaml.exists():
                shutil.copyfile(hypes_yaml, dirs["configs"] / "model_config_snapshot.yaml")

            with _patch_bev_warp_for_export(args.bev_warp_export_mode) as bev_patch_info:
                with _patch_pillar_vfe_for_export(args.pillar_vfe_export_fix) as pillar_patch_info:
                    with _patch_default_domain_plugin_checker(export_mode in {"fixed_k_scatter_plugin", "dynamic_agent_dim_fixed_k_scatter_plugin"}) as plugin_checker_patch:
                        print(f"bev_warp_patch={bev_patch_info}")
                        print(f"pillar_vfe_patch={pillar_patch_info}")
                        print(f"plugin_checker_patch={plugin_checker_patch}")
                        torch.onnx.export(
                            wrapper,
                            tensors,
                            str(onnx_path),
                            input_names=input_names,
                            output_names=output_names,
                            dynamic_axes=dynamic_axes,
                            opset_version=args.opset,
                            do_constant_folding=True,
                        )
            if onnx_path != compatibility_onnx_path:
                shutil.copyfile(onnx_path, compatibility_onnx_path)
            try:
                import onnx

                onnx_model = onnx.load(str(onnx_path))
                if export_mode in {"fixed_k_scatter_plugin", "dynamic_agent_dim_fixed_k_scatter_plugin"}:
                    (dirs["onnx_fp32"] / "onnx_check.log").write_text(
                        "ONNX standard checker skipped because this graph contains the TensorRT custom op PointPillarScatterTRT.\n",
                        encoding="utf-8",
                    )
                else:
                    onnx.checker.check_model(onnx_model)
                    (dirs["onnx_fp32"] / "onnx_check.log").write_text("ONNX basic check passed.\n", encoding="utf-8")
            except Exception:
                (dirs["onnx_fp32"] / "onnx_check.log").write_text(traceback.format_exc(), encoding="utf-8")
                raise
            graph_info = onnx_graph_info(onnx_path)
            special_ops = detect_special_ops_in_onnx(onnx_path)
            save_json(graph_info, dirs["onnx_fp32"] / "onnx_graph_info.json")
            save_json(special_ops, dirs["debug"] / "special_ops_report.json")
            save_json(special_ops.get("unsupported_ops", []), dirs["debug"] / "unsupported_ops.json")
            save_json([], dirs["debug"] / "failed_nodes.json")
            save_json({"input_shapes": {name: list(t.shape) for name, t in zip(input_names, tensors)}}, dirs["debug"] / "tensor_shapes_report.json")
            summary.update(
                {
                    "success": True,
                    "error": None,
                    "detected_special_ops": special_ops,
                    "bev_warp_export_mode": args.bev_warp_export_mode,
                    "bev_warp_equivalence": bev_warp_report,
                    "pillar_vfe_export_fix": args.pillar_vfe_export_fix,
                    "pillar_vfe_export_fix_report": pillar_vfe_report,
                    "pyramid_forward_export_mode": export_mode,
                    "export_forward_mode": export_mode,
                    "is_export_specialized_wrapper": export_mode in {"fixed_static", "dynamic_agent_dim", "padded_agent_static", "fixed_k_scatter_plugin", "dynamic_agent_dim_fixed_k_scatter_plugin"},
                    "is_original_forward": export_mode == "original",
                    "num_pyramid_scales": num_pyramid_scales,
                    "sequence_ops_removed": int(special_ops.get("sequence_op_count") or 0) == 0,
                    "fixed_pyramid_wrapper_source": SOURCE_FIXED_WRAPPER if export_mode == "fixed_static" else None,
                    "export_wrapper_source": _wrapper_source_for_export_mode(export_mode),
                    "valid_for_multi_agent_deployment": export_mode in {"dynamic_agent_dim", "padded_agent_static", "fixed_k_scatter_plugin", "dynamic_agent_dim_fixed_k_scatter_plugin"},
                    "fixed_k_scatter_plugin": export_mode == "fixed_k_scatter_plugin",
                    "dynamic_agent_dim_fixed_k_scatter_plugin": export_mode == "dynamic_agent_dim_fixed_k_scatter_plugin",
                    "fixed_k_padding": fixed_k_padding_report,
                    "fixed_pyramid_forward_report": fixed_wrapper_report,
                    "env_report": env_report,
                    "onnx_path": str(onnx_path),
                    "compatibility_onnx_path": str(compatibility_onnx_path),
                }
            )
        except Exception as exc:
            tb = traceback.format_exc()
            special_ops = _special_ops_from_export_error(str(exc) + "\n" + tb)
            summary.update(
                {
                    "success": False,
                    "error": str(exc),
                    "traceback": tb,
                    "detected_special_ops": special_ops,
                    "bev_warp_export_mode": getattr(args, "bev_warp_export_mode", None),
                    "pillar_vfe_export_fix": getattr(args, "pillar_vfe_export_fix", None),
                    "pyramid_forward_export_mode": getattr(args, "pyramid_forward_export_mode", None),
                    "env_report": locals().get("env_report"),
                }
            )
            save_json(special_ops, dirs["debug"] / "special_ops_report.json")
            save_json(special_ops.get("unsupported_ops", []), dirs["debug"] / "unsupported_ops.json")
            save_json(
                [{"source": "torch.onnx.export", "error": str(exc)}],
                dirs["debug"] / "failed_nodes.json",
            )
            (dirs["onnx_fp32"] / "onnx_check.log").write_text(
                "ONNX export failed before basic check.\n" + str(exc) + "\n",
                encoding="utf-8",
            )
            print(traceback.format_exc())
        finally:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(buffer.getvalue(), encoding="utf-8")
            save_json(summary, dirs["summary"] / "summary_export.json")
            save_json({"dirs": dirs_for_summary(dirs)}, dirs["configs"] / "output_dirs.json")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = export_lidar_pyramid_onnx(args)
    print(result["onnx_path"])
    if not result["success"]:
        print(f"ONNX export failed: {result['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
