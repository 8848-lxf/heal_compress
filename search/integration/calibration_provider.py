"""Real Fisher and Q/DQ calibration providers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from ..hashing import canonical_json_hash
from ..proxy.fisher_proxy import FisherStatistics
from .data_provider import build_dataset_and_loader, iter_limited, move_batch_to_device


QDQ_CALIBRATION_SEMANTICS_VERSION = "onnx-bn-fold-fixedk-entropy-v6-exact-tensor-manifest"
TENSORRT_ENTROPY_CALIBRATION_SEMANTICS_VERSION = (
    "onnx-trt-entropycalibration2-fixedk-v1-exact-tensor-manifest"
)
FIXED_K_CALIBRATION_INPUT_NAMES = (
    "voxel_features",
    "voxel_coords",
    "voxel_num_points",
    "pairwise_t_matrix",
    "valid_voxel_mask",
)
BASELINE_FIXED_K_CALIBRATION_INPUT_NAMES = (
    *FIXED_K_CALIBRATION_INPUT_NAMES,
    "agent_mask",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_k_calibration_npz_manifest_identity(
    manifest_path: str | Path,
    *,
    num_batches: int,
    fixed_k: int,
    input_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return a content-addressed identity for preprocessed train calibration tensors."""

    expected_inputs = tuple(str(value) for value in (input_names or FIXED_K_CALIBRATION_INPUT_NAMES))
    if not expected_inputs or len(expected_inputs) != len(set(expected_inputs)):
        raise RuntimeError(f"calibration_npz_expected_input_names_invalid:{expected_inputs}")
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"calibration_npz_manifest_missing:{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = list(payload.get("files", []) or [])
    if len(files) != int(num_batches):
        raise RuntimeError(f"calibration_npz_manifest_count_mismatch:{len(files)}!={int(num_batches)}")
    if int(payload.get("fixed_K", -1)) != int(fixed_k):
        raise RuntimeError(f"calibration_npz_fixed_k_mismatch:{payload.get('fixed_K')}!={int(fixed_k)}")
    if str(payload.get("calibration_split", "")) != "train":
        raise RuntimeError(f"calibration_npz_split_not_train:{payload.get('calibration_split')}")
    if str(payload.get("strategy", "")) != "single_engine_maxK":
        raise RuntimeError(f"calibration_npz_strategy_mismatch:{payload.get('strategy')}")
    manifest_input_names = [str(value) for value in payload.get("input_names", [])]
    if tuple(manifest_input_names) != expected_inputs:
        raise RuntimeError(
            f"calibration_npz_input_names_mismatch:{manifest_input_names}!={list(expected_inputs)}"
        )
    file_rows = [
        {
            "index": int(index),
            "name": str(row.get("name") or Path(str(row.get("path", ""))).name),
            "sha256": str(row.get("sha256", "")),
            "bytes": int(row.get("bytes", 0) or 0),
            "path": str(row.get("path", "")),
        }
        for index, row in enumerate(files)
    ]
    if any(not row["name"] or len(row["sha256"]) != 64 for row in file_rows):
        raise RuntimeError("calibration_npz_manifest_file_provenance_incomplete")
    tensor_manifest_hash = canonical_json_hash(
        [{key: row[key] for key in ("index", "name", "sha256", "bytes")} for row in file_rows]
    )
    return {
        "source": "preprocessed_train_npz_manifest",
        "manifest_path": str(path),
        "manifest_sha256": _sha256_file(path),
        "tensor_manifest_hash": tensor_manifest_hash,
        "sample_count": len(file_rows),
        "fixed_k": int(fixed_k),
        "strategy": "single_engine_maxK",
        "calibration_split": "train",
        "input_names": list(expected_inputs),
        "train_dataset_indices": [int(value) for value in payload.get("train_dataset_indices", [])],
        "files": file_rows,
    }


def load_fixed_k_calibration_npz_batches(
    manifest_path: str | Path,
    *,
    num_batches: int,
    fixed_k: int,
    device: torch.device,
    input_names: Sequence[str] | None = None,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    """Load and verify exact preprocessed tensors in manifest order."""

    import numpy as np

    identity = fixed_k_calibration_npz_manifest_identity(
        manifest_path,
        num_batches=num_batches,
        fixed_k=fixed_k,
        input_names=input_names,
    )
    expected_inputs = tuple(str(value) for value in identity["input_names"])
    manifest = Path(identity["manifest_path"])
    batches: list[dict[str, torch.Tensor]] = []
    verified_files: list[dict[str, Any]] = []
    for row in identity["files"]:
        raw_path = Path(str(row["path"])).expanduser()
        candidates = [raw_path, manifest.parent / str(row["name"]), manifest.parent / raw_path.name]
        source = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
        if source is None:
            raise RuntimeError(f"calibration_npz_file_missing:{row['index']}:{row['name']}")
        size = int(source.stat().st_size)
        digest = _sha256_file(source)
        if int(row["bytes"]) > 0 and size != int(row["bytes"]):
            raise RuntimeError(f"calibration_npz_file_size_mismatch:{row['name']}:{size}!={row['bytes']}")
        if digest != str(row["sha256"]):
            raise RuntimeError(f"calibration_npz_file_hash_mismatch:{row['name']}:{digest}!={row['sha256']}")
        with np.load(source) as values:
            missing = [name for name in expected_inputs if name not in values.files]
            if missing:
                raise RuntimeError(f"calibration_npz_inputs_missing:{row['name']}:{missing}")
            arrays = {name: np.ascontiguousarray(values[name]) for name in expected_inputs}
        for name in ("voxel_features", "voxel_coords", "voxel_num_points", "valid_voxel_mask"):
            if int(arrays[name].shape[0]) != int(fixed_k):
                raise RuntimeError(
                    f"calibration_npz_fixed_k_tensor_mismatch:{row['name']}:{name}:{arrays[name].shape[0]}!={int(fixed_k)}"
                )
        if "agent_mask" in arrays:
            pairwise = tuple(int(value) for value in arrays["pairwise_t_matrix"].shape)
            if len(pairwise) != 5 or pairwise[1] != 2 or pairwise[2] != 2:
                raise RuntimeError(
                    f"baseline_calibration_requires_static_two_agent_tensors:{row['name']}:{pairwise}"
                )
            agent_mask = arrays["agent_mask"]
            agent_mask_shape = tuple(int(value) for value in agent_mask.shape)
            if len(pairwise) != 5 or agent_mask_shape != (1, pairwise[1]) or pairwise[1] != pairwise[2]:
                raise RuntimeError(
                    f"calibration_npz_agent_mask_shape_mismatch:{row['name']}:{agent_mask_shape}:{pairwise}"
                )
            if (
                not np.all(np.isfinite(agent_mask))
                or not np.all((agent_mask == 0) | (agent_mask == 1))
                or float(agent_mask[0, 0]) != 1.0
                or float(agent_mask.sum()) < 1.0
            ):
                raise RuntimeError(
                    f"calibration_npz_agent_mask_values_invalid:{row['name']}:{agent_mask.tolist()}"
                )
        batches.append({name: torch.as_tensor(value, device=device) for name, value in arrays.items()})
        verified_files.append({"index": row["index"], "name": row["name"], "path": str(source), "size": size, "sha256": digest})
    return batches, {**identity, "files_verified": True, "verified_files": verified_files}


def parse_tensorrt_entropy_calibration_cache(cache_path: str | Path) -> dict[str, float]:
    """Parse a TensorRT EntropyCalibration2 cache without fuzzy tensor matching."""

    import struct

    path = Path(cache_path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"tensorrt_entropy_cache_missing:{path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or "EntropyCalibration2" not in lines[0]:
        raise RuntimeError(f"tensorrt_entropy_cache_header_invalid:{path}")
    values: dict[str, float] = {}
    for line in lines[1:]:
        if ": " not in line:
            continue
        tensor_name, encoded = line.rsplit(": ", 1)
        try:
            value = float(struct.unpack("!f", bytes.fromhex(encoded))[0])
        except (ValueError, struct.error):
            continue
        if math.isfinite(value) and value > 0.0:
            values[str(tensor_name)] = value
    if not values:
        raise RuntimeError(f"tensorrt_entropy_cache_contains_no_positive_scales:{path}")
    return values


def qdq_scales_from_tensorrt_entropy_cache(
    *,
    onnx_path: str | Path,
    origin_map: Any,
    module_paths: Sequence[str],
    cache_path: str | Path,
    weight_granularity: str = "per_channel",
    activation_scale_source: str = "fresh_TensorRT_IInt8EntropyCalibrator2_exact_tensor_match",
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Create production Q/DQ scales from exact TensorRT cache tensor names.

    Activation scales are accepted only when both the canonical weighted-node
    input and output names match cache entries exactly.  Weight scales always
    come from the final ONNX initializer and therefore never come from the
    implicit TensorRT cache.
    """

    import numpy as np
    import onnx
    from onnx import numpy_helper
    try:
        from quantization.precision.activation_boundary import resolve_activation_output_boundary
    except ImportError:
        from heal_compress.quantization.precision.activation_boundary import resolve_activation_output_boundary

    if str(weight_granularity) not in {"per_tensor", "per_channel"}:
        raise RuntimeError(f"unsupported_qdq_weight_granularity:{weight_granularity}")
    requested = [str(value) for value in module_paths]
    if len(requested) != len(set(requested)):
        raise RuntimeError("qdq_calibration_module_paths_not_unique")
    cache = parse_tensorrt_entropy_calibration_cache(cache_path)
    model = onnx.load(str(onnx_path))
    nodes = {str(node.name): node for node in model.graph.node}
    initializers = {str(row.name): numpy_helper.to_array(row) for row in model.graph.initializer}
    entries = {_field(row, "module_path"): row for row in _origin_entries(origin_map)}
    missing_entries = [name for name in requested if name not in entries]
    if missing_entries:
        raise RuntimeError(f"qdq_calibration_origin_entries_missing:{missing_entries}")

    scales: dict[str, dict[str, Any]] = {}
    exact_matches: list[dict[str, str]] = []
    for name in requested:
        entry = entries[name]
        node = nodes.get(str(_field(entry, "canonical_node_name")))
        if node is None:
            raise RuntimeError(f"qdq_calibration_onnx_node_missing:{name}")
        if not node.input or not node.output:
            raise RuntimeError(f"qdq_calibration_weighted_node_boundary_missing:{name}")
        input_tensor = str(node.input[0])
        output_boundary = resolve_activation_output_boundary(model, str(node.name))
        output_tensor = str(output_boundary["boundary_output_tensor"])
        missing_boundaries = [tensor for tensor in (input_tensor, output_tensor) if tensor not in cache]
        if missing_boundaries:
            raise RuntimeError(
                f"tensorrt_entropy_exact_tensor_match_missing:{name}:{missing_boundaries}"
            )
        initializer_name = str(_field(entry, "weight_initializer"))
        weight = initializers.get(initializer_name)
        if weight is None:
            raise RuntimeError(f"qdq_calibration_onnx_initializer_missing:{name}:{initializer_name}")
        weight_amax = float(np.max(np.abs(weight)))
        if not math.isfinite(weight_amax) or weight_amax <= 0.0:
            raise RuntimeError(f"qdq_calibration_nonpositive_or_nonfinite_weight_amax:{name}")
        weight_axis: int | None = None
        weight_scale: float | list[float]
        if str(weight_granularity) == "per_channel":
            if str(node.op_type) == "Conv":
                weight_axis = 0
            elif str(node.op_type) == "ConvTranspose":
                weight_axis = 1
            elif str(node.op_type) == "MatMul":
                weight_axis = 1
            elif str(node.op_type) == "Gemm":
                attributes = {
                    str(attr.name): int(attr.i)
                    for attr in node.attribute
                    if str(attr.name) == "transB"
                }
                weight_axis = 0 if attributes.get("transB", 0) else 1
            else:
                raise RuntimeError(f"qdq_calibration_per_channel_unsupported_op:{name}:{node.op_type}")
            if weight_axis >= weight.ndim:
                raise RuntimeError(
                    f"qdq_calibration_weight_axis_out_of_range:{name}:{weight_axis}:{list(weight.shape)}"
                )
            reduce_axes = tuple(axis for axis in range(weight.ndim) if axis != weight_axis)
            channel_amax = np.max(np.abs(weight), axis=reduce_axes)
            if not np.all(np.isfinite(channel_amax)) or np.any(channel_amax <= 0.0):
                raise RuntimeError(f"qdq_calibration_nonpositive_per_channel_weight_amax:{name}")
            weight_scale = (channel_amax / 127.0).astype(np.float32).tolist()
        else:
            weight_scale = weight_amax / 127.0
        scales[name] = {
            "activation_input_scale": float(cache[input_tensor]),
            "activation_output_scale": float(cache[output_tensor]),
            "activation_input_tensor": input_tensor,
            "activation_output_tensor": output_tensor,
            "activation_output_boundary_resolution": str(output_boundary["resolution"]),
            "activation_output_boundary_node": str(output_boundary["boundary_node_name"]),
            "activation_scale_source": str(activation_scale_source),
            "weight_scale": weight_scale,
            "weight_axis": weight_axis,
            "weight_granularity": str(weight_granularity),
            "weight_scale_shape": [len(weight_scale)] if isinstance(weight_scale, list) else [],
            "weight_scale_source": "final_folded_onnx_initializer",
        }
        exact_matches.extend(
            [
                {"module_path": name, "role": "input", "tensor": input_tensor},
                {"module_path": name, "role": "output", "tensor": output_tensor},
            ]
        )
    return scales, {
        "semantics_version": TENSORRT_ENTROPY_CALIBRATION_SEMANTICS_VERSION,
        "activation_calibration_method": "tensorrt_entropy_calibration2",
        "activation_scale_source": str(activation_scale_source),
        "cache_path": str(Path(cache_path).expanduser().resolve()),
        "cache_sha256": _sha256_file(Path(cache_path).expanduser().resolve()),
        "cache_positive_scale_count": len(cache),
        "exact_activation_scale_match_count": len(exact_matches),
        "exact_activation_scale_matches": exact_matches,
        "weight_granularity": str(weight_granularity),
    }


def build_tensorrt_entropy_calibration_cache_modelopt(
    *,
    onnx_path: str | Path,
    calibration_npz_manifest: str | Path,
    output_dir: str | Path,
    tensorrt_root: str | Path,
    plugin_path: str | Path,
    physical_gpu_id: int,
    num_batches: int,
    fixed_k: int = 29696,
    input_names: Sequence[str] | None = None,
    conda_env: str = "modelopt",
    force_rebuild: bool = True,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Fresh-build a TensorRT EntropyCalibration2 cache in an isolated subprocess."""

    import subprocess

    from .runtime_environment import modelopt_python_command, modelopt_subprocess_env

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    request_path = destination / "calibration_request.json"
    result_path = destination / "calibration_result.json"
    log_path = destination / "calibration_worker.log"
    cache_path = destination / "calibration.cache"
    engine_path = destination / "calibration.engine"
    identity = fixed_k_calibration_npz_manifest_identity(
        calibration_npz_manifest,
        num_batches=int(num_batches),
        fixed_k=int(fixed_k),
        input_names=input_names,
    )
    dependencies = {
        "onnx_sha256": _sha256_file(Path(onnx_path).expanduser().resolve()),
        "calibration_manifest_sha256": identity["manifest_sha256"],
        "calibration_tensor_manifest_hash": identity["tensor_manifest_hash"],
        "plugin_sha256": _sha256_file(Path(plugin_path).expanduser().resolve()),
        "fixed_k": int(fixed_k),
        "num_batches": int(num_batches),
        "input_names": list(identity["input_names"]),
        "semantics_version": TENSORRT_ENTROPY_CALIBRATION_SEMANTICS_VERSION,
    }
    if result_path.is_file() and not force_rebuild:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("dependencies") != dependencies:
            raise RuntimeError("tensorrt_entropy_calibration_cache_dependency_mismatch")
        if result.get("status") != "ok" or not cache_path.is_file():
            raise RuntimeError("tensorrt_entropy_calibration_cache_reuse_invalid")
        if _sha256_file(cache_path) != str(result.get("calibration_cache_sha256", "")):
            raise RuntimeError("tensorrt_entropy_calibration_cache_hash_mismatch")
        return {**result, "reused": True, "log_path": str(log_path)}
    existing = [path for path in (request_path, result_path, cache_path, engine_path, log_path) if path.exists()]
    if existing:
        raise RuntimeError(f"tensorrt_entropy_force_rebuild_destination_not_clean:{existing}")
    request = {
        "onnx_path": str(Path(onnx_path).expanduser().resolve()),
        "calibration_npz_manifest": str(Path(calibration_npz_manifest).expanduser().resolve()),
        "plugin_path": str(Path(plugin_path).expanduser().resolve()),
        "cache_path": str(cache_path.resolve()),
        "engine_path": str(engine_path.resolve()),
        "output_path": str(result_path.resolve()),
        "fixed_k": int(fixed_k),
        "num_batches": int(num_batches),
        "input_names": list(identity["input_names"]),
        "dependencies": dependencies,
    }
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True), encoding="utf-8")
    env = modelopt_subprocess_env(
        tensorrt_root=tensorrt_root,
        conda_env=conda_env,
        pythonpath_entries=[
            "/home/lixingfeng/UniAD_examine/HEAL",
            "/home/lixingfeng/UniAD_examine/heal_compress",
            "/home/lixingfeng/UniAD_examine",
        ],
        cuda_visible_devices=int(physical_gpu_id),
    )
    command = modelopt_python_command(conda_env) + [
        "-m",
        "search.integration.tensorrt_entropy_calibration_worker",
        "--request",
        str(request_path),
    ]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
        timeout=int(timeout_seconds),
    )
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    if not result_path.is_file():
        raise RuntimeError(f"tensorrt_entropy_calibration_worker_no_output_rc_{completed.returncode}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["worker_returncode"] = int(completed.returncode)
    result["log_path"] = str(log_path)
    if completed.returncode != 0 or result.get("status") != "ok":
        raise RuntimeError(
            f"tensorrt_entropy_calibration_failed:{result.get('failure_reason', completed.returncode)}"
        )
    if not cache_path.is_file() or cache_path.stat().st_size <= 0:
        raise RuntimeError("tensorrt_entropy_calibration_cache_not_created")
    if _sha256_file(cache_path) != str(result.get("calibration_cache_sha256", "")):
        raise RuntimeError("tensorrt_entropy_calibration_cache_result_hash_mismatch")
    return result


def _tensor_dict_to_cpu(rows: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in rows.items()}


def collect_or_load_fisher_statistics(
    *,
    model: torch.nn.Module,
    adapter: Any,
    model_config_path: str | Path,
    device: torch.device,
    cache_path: str | Path,
    num_batches: int,
) -> FisherStatistics:
    path = Path(cache_path)
    if path.is_file():
        payload = torch.load(path, map_location="cpu")
        return FisherStatistics(
            gradients={key: value for key, value in payload["gradients"].items()},
            fisher_diag={key: value for key, value in payload["fisher_diag"].items()},
            manifest_hash=str(payload.get("manifest_hash", "")),
            statistics_version=str(payload.get("statistics_version", "fisher-diagonal-v1")),
        )
    if int(num_batches) <= 0:
        raise RuntimeError("fisher_statistics_missing:num_batches")
    _dataset, loader = build_dataset_and_loader(adapter, model_config_path, split="train", num_workers=0, visualize=False)
    batches = iter_limited(loader, int(num_batches))
    if not batches:
        raise RuntimeError("fisher_statistics_missing:no_calibration_batches")
    gradients: dict[str, torch.Tensor] = {}
    fisher: dict[str, torch.Tensor] = {}
    model.train(False)
    for batch in batches:
        batch = move_batch_to_device(batch, device)
        model.zero_grad(set_to_none=True)
        output = adapter.forward_for_task(model, batch)
        loss = adapter.compute_task_loss(output, batch)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            gradients.setdefault(name, torch.zeros_like(param.detach(), device=grad.device))
            fisher.setdefault(name, torch.zeros_like(param.detach(), device=grad.device))
            gradients[name] += grad
            fisher[name] += grad.pow(2)
    count = float(len(batches))
    for name in list(gradients):
        gradients[name] = gradients[name] / count
        fisher[name] = fisher[name] / count
    manifest_hash = canonical_json_hash({"split": "train", "num_batches": int(num_batches), "model_config": str(model_config_path)})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gradients": _tensor_dict_to_cpu(gradients),
            "fisher_diag": _tensor_dict_to_cpu(fisher),
            "manifest_hash": manifest_hash,
            "statistics_version": "fisher-diagonal-v1",
        },
        path,
    )
    return FisherStatistics(_tensor_dict_to_cpu(gradients), _tensor_dict_to_cpu(fisher), manifest_hash=manifest_hash)


def weight_only_calibration_scales(module_paths: list[str], model: torch.nn.Module) -> dict[str, dict[str, float]]:
    """Fallback scale payload with real per-module weight scales and conservative activation scales."""

    modules = dict(model.named_modules())
    scales: dict[str, dict[str, float]] = {}
    for name in module_paths:
        module = modules.get(name)
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        weight_amax = float(weight.detach().abs().amax().item())
        scale = max(weight_amax / 127.0, 1.0e-8)
        scales[name] = {
            "activation_input_scale": 1.0,
            "weight_scale": scale,
            "activation_output_scale": 1.0,
            "activation_source": "fallback_static_unit_scale",
            "weight_source": "actual_pruned_weight_absmax_div127",
        }
    return scales


def collect_or_load_qdq_calibration_scales(
    *,
    model: torch.nn.Module,
    adapter: Any,
    model_config_path: str | Path,
    module_paths: list[str],
    device: torch.device,
    cache_path: str | Path,
    num_batches: int,
    onnx_path: str | Path | None = None,
    origin_map: Any | None = None,
    weight_granularity: str = "per_channel",
    activation_calibration_method: str = "entropy",
    histogram_bins: int = 2048,
    fixed_k: int = 29696,
    calibration_frame_ids: Sequence[str] | None = None,
    calibration_seed: int = 20260713,
    calibration_npz_manifest: str | Path | None = None,
    calibration_input_names: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    expected_calibration_inputs = tuple(
        str(value) for value in (calibration_input_names or FIXED_K_CALIBRATION_INPUT_NAMES)
    )
    path = Path(cache_path)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        cached_inputs = tuple(
            str(value)
            for value in payload.get("metadata", {}).get(
                "input_names", FIXED_K_CALIBRATION_INPUT_NAMES
            )
        )
        if cached_inputs != expected_calibration_inputs:
            raise RuntimeError(
                f"calibration_cache_input_contract_mismatch:{cached_inputs}!={expected_calibration_inputs}"
            )
        return dict(payload["scales"])
    if int(num_batches) <= 0:
        raise RuntimeError("calibration_scales_missing:num_batches")
    import random

    import numpy as np

    expected_frame_ids = [str(value) for value in (calibration_frame_ids or [])]
    if expected_frame_ids and len(expected_frame_ids) != int(num_batches):
        raise RuntimeError(
            f"calibration_manifest_count_mismatch:{len(expected_frame_ids)}!={int(num_batches)}"
        )
    calibration_input_provenance: dict[str, Any] = {}
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(int(calibration_seed))
        np.random.seed(int(calibration_seed) % (2**32))
        torch.manual_seed(int(calibration_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(calibration_seed))
        if calibration_npz_manifest is not None:
            batches, calibration_input_provenance = load_fixed_k_calibration_npz_batches(
                calibration_npz_manifest,
                num_batches=int(num_batches),
                fixed_k=int(fixed_k),
                device=device,
                input_names=expected_calibration_inputs,
            )
        else:
            _dataset, loader = build_dataset_and_loader(adapter, model_config_path, split="train", num_workers=0, visualize=False)
            if len(_dataset) < int(num_batches):
                raise RuntimeError(f"calibration_dataset_too_short:{len(_dataset)}<{int(num_batches)}")
            batches = [move_batch_to_device(batch, device) for batch in iter_limited(loader, int(num_batches))]
            calibration_input_provenance = {
                "source": "dataset_manifest_with_seeded_train_augmentation",
                "sample_count": len(batches),
                "fixed_k": int(fixed_k),
            }
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(torch_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
    if not batches:
        raise RuntimeError("calibration_scales_missing:no_calibration_batches")

    def forward_fn(inner_model: torch.nn.Module, batch: Any) -> Any:
        return adapter.forward_for_task(inner_model, batch)

    if onnx_path is not None and origin_map is not None:
        if expected_calibration_inputs == BASELINE_FIXED_K_CALIBRATION_INPUT_NAMES:
            from search.model_family.export.heal_lidar_baselines import (
                HealLidarBaselineExportPolicy,
                build_heal_lidar_baseline_export_module,
                prepare_heal_lidar_baseline_inputs,
            )

            baseline_policy = HealLidarBaselineExportPolicy(fixed_k=int(fixed_k), max_agents=2)
            wrapper = build_heal_lidar_baseline_export_module(
                model,
                policy=baseline_policy,
            ).to(device).eval()

            def prepare_fixed_k(ego: Any) -> Mapping[str, torch.Tensor]:
                return prepare_heal_lidar_baseline_inputs(ego, policy=baseline_policy)
        elif expected_calibration_inputs == FIXED_K_CALIBRATION_INPUT_NAMES:
            from quantization.config import OnnxExportConfig
            from quantization.export.heal_lidar_pyramid import prepare_signal_maxk_inputs
            from search.integration.trt_compatible_export import build_search_trt_compatible_export_module

            export_config = OnnxExportConfig(fixed_k=int(fixed_k), min_agents=1, opt_agents=2, max_agents=2)
            wrapper = build_search_trt_compatible_export_module(
                model,
                output_names=export_config.output_names,
                fixed_k=export_config.fixed_k,
                modality="m1",
            ).to(device).eval()

            def prepare_fixed_k(ego: Any) -> Mapping[str, torch.Tensor]:
                return prepare_signal_maxk_inputs(ego, config=export_config, modality="m1")
        else:
            raise RuntimeError(
                f"unsupported_qdq_calibration_input_contract:{expected_calibration_inputs}"
            )

        def fixed_k_forward(_inner_model: torch.nn.Module, batch: Any) -> Any:
            if isinstance(batch, Mapping) and all(name in batch for name in expected_calibration_inputs):
                prepared = {name: batch[name] for name in expected_calibration_inputs}
            else:
                ego = batch["ego"] if isinstance(batch, Mapping) and "ego" in batch else batch
                prepared = prepare_fixed_k(ego)
            tensors = tuple(prepared[name].to(device) for name in expected_calibration_inputs)
            return wrapper(*tensors)

        scales, calibration_details = collect_onnx_bn_fold_aware_qdq_scales(
            model=model,
            batches=batches,
            module_paths=module_paths,
            forward_fn=fixed_k_forward,
            onnx_path=onnx_path,
            origin_map=origin_map,
            weight_granularity=weight_granularity,
            activation_calibration_method=activation_calibration_method,
            histogram_bins=histogram_bins,
        )
    else:
        try:
            from quantization.api import collect_calibration_scales
            from quantization.config import CalibrationConfig
        except ImportError:
            from heal_compress.quantization.api import collect_calibration_scales
            from heal_compress.quantization.config import CalibrationConfig
        result = collect_calibration_scales(
            model,
            batches,
            module_paths=module_paths,
            forward_fn=forward_fn,
            config=CalibrationConfig(frame_count=len(batches), require_observed_scales=True),
        )
        scales = result.scales()
        calibration_details = {"semantics_version": "pytorch-weighted-module-v1"}
    save_calibration_scales(
        path,
        scales,
        {
            "frame_count": len(batches),
            "module_count": len(module_paths),
            "manifest_hash": canonical_json_hash(
                {
                    "split": "train",
                    "frames": len(batches),
                    "modules": sorted(module_paths),
                    "activation_calibration_method": activation_calibration_method,
                    "histogram_bins": int(histogram_bins),
                    "fixed_k": int(fixed_k),
                    "input_names": list(expected_calibration_inputs),
                }
            ),
            "source": "search.collect_onnx_bn_fold_aware_qdq_scales" if onnx_path is not None and origin_map is not None else "quantization.collect_calibration_scales",
            "fixed_k": int(fixed_k),
            "input_names": list(expected_calibration_inputs),
            "calibration_seed": int(calibration_seed),
            "calibration_frame_ids": expected_frame_ids,
            "calibration_frame_manifest_hash": canonical_json_hash(
                {"split": "train", "frame_ids": expected_frame_ids, "order": "dataset_manifest_order"}
            ),
            "calibration_order": (
                "npz_manifest_file_order"
                if calibration_npz_manifest is not None
                else "dataset_manifest_order_shuffle_false"
            ),
            "calibration_input_provenance": calibration_input_provenance,
            **calibration_details,
        },
    )
    return scales


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
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


def _paired_batchnorm_path(model: torch.nn.Module, module_path: str) -> str | None:
    modules = dict(model.named_modules())
    if module_path not in modules or "." not in module_path:
        return None
    parent_path, child_name = module_path.rsplit(".", 1)
    parent = modules.get(parent_path)
    if parent is None:
        return None
    if child_name == "conv" or (child_name.startswith("conv") and child_name[4:].isdigit()):
        sibling = "bn" if child_name == "conv" else f"bn{child_name[4:]}"
        candidate = f"{parent_path}.{sibling}"
        if isinstance(modules.get(candidate), torch.nn.modules.batchnorm._BatchNorm):
            return candidate
    if isinstance(parent, torch.nn.Sequential) and child_name.isdigit():
        candidate = f"{parent_path}.{int(child_name) + 1}"
        if isinstance(modules.get(candidate), torch.nn.modules.batchnorm._BatchNorm):
            return candidate
    return None


def _origin_entries(origin_map: Any) -> list[Any]:
    if isinstance(origin_map, Mapping):
        return list(origin_map.get("entries", []) or [])
    return list(getattr(origin_map, "entries", []) or [])


def _field(row: Any, name: str) -> Any:
    return row.get(name) if isinstance(row, Mapping) else getattr(row, name)


def collect_onnx_bn_fold_aware_qdq_scales(
    *,
    model: torch.nn.Module,
    batches: Iterable[Any],
    module_paths: Sequence[str],
    forward_fn: Any,
    onnx_path: str | Path,
    origin_map: Any,
    weight_granularity: str = "per_tensor",
    activation_calibration_method: str = "entropy",
    histogram_bins: int = 2048,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Collect scales against the final, BatchNorm-folded ONNX compute nodes.

    PyTorch Conv hooks observe pre-BatchNorm weights and outputs.  ONNX export
    normally folds the following BatchNorm into the Conv initializer, so those
    scales are not valid at the explicit Q/DQ insertion points.  Inputs remain
    attached to the Conv, outputs move to the folded BatchNorm output, and
    weights are measured from the actual ONNX initializer.
    """

    import numpy as np
    import onnx
    from onnx import numpy_helper

    requested = [str(value) for value in module_paths]
    batch_rows = list(batches)
    if weight_granularity not in {"per_tensor", "per_channel"}:
        raise RuntimeError(f"unsupported_qdq_weight_granularity:{weight_granularity}")
    if activation_calibration_method not in {"absmax", "entropy"}:
        raise RuntimeError(f"unsupported_activation_calibration_method:{activation_calibration_method}")
    if int(histogram_bins) < 128:
        raise RuntimeError(f"activation_histogram_bins_too_small:{histogram_bins}")
    if len(requested) != len(set(requested)):
        raise RuntimeError("qdq_calibration_module_paths_not_unique")
    modules = dict(model.named_modules())
    missing = [name for name in requested if name not in modules]
    if missing:
        raise RuntimeError(f"qdq_calibration_modules_missing:{missing}")
    entries = {_field(row, "module_path"): row for row in _origin_entries(origin_map)}
    missing_entries = [name for name in requested if name not in entries]
    if missing_entries:
        raise RuntimeError(f"qdq_calibration_origin_entries_missing:{missing_entries}")
    onnx_model = onnx.load(str(onnx_path))
    initializers = {row.name: numpy_helper.to_array(row) for row in onnx_model.graph.initializer}
    node_by_name = {str(row.name): row for row in onnx_model.graph.node}
    consumers: dict[str, list[Any]] = {}
    for node in onnx_model.graph.node:
        for input_name in node.input:
            consumers.setdefault(str(input_name), []).append(node)
    output_paths: dict[str, str] = {}
    for name in requested:
        entry = entries[name]
        node = node_by_name.get(str(_field(entry, "canonical_node_name")))
        paired = _paired_batchnorm_path(model, name)
        onnx_has_bn_consumer = bool(
            node is not None
            and any(str(consumer.op_type) == "BatchNormalization" for output in node.output for consumer in consumers.get(str(output), []))
        )
        output_paths[name] = paired if paired is not None and not onnx_has_bn_consumer else name
    state = {
        name: {
            "input_amax": None,
            "output_amax": None,
            "input_hist": None,
            "output_hist": None,
            "input_count": 0,
            "output_count": 0,
        }
        for name in requested
    }
    phase = {"name": "amax"}
    handles = []

    def observe(name: str, role: str, tensor: torch.Tensor) -> None:
        value = tensor.detach().float().abs()
        row = state[name]
        if phase["name"] == "amax":
            amax = value.amax()
            key = f"{role}_amax"
            row[key] = amax if row[key] is None else torch.maximum(row[key], amax)
            row[f"{role}_count"] += 1
            return
        maximum = float(row[f"{role}_amax"].item())
        histogram = torch.histc(value, bins=int(histogram_bins), min=0.0, max=maximum)
        key = f"{role}_hist"
        row[key] = histogram if row[key] is None else row[key] + histogram

    def input_hook(name: str) -> Any:
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            tensor = _first_tensor(inputs)
            if tensor is None:
                raise RuntimeError(f"qdq_calibration_input_tensor_missing:{name}")
            observe(name, "input", tensor)

        return hook

    def output_hook(name: str) -> Any:
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = _first_tensor(output)
            if tensor is None:
                raise RuntimeError(f"qdq_calibration_output_tensor_missing:{name}")
            observe(name, "output", tensor)

        return hook

    for name in requested:
        handles.append(modules[name].register_forward_pre_hook(input_hook(name)))
        handles.append(modules[output_paths[name]].register_forward_hook(output_hook(name)))
    was_training = bool(model.training)
    frame_count = 0
    model.eval()
    try:
        with torch.inference_mode():
            for batch in batch_rows:
                forward_fn(model, batch)
                frame_count += 1
            if activation_calibration_method == "entropy":
                phase["name"] = "histogram"
                for batch in batch_rows:
                    forward_fn(model, batch)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    if frame_count <= 0:
        raise RuntimeError("qdq_calibration_received_no_frames")
    scales: dict[str, dict[str, Any]] = {}
    weight_scale_sources: dict[str, str] = {}
    weight_axes: dict[str, int | None] = {}
    weight_scale_shapes: dict[str, list[int]] = {}
    activation_thresholds: dict[str, dict[str, Any]] = {}

    def calibrated_amax(row: dict[str, Any], role: str) -> tuple[float, dict[str, Any]]:
        absolute_maximum = float(row[f"{role}_amax"].item())
        if activation_calibration_method == "absmax":
            return absolute_maximum, {
                "method": "absmax",
                "absolute_maximum": absolute_maximum,
                "clipping_threshold": absolute_maximum,
                "selected_bin": int(histogram_bins),
                "clipped_fraction": 0.0,
            }
        import numpy as np
        from modelopt.torch.quantization.calib.histogram import _compute_amax_entropy

        histogram = row[f"{role}_hist"].cpu().numpy().astype(np.int64)
        edges = np.linspace(0.0, absolute_maximum, int(histogram_bins) + 1, dtype=np.float64)
        threshold = float(
            _compute_amax_entropy(
                histogram.copy(),
                edges,
                num_bits=8,
                unsigned=False,
                stride=1,
                start_bin=128,
            ).item()
        )
        selected_bin = min(
            max(int(round(threshold / absolute_maximum * int(histogram_bins))), 128),
            int(histogram_bins),
        )
        return threshold, {
            "method": "entropy",
            "implementation": "modelopt.torch.quantization.calib.histogram._compute_amax_entropy",
            "absolute_maximum": absolute_maximum,
            "clipping_threshold": threshold,
            "selected_bin": selected_bin,
            "clipped_fraction": float(histogram[selected_bin:].sum() / max(histogram.sum(), 1)),
        }

    for name in requested:
        row = state[name]
        if row["input_count"] != frame_count or row["output_count"] != frame_count:
            raise RuntimeError(f"qdq_calibration_observation_count_mismatch:{name}")
        initializer_name = str(_field(entries[name], "weight_initializer"))
        node = node_by_name.get(str(_field(entries[name], "canonical_node_name")))
        if node is None:
            raise RuntimeError(f"qdq_calibration_onnx_node_missing:{name}")
        weight = initializers.get(initializer_name)
        if weight is None:
            raise RuntimeError(f"qdq_calibration_onnx_initializer_missing:{name}:{initializer_name}")
        input_amax, input_threshold = calibrated_amax(row, "input")
        output_amax, output_threshold = calibrated_amax(row, "output")
        activation_thresholds[name] = {"input": input_threshold, "output": output_threshold}
        weight_amax = float(np.max(np.abs(weight)))
        values = (input_amax, output_amax, weight_amax)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise RuntimeError(f"qdq_calibration_nonpositive_or_nonfinite_amax:{name}")
        weight_axis: int | None = None
        weight_scale: float | list[float]
        if weight_granularity == "per_channel":
            if str(node.op_type) == "Conv":
                weight_axis = 0
            elif str(node.op_type) == "ConvTranspose":
                # ONNX ConvTranspose weights are [C_in, C_out/group, ...].
                # Axis 1 is the physical output-channel axis for group=1 and
                # the only representable output-channel axis for this layout.
                weight_axis = 1
            elif str(node.op_type) == "MatMul":
                # Linear exported as A @ B uses B=[C_in, C_out].
                weight_axis = 1
            elif str(node.op_type) == "Gemm":
                attributes = {str(attr.name): int(attr.i) for attr in node.attribute if str(attr.name) == "transB"}
                weight_axis = 0 if attributes.get("transB", 0) else 1
            else:
                raise RuntimeError(f"qdq_calibration_per_channel_unsupported_op:{name}:{node.op_type}")
            if weight_axis >= weight.ndim:
                raise RuntimeError(f"qdq_calibration_weight_axis_out_of_range:{name}:{weight_axis}:{list(weight.shape)}")
            reduce_axes = tuple(axis for axis in range(weight.ndim) if axis != weight_axis)
            channel_amax = np.max(np.abs(weight), axis=reduce_axes)
            if not np.all(np.isfinite(channel_amax)) or np.any(channel_amax <= 0.0):
                raise RuntimeError(f"qdq_calibration_nonpositive_per_channel_weight_amax:{name}")
            weight_scale = (channel_amax / 127.0).astype(np.float32).tolist()
        else:
            weight_scale = weight_amax / 127.0
        scales[name] = {
            "activation_input_scale": input_amax / 127.0,
            "weight_scale": weight_scale,
            "weight_axis": weight_axis,
            "weight_granularity": weight_granularity,
            "weight_scale_shape": [len(weight_scale)] if isinstance(weight_scale, list) else [],
            "activation_output_scale": output_amax / 127.0,
            "activation_input_tensor": str(node.input[0]) if node is not None else "",
            "activation_output_tensor": str(node.output[0]) if node is not None else "",
            "activation_scale_source": (
                "fixedK_export_wrapper_NVIDIA_ModelOpt_entropy"
                if activation_calibration_method == "entropy"
                else "fixedK_export_wrapper_absmax"
            ),
            "weight_scale_source": "final_folded_onnx_initializer",
        }
        weight_scale_sources[name] = initializer_name
        weight_axes[name] = weight_axis
        weight_scale_shapes[name] = [len(weight_scale)] if isinstance(weight_scale, list) else []
    return scales, {
        "semantics_version": QDQ_CALIBRATION_SEMANTICS_VERSION,
        "frame_count": frame_count,
        "output_module_paths": output_paths,
        "weight_initializer_names": weight_scale_sources,
        "weight_granularity": weight_granularity,
        "weight_axes": weight_axes,
        "weight_scale_shapes": weight_scale_shapes,
        "activation_calibration_method": activation_calibration_method,
        "histogram_bins": int(histogram_bins),
        "activation_thresholds": activation_thresholds,
        "passes": 2 if activation_calibration_method == "entropy" else 1,
    }


def save_calibration_scales(path: str | Path, scales: dict[str, Any], metadata: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"metadata": metadata, "scales": scales}, indent=2, sort_keys=True), encoding="utf-8")
