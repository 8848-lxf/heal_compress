"""Evaluate external-scatter HEAL TensorRT engines on collected CARLA frames."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from point_frontend.gpu_voxelization import (
    DeterministicGpuVoxelizer,
    mathematical_voxel_capacity,
)


IOU_THRESHOLDS = (0.30, 0.50, 0.70)
ENGINE_NAMES = ("candidate", "baseline")
MODEL_NAMES = ("candidate", "baseline", "fp32_pytorch")
PROTOCOLS = ("range_all", "visible_5plus")


def _rotated_model_order(frame_index: int) -> Tuple[str, ...]:
    """Balance per-frame first-call postprocess overhead across backends."""
    offset = int(frame_index) % len(MODEL_NAMES)
    return tuple(MODEL_NAMES[offset:] + MODEL_NAMES[:offset])


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def _distribution(values: Sequence[float]) -> Dict[str, float]:
    rows = [float(value) for value in values]
    if not rows:
        return {key: 0.0 for key in ("mean", "p50", "p90", "p95", "p99", "min", "max")}
    return {
        "mean": float(statistics.mean(rows)),
        "p50": _percentile(rows, 50.0),
        "p90": _percentile(rows, 90.0),
        "p95": _percentile(rows, 95.0),
        "p99": _percentile(rows, 99.0),
        "min": min(rows),
        "max": max(rows),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mask_ego(points: np.ndarray) -> np.ndarray:
    mask = (
        (points[:, 0] >= -1.95)
        & (points[:, 0] <= 2.95)
        & (points[:, 1] >= -1.1)
        & (points[:, 1] <= 1.1)
    )
    return points[~mask]


class DynamicPointPillarFrontend:
    """Selectable CPU/GPU voxelization plus the shared FP32 PFN/scatter frontend."""

    def __init__(
        self,
        hypes: Mapping[str, Any],
        checkpoint: Path,
        device: Any,
        *,
        voxelization_backend: str = "gpu",
    ) -> None:
        import torch
        from opencood.models.heter_encoders import PointPillar

        backend = str(voxelization_backend).lower()
        if backend not in {"cpu", "gpu"}:
            raise ValueError(f"unsupported voxelization backend: {backend}")
        modality = hypes["heter"]["modality_setting"]["m1"]
        preprocess_config = copy.deepcopy(modality["preprocess"])
        lidar_range = list(preprocess_config["cav_lidar_range"])
        voxel_size = list(preprocess_config["args"]["voxel_size"])
        self.per_agent_capacity = mathematical_voxel_capacity(lidar_range, voxel_size)
        # Point2Voxel requires a capacity. The complete voxel grid is a
        # mathematical bound, not a dataset-derived fixed K.
        self.preprocessor = None
        self.gpu_voxelizer = None
        if backend == "cpu":
            from opencood.data_utils.pre_processor import build_preprocessor

            preprocess_config["args"]["max_voxel_test"] = self.per_agent_capacity
            self.preprocessor = build_preprocessor(preprocess_config, train=False)
        else:
            self.gpu_voxelizer = DeterministicGpuVoxelizer(
                preprocess_config, device
            )
        encoder_config = copy.deepcopy(hypes["model"]["args"]["m1"]["encoder_args"])
        self.encoder = PointPillar(encoder_config)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model_state = payload.get("model", payload)
        state = {
            name[len("encoder_m1.") :]: value
            for name, value in model_state.items()
            if name.startswith("encoder_m1.")
        }
        self.encoder.load_state_dict(state, strict=True)
        self.encoder.to(device).eval()
        self.device = device
        self.voxelization_backend = backend

    def runtime_contract(self) -> Mapping[str, Any]:
        if self.gpu_voxelizer is not None:
            return self.gpu_voxelizer.runtime_contract()
        return {
            "backend": "heal_spconv_point_to_voxel_cpu",
            "device": "cpu",
            "per_agent_mathematical_capacity": self.per_agent_capacity,
            "dataset_derived_max_k": False,
        }

    @staticmethod
    def _seed(sample_id: str) -> int:
        return int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16)

    def encode(
        self, clouds: Sequence[np.ndarray], sample_id: str
    ) -> Tuple[Any, Mapping[str, Any], Mapping[str, float]]:
        import torch

        preprocess_started = time.perf_counter()
        rng = np.random.default_rng(self._seed(sample_id))
        point_rows = []
        point_counts = []
        for points in clouds:
            values = _mask_ego(np.asarray(points, dtype=np.float32))
            values = np.ascontiguousarray(values[rng.permutation(values.shape[0])])
            point_counts.append(int(values.shape[0]))
            point_rows.append(values)
        point_preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0

        voxelize_cpu_ms = 0.0
        voxelize_gpu_ms = 0.0
        if self.voxelization_backend == "cpu":
            voxel_started = time.perf_counter()
            processed = [self.preprocessor.preprocess(values) for values in point_rows]
            batch = self.preprocessor.collate_batch(processed)
            voxelize_cpu_ms = (time.perf_counter() - voxel_started) * 1000.0
            per_agent_k = [int(item["voxel_features"].shape[0]) for item in processed]
            saturated = [
                int(
                    np.count_nonzero(
                        item["voxel_num_points"]
                        == int(self.preprocessor.max_points_per_voxel)
                    )
                )
                for item in processed
            ]
            audit = {
                "input_point_counts": point_counts,
                "voxel_counts": per_agent_k,
                "total_voxel_count": int(sum(per_agent_k)),
                "saturated_voxel_counts": saturated,
                "total_saturated_voxel_count": int(sum(saturated)),
                "per_agent_mathematical_capacity": self.per_agent_capacity,
            }
            h2d_started = time.perf_counter()
            batch = {name: tensor.to(self.device) for name, tensor in batch.items()}
            torch.cuda.synchronize(self.device)
            host_to_device_ms = (time.perf_counter() - h2d_started) * 1000.0
        else:
            h2d_started = time.perf_counter()
            gpu_points = [torch.from_numpy(values).to(self.device) for values in point_rows]
            torch.cuda.synchronize(self.device)
            host_to_device_ms = (time.perf_counter() - h2d_started) * 1000.0
            batch, audit, voxelize_gpu_ms = self.gpu_voxelizer.voxelize(gpu_points)

        for voxel_count in audit["voxel_counts"]:
            if int(voxel_count) > self.per_agent_capacity:
                raise RuntimeError(
                    "voxelizer exceeded the mathematical grid capacity for "
                    f"{sample_id}: K={voxel_count}>{self.per_agent_capacity}"
                )
            if int(voxel_count) <= 0:
                raise RuntimeError(f"empty voxel cloud for {sample_id}")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            encoded = self.encoder.pillar_vfe(batch)
            spatial = self.encoder.scatter(encoded)["spatial_features"]
        end.record()
        end.synchronize()
        frontend_ms = float(start.elapsed_time(end))
        if int(spatial.shape[0]) != len(clouds):
            raise RuntimeError(
                f"scatter agent count mismatch for {sample_id}: {tuple(spatial.shape)}"
            )
        audit = dict(audit)
        audit.update(
            {
                "sample_id": sample_id,
                "input_point_counts_after_ego_mask": point_counts,
                "voxelization_backend": self.voxelization_backend,
            }
        )
        timings = {
            "point_preprocess_cpu_ms": point_preprocess_ms,
            "voxelize_cpu_ms": voxelize_cpu_ms,
            "host_to_device_ms": host_to_device_ms,
            "voxelize_gpu_ms": voxelize_gpu_ms,
            "pfn_scatter_gpu_ms": frontend_ms,
        }
        return spatial.contiguous(), audit, timings


class UnprunedFP32PostScatter:
    """Run the original PyTorch model from the shared post-scatter boundary."""

    def __init__(
        self, hypes: Mapping[str, Any], checkpoint: Path, device: Any
    ) -> None:
        import torch
        from opencood.tools import train_utils

        self.model = train_utils.create_model(hypes)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model_state = payload.get("model", payload)
        self.model.load_state_dict(model_state, strict=True)
        self.model.to(device).eval()
        self.device = device

    def run(self, spatial: Any, pairwise: Any) -> Tuple[Mapping[str, Any], float]:
        import torch
        from opencood.utils.transformation_utils import normalize_pairwise_tfm

        model = self.model
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            feature = model.backbone_m1({"spatial_features": spatial})[
                "spatial_features_2d"
            ]
            affine = normalize_pairwise_tfm(
                pairwise, model.H, model.W, model.fake_voxel_size
            )
            record_len = torch.tensor([spatial.shape[0]], device=self.device)
            if hasattr(model, "pyramid_backbone"):
                feature = model.aligner_m1(feature)
                fused, _ = model.pyramid_backbone.forward_collab(
                    feature,
                    record_len,
                    affine,
                    ["m1"] * int(spatial.shape[0]),
                    model.cam_crop_info,
                )
            else:
                feature = model.shrinker_m1(feature)
                if bool(getattr(model, "compress", False)):
                    feature = model.compressor(feature)
                fused = model.fusion_net(feature, record_len, affine)
            if model.shrink_flag:
                fused = model.shrink_conv(fused)
            outputs = {
                "cls_preds": model.cls_head(fused),
                "reg_preds": model.reg_head(fused),
                "dir_preds": model.dir_head(fused),
            }
        end.record()
        end.synchronize()
        return outputs, float(start.elapsed_time(end))


def _new_stat() -> MutableMapping[float, MutableMapping[str, Any]]:
    return {
        threshold: {"tp": [], "fp": [], "gt": 0, "score": []}
        for threshold in IOU_THRESHOLDS
    }


def _ap_summary(stat: Mapping[float, Mapping[str, Any]]) -> Mapping[str, Any]:
    from opencood.utils import eval_utils

    if not any(int(stat[threshold]["gt"]) for threshold in IOU_THRESHOLDS):
        return {
            "gt_count": 0,
            "ap30": 0.0,
            "ap50": 0.0,
            "ap70": 0.0,
            "map": 0.0,
            "precision_at_score_floor_iou30": 0.0,
            "recall_at_score_floor_iou30": 0.0,
        }
    aps = {
        threshold: float(eval_utils.calculate_ap(stat, threshold)[0])
        for threshold in IOU_THRESHOLDS
    }
    true_positives = int(sum(stat[0.30]["tp"]))
    false_positives = int(sum(stat[0.30]["fp"]))
    gt_count = int(stat[0.30]["gt"])
    return {
        "gt_count": gt_count,
        "ap30": aps[0.30],
        "ap50": aps[0.50],
        "ap70": aps[0.70],
        "map": float(statistics.mean(aps.values())),
        "precision_at_score_floor_iou30": true_positives
        / max(true_positives + false_positives, 1),
        "recall_at_score_floor_iou30": true_positives / max(gt_count, 1),
        "true_positives_at_iou30": true_positives,
        "false_positives_at_iou30": false_positives,
    }


def _postprocess(
    postprocessor: Any,
    anchors: Any,
    outputs: Mapping[str, Any],
    device: Any,
) -> Tuple[Any, Any, float]:
    import torch

    fp32_outputs = _postprocess_outputs_fp32(outputs)
    data = {
        "ego": {
            "anchor_box": anchors,
            "transformation_matrix": torch.eye(4, dtype=torch.float32, device=device),
        }
    }
    prediction = {
        "ego": {
            "cls_preds": fp32_outputs["cls_preds"],
            "reg_preds": fp32_outputs["reg_preds"],
            "dir_preds": fp32_outputs["dir_preds"],
        }
    }
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    boxes, scores = postprocessor.post_process(data, prediction)
    torch.cuda.synchronize(device)
    return boxes, scores, (time.perf_counter() - started) * 1000.0


def _postprocess_outputs_fp32(outputs: Mapping[str, Any]) -> Mapping[str, Any]:
    """Normalize mixed-precision engine heads to HEAL's FP32 postprocess API."""
    required = ("cls_preds", "reg_preds", "dir_preds")
    missing = [name for name in required if name not in outputs]
    if missing:
        raise RuntimeError(f"postprocess outputs missing required heads: {missing}")
    return {name: outputs[name].float() for name in required}


def _frame_rows(
    manifest: Mapping[str, Any], data_root: Path
) -> Iterable[Tuple[str, Mapping[str, Any], Path]]:
    for scene in manifest["scenes"]:
        scene_id = str(scene["scene_id"])
        for frame in scene["frames"]:
            yield scene_id, frame, data_root / str(frame["relative_path"])


def _ground_truth_protocols(payload: Any, device: Any) -> Mapping[str, Any]:
    import torch

    boxes = torch.from_numpy(payload["gt_boxes"]).float().to(device)
    if "gt_lidar_hits" not in payload:
        raise RuntimeError(
            "capture lacks gt_lidar_hits; recollect with cooperative-frames-v2"
        )
    hits = torch.from_numpy(payload["gt_lidar_hits"]).to(device)
    return {
        "range_all": boxes,
        "visible_5plus": boxes[hits >= 5],
    }


def evaluate(arguments: argparse.Namespace) -> Mapping[str, Any]:
    heal_root = arguments.heal_root.expanduser().resolve()
    sys.path.insert(0, str(heal_root))
    import torch
    from opencood.data_utils.post_processor import build_postprocessor
    from opencood.hypes_yaml import yaml_utils
    from opencood.utils import eval_utils

    from carla_integration.intensity_calibration import IntensityCalibration
    from quantization.tensorrt.runtime import TensorRTEngineRunner, load_trt_engine

    if not torch.cuda.is_available():
        raise RuntimeError("CARLA TensorRT evaluation requires CUDA")
    device = torch.device(arguments.device)
    torch.cuda.set_device(device)
    hypes = yaml_utils.load_yaml(str(arguments.model_config.expanduser().resolve()))
    hypes["postprocess"]["target_args"]["score_threshold"] = float(
        arguments.score_floor
    )
    voxelization_backend = str(
        getattr(arguments, "voxelization_backend", "gpu")
    ).lower()
    frontend = DynamicPointPillarFrontend(
        hypes,
        arguments.checkpoint.expanduser().resolve(),
        device,
        voxelization_backend=voxelization_backend,
    )
    fp32_checkpoint = arguments.fp32_checkpoint.expanduser().resolve()
    fp32_runner = UnprunedFP32PostScatter(hypes, fp32_checkpoint, device)
    postprocessor = build_postprocessor(hypes["postprocess"], train=False)
    anchors = torch.from_numpy(postprocessor.generate_anchor_box()).float().to(device)

    engines = {
        "candidate": arguments.candidate_engine.expanduser().resolve(),
        "baseline": arguments.baseline_engine.expanduser().resolve(),
    }
    runners = {
        name: TensorRTEngineRunner(load_trt_engine(path), device=str(device))
        for name, path in engines.items()
    }
    current_stream = torch.cuda.current_stream(device)
    for runner in runners.values():
        runner.stream = current_stream
        input_names = set(runner.input_names())
        if input_names not in (
            {"spatial_features", "pairwise_t_matrix"},
            {"spatial_features", "pairwise_t_matrix", "agent_mask"},
        ):
            raise RuntimeError(f"unexpected engine inputs: {runner.input_names()}")
    candidate_inputs = set(runners["candidate"].input_names())
    baseline_inputs = set(runners["baseline"].input_names())
    if candidate_inputs != baseline_inputs:
        raise RuntimeError(
            "candidate/baseline post-scatter input mismatch: "
            f"{sorted(candidate_inputs)} != {sorted(baseline_inputs)}"
        )

    data_root = arguments.data.expanduser().resolve()
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration_path = arguments.intensity_calibration.expanduser().resolve()
    calibration = IntensityCalibration.from_path(calibration_path)
    manifest_maps = {str(scene["map"]) for scene in manifest["scenes"]}
    held_out_maps = set(calibration.payload["held_out_test_maps"])
    source_maps = set(calibration.payload["source_provenance"]["maps"])
    if not manifest_maps.issubset(held_out_maps):
        raise RuntimeError(
            "test manifest contains maps not frozen as held out: "
            f"{sorted(manifest_maps - held_out_maps)}"
        )
    if manifest_maps & source_maps:
        raise RuntimeError(
            f"CARLA calibration/test map leakage: {sorted(manifest_maps & source_maps)}"
        )
    frames = list(_frame_rows(manifest, data_root))
    if len(frames) < 5:
        raise RuntimeError("evaluation requires at least five collected frames")

    overall_stats = {
        protocol: {name: _new_stat() for name in MODEL_NAMES}
        for protocol in PROTOCOLS
    }
    scene_stats: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(
        lambda: {
            protocol: {name: _new_stat() for name in MODEL_NAMES}
            for protocol in PROTOCOLS
        }
    )
    latencies: Dict[str, list] = defaultdict(list)
    forward_latencies: Dict[str, list] = {name: [] for name in MODEL_NAMES}
    post_latencies: Dict[str, list] = {name: [] for name in MODEL_NAMES}
    voxel_audit = []
    frame_reports = []

    for frame_index, (scene_id, frame, path) in enumerate(frames):
        sample_id = str(frame["sample_id"])
        with np.load(path) as payload:
            clouds = [payload["vehicle_points"], payload["infrastructure_points"]]
            if not arguments.raw_intensity:
                clouds = [
                    calibration.apply(clouds[0], "vehicle"),
                    calibration.apply(clouds[1], "infrastructure"),
                ]
            spatial, audit, frontend_timings = frontend.encode(clouds, sample_id)
            pairwise = (
                torch.from_numpy(payload["pairwise_t_matrix"])
                .float()
                .unsqueeze(0)
                .to(device)
            )
            pairwise = pairwise.contiguous()
            inputs = {
                "spatial_features": spatial,
                "pairwise_t_matrix": pairwise,
            }
            if "agent_mask" in candidate_inputs:
                inputs["agent_mask"] = torch.ones(
                    (1, int(spatial.shape[0])),
                    dtype=spatial.dtype,
                    device=device,
                )
            run_outputs = {}
            frame_forward_ms = {}
            for name in ENGINE_NAMES:
                outputs, forward_ms = runners[name].run(inputs)
                run_outputs[name] = outputs
                frame_forward_ms[name] = float(forward_ms)
            fp32_outputs, fp32_ms = fp32_runner.run(spatial, pairwise)
            run_outputs["fp32_pytorch"] = fp32_outputs
            frame_forward_ms["fp32_pytorch"] = fp32_ms
            if frame_index >= int(arguments.warmup_frames):
                for name in MODEL_NAMES:
                    forward_latencies[name].append(frame_forward_ms[name])
            gt_by_protocol = _ground_truth_protocols(payload, device)
            model_rows = {}
            postprocess_order = _rotated_model_order(frame_index)
            for name in postprocess_order:
                boxes, scores, post_ms = _postprocess(
                    postprocessor, anchors, run_outputs[name], device
                )
                for protocol, gt_boxes in gt_by_protocol.items():
                    for threshold in IOU_THRESHOLDS:
                        eval_utils.caluclate_tp_fp(
                            boxes,
                            scores,
                            gt_boxes,
                            overall_stats[protocol][name],
                            threshold,
                        )
                        eval_utils.caluclate_tp_fp(
                            boxes,
                            scores,
                            gt_boxes,
                            scene_stats[scene_id][protocol][name],
                            threshold,
                        )
                if frame_index >= int(arguments.warmup_frames):
                    post_latencies[name].append(post_ms)
                model_rows[name] = {
                    "prediction_count": 0 if boxes is None else int(boxes.shape[0]),
                    "forward_gpu_ms": frame_forward_ms[name],
                    "postprocess_ms": post_ms,
                }
            gt_counts = {
                protocol: int(boxes.shape[0])
                for protocol, boxes in gt_by_protocol.items()
            }
        if frame_index >= int(arguments.warmup_frames):
            for key, value in frontend_timings.items():
                latencies[key].append(float(value))
        voxel_audit.append(audit)
        frame_reports.append(
            {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "gt_counts": gt_counts,
                "frontend_timings_ms": dict(frontend_timings),
                "models": model_rows,
                "postprocess_order": list(postprocess_order),
            }
        )

    frontend_summary = {
        key: _distribution(latencies[key])
        for key in (
            "point_preprocess_cpu_ms",
            "voxelize_cpu_ms",
            "host_to_device_ms",
            "voxelize_gpu_ms",
            "pfn_scatter_gpu_ms",
        )
    }
    frontend_summary["voxelization_backend"] = voxelization_backend
    frontend_summary["runtime_contract"] = frontend.runtime_contract()
    shared_frontend = [
        sum(values)
        for values in zip(
            latencies["point_preprocess_cpu_ms"],
            latencies["voxelize_cpu_ms"],
            latencies["host_to_device_ms"],
            latencies["voxelize_gpu_ms"],
            latencies["pfn_scatter_gpu_ms"],
        )
    ]
    frontend_summary["total_frontend_ms"] = _distribution(shared_frontend)
    models: Dict[str, Any] = {}
    for name in MODEL_NAMES:
        forward_distribution = _distribution(forward_latencies[name])
        post_distribution = _distribution(post_latencies[name])
        model_path = [
            voxel + pfn + forward
            for voxel, pfn, forward in zip(
                latencies["voxelize_gpu_ms"],
                latencies["pfn_scatter_gpu_ms"],
                forward_latencies[name],
            )
        ]
        composed = [
            point_cpu + voxel_cpu + h2d + voxel_gpu + pfn + forward + post
            for point_cpu, voxel_cpu, h2d, voxel_gpu, pfn, forward, post in zip(
                latencies["point_preprocess_cpu_ms"],
                latencies["voxelize_cpu_ms"],
                latencies["host_to_device_ms"],
                latencies["voxelize_gpu_ms"],
                latencies["pfn_scatter_gpu_ms"],
                forward_latencies[name],
                post_latencies[name],
            )
        ]
        models[name] = {
            "backend_path": str(engines[name]) if name in engines else str(fp32_checkpoint),
            "backend_sha256": (
                _sha256(engines[name])
                if name in engines
                else _sha256(fp32_checkpoint)
            ),
            "accuracy": {
                protocol: _ap_summary(overall_stats[protocol][name])
                for protocol in PROTOCOLS
            },
            "forward_gpu_ms": forward_distribution,
            "model_path_gpu_ms": _distribution(model_path),
            "postprocess_ms": post_distribution,
            "composed_end_to_end_ms": _distribution(composed),
        }
    speedup = {
        "candidate_vs_strict_fp32_trt_forward_mean": models["baseline"]["forward_gpu_ms"]["mean"]
        / max(models["candidate"]["forward_gpu_ms"]["mean"], 1e-12),
        "candidate_vs_unpruned_fp32_pytorch_forward_mean": models[
            "fp32_pytorch"
        ]["forward_gpu_ms"]["mean"]
        / max(models["candidate"]["forward_gpu_ms"]["mean"], 1e-12),
        "candidate_vs_unpruned_fp32_pytorch_model_path_mean": models[
            "fp32_pytorch"
        ]["model_path_gpu_ms"]["mean"]
        / max(models["candidate"]["model_path_gpu_ms"]["mean"], 1e-12),
        "candidate_vs_unpruned_fp32_pytorch_composed_mean": models[
            "fp32_pytorch"
        ]["composed_end_to_end_ms"]["mean"]
        / max(models["candidate"]["composed_end_to_end_ms"]["mean"], 1e-12),
    }
    per_scene = {
        scene_id: {
            protocol: {
                name: _ap_summary(scene_stats[scene_id][protocol][name])
                for name in MODEL_NAMES
            }
            for protocol in PROTOCOLS
        }
        for scene_id in sorted(scene_stats)
    }
    return {
        "schema_version": "heal-carla-no-leak-evaluation-v3",
        "success": True,
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": _sha256(manifest_path),
        "scene_count": len(manifest["scenes"]),
        "frame_count": len(frames),
        "timed_frame_count": max(0, len(frames) - int(arguments.warmup_frames)),
        "point_frontend": (
            f"{voxelization_backend}_dynamic_voxel_pfn_scatter_outside_tensorrt"
        ),
        "voxelization_backend": voxelization_backend,
        "legacy_fixed_k_used": False,
        "score_floor": float(arguments.score_floor),
        "intensity_mode": (
            "raw_carla_posthoc_ablation"
            if arguments.raw_intensity
            else "frozen_dair_train_quantile_map"
        ),
        "metric_protocols": list(PROTOCOLS),
        "postprocess_timing_order": "deterministic_round_robin_per_frame",
        "intensity_calibration": {
            "path": str(calibration_path),
            "sha256": _sha256(calibration_path),
            "source_maps": sorted(source_maps),
            "held_out_test_maps": sorted(held_out_maps),
            "dair_train_pair_count": calibration.payload["target_provenance"]["train_pair_count"],
        },
        "artifacts": {
            "candidate_frontend_checkpoint": str(arguments.checkpoint.expanduser().resolve()),
            "candidate_frontend_checkpoint_sha256": _sha256(
                arguments.checkpoint.expanduser().resolve()
            ),
            "unpruned_fp32_checkpoint": str(fp32_checkpoint),
            "unpruned_fp32_checkpoint_sha256": _sha256(fp32_checkpoint),
            "model_config": str(arguments.model_config.expanduser().resolve()),
            "model_config_sha256": _sha256(arguments.model_config.expanduser().resolve()),
            "heal_root": str(heal_root),
        },
        "runtime": {
            "torch_version": str(torch.__version__),
            "gpu": str(torch.cuda.get_device_name(device)),
            "point_frontend": frontend.runtime_contract(),
        },
        "frontend": frontend_summary,
        "voxel_count": {
            "min": min(row["total_voxel_count"] for row in voxel_audit),
            "max": max(row["total_voxel_count"] for row in voxel_audit),
            "mean": float(statistics.mean(row["total_voxel_count"] for row in voxel_audit)),
            "per_frame": voxel_audit,
        },
        "models": models,
        "speedup": speedup,
        "per_scene_accuracy": per_scene,
        "frames": frame_reports,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--candidate-engine", required=True, type=Path)
    parser.add_argument("--baseline-engine", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--fp32-checkpoint", required=True, type=Path)
    parser.add_argument("--intensity-calibration", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--heal-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--score-floor", type=float, default=0.02)
    parser.add_argument(
        "--voxelization-backend", choices=("gpu", "cpu"), default="gpu"
    )
    parser.add_argument("--raw-intensity", action="store_true")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = evaluate(arguments)
    destination = arguments.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_keys = ("success", "scene_count", "frame_count", "models", "speedup")
    print(json.dumps({key: report[key] for key in summary_keys}, indent=2))


if __name__ == "__main__":
    main()
