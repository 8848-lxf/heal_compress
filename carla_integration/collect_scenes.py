"""Collect deterministic, DAIR-aligned cooperative LiDAR scenes from CARLA."""

from __future__ import annotations

import argparse
import copy
import json
import math
import queue
import random
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import LidarProfile, load_collection_config
from .coordinates import (
    boxes_to_corners,
    canonical_sensor_to_world,
    canonicalize_lidar,
    carla_transform_to_right_handed,
    pairwise_transforms,
    point_counts_in_boxes,
    transform_points,
)


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/carla/dair_v2x_aligned.yaml"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _progress(scene_id: str, stage: str) -> None:
    print(f"[carla-collect] scene={scene_id} stage={stage}", file=sys.stderr, flush=True)


def _matrix(transform: Any) -> np.ndarray:
    return np.asarray(transform.get_matrix(), dtype=np.float64)


def _weather(carla: Any, name: str) -> Any:
    value = getattr(carla.WeatherParameters, str(name), None)
    if value is None:
        raise ValueError(f"CARLA weather preset does not exist: {name}")
    return value


def _driving_waypoint(world: Any, location: Any) -> Any:
    import carla

    waypoint = world.get_map().get_waypoint(
        location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        raise RuntimeError(
            "unable to project sensor location onto a driving lane: "
            f"({location.x:.3f}, {location.y:.3f}, {location.z:.3f})"
        )
    return waypoint


def _height_above_road(world: Any, transform: Any) -> float:
    waypoint = _driving_waypoint(world, transform.location)
    return float(transform.location.z - waypoint.transform.location.z)


def _set_lidar_attributes(blueprint: Any, profile: LidarProfile, sensor_tick: float) -> None:
    attributes = profile.blueprint_attributes(sensor_tick)
    # CARLA 0.9.10 can skip the frame immediately following a 0.1 s sensor
    # deadline. The world itself advances at 0.1 s, so sampling every world
    # tick preserves the configured 10 Hz acquisition rate without gaps.
    attributes["sensor_tick"] = "0.0"
    if not blueprint.has_attribute("horizontal_fov"):
        # CARLA 0.9.10 always emits a full revolution. The configured ray rate
        # includes the 360/100 density compensation; crop to the requested
        # sector after acquisition.
        attributes.pop("horizontal_fov")
    for name, value in attributes.items():
        if not blueprint.has_attribute(name):
            raise RuntimeError(f"CARLA lidar blueprint lacks required attribute: {name}")
        blueprint.set_attribute(name, value)


def _spawn_ego(world: Any, spawn_points: Sequence[Any], index: int) -> Any:
    library = world.get_blueprint_library()
    candidates = list(library.filter("vehicle.tesla.model3")) or list(library.filter("vehicle.*"))
    if not candidates:
        raise RuntimeError("CARLA map has no vehicle blueprints")
    for offset in range(len(spawn_points)):
        transform = spawn_points[(int(index) + offset) % len(spawn_points)]
        blueprint = candidates[offset % len(candidates)]
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "hero")
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            return actor
    raise RuntimeError("unable to spawn ego vehicle at any map spawn point")


def _spawn_traffic(
    world: Any,
    ego: Any,
    spawn_points: Sequence[Any],
    count: int,
    traffic_manager_port: int,
    rng: random.Random,
) -> List[Any]:
    library = world.get_blueprint_library()
    blueprints = [bp for bp in library.filter("vehicle.*") if int(bp.get_attribute("number_of_wheels")) == 4]
    if not blueprints:
        raise RuntimeError("CARLA map has no four-wheel vehicle blueprints")
    candidates = list(spawn_points)
    rng.shuffle(candidates)
    actors = []
    ego_location = ego.get_location()
    for transform in candidates:
        if len(actors) >= int(count):
            break
        if transform.location.distance(ego_location) < 8.0:
            continue
        blueprint = rng.choice(blueprints)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "autopilot")
        if blueprint.has_attribute("color"):
            colors = blueprint.get_attribute("color").recommended_values
            if colors:
                blueprint.set_attribute("color", rng.choice(colors))
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            actor.set_autopilot(True, int(traffic_manager_port))
            actors.append(actor)
    return actors


def _rsu_transform(world: Any, ego: Any, scene: Dict[str, Any], profile: LidarProfile) -> Any:
    import carla

    ego_waypoint = _driving_waypoint(world, ego.get_location())
    forward = float(scene.get("rsu_forward_m", 20.0))
    later = float(scene.get("rsu_lateral_m", 5.0))
    next_waypoints = ego_waypoint.next(max(forward, 1.0))
    waypoint = next_waypoints[0] if next_waypoints else ego_waypoint
    right = waypoint.transform.get_right_vector()
    location = carla.Location(
        x=waypoint.transform.location.x + right.x * later,
        y=waypoint.transform.location.y + right.y * later,
        z=waypoint.transform.location.z + profile.physical_height_m,
    )
    return carla.Transform(location, carla.Rotation(yaw=waypoint.transform.rotation.yaw))


def _spawn_lidars(
    world: Any,
    ego: Any,
    scene: Dict[str, Any],
    vehicle_profile: LidarProfile,
    infrastructure_profile: LidarProfile,
    sensor_tick: float,
) -> Tuple[Any, Any]:
    import carla

    library = world.get_blueprint_library()
    vehicle_bp = library.find("sensor.lidar.ray_cast")
    infrastructure_bp = library.find("sensor.lidar.ray_cast")
    _set_lidar_attributes(vehicle_bp, vehicle_profile, sensor_tick)
    _set_lidar_attributes(infrastructure_bp, infrastructure_profile, sensor_tick)

    actor_transform = ego.get_transform()
    road = _driving_waypoint(world, actor_transform.location)
    actor_origin_above_road = float(actor_transform.location.z - road.transform.location.z)
    relative_height = vehicle_profile.physical_height_m - actor_origin_above_road
    if relative_height <= 0:
        raise RuntimeError(
            "vehicle lidar mount would be below the actor origin: "
            f"actor_origin_above_road={actor_origin_above_road:.3f}"
        )
    vehicle_transform = carla.Transform(
        carla.Location(x=0.0, y=0.0, z=relative_height),
        carla.Rotation(yaw=0.0),
    )
    vehicle_sensor = world.spawn_actor(
        vehicle_bp,
        vehicle_transform,
        attach_to=ego,
        attachment_type=carla.AttachmentType.Rigid,
    )
    print("[carla-collect] lidar=vehicle stage=spawned", file=sys.stderr, flush=True)
    try:
        infrastructure_sensor = world.spawn_actor(
            infrastructure_bp,
            _rsu_transform(world, ego, scene, infrastructure_profile),
        )
    except Exception:
        vehicle_sensor.destroy()
        raise
    print("[carla-collect] lidar=infrastructure stage=spawned", file=sys.stderr, flush=True)
    return vehicle_sensor, infrastructure_sensor


def _measurement_for_frame(values: "queue.Queue[Any]", frame: int, timeout: float) -> Any:
    deadline = time.monotonic() + float(timeout)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for CARLA sensor frame {frame}")
        measurement = values.get(timeout=remaining)
        if int(measurement.frame) == int(frame):
            return measurement
        if int(measurement.frame) > int(frame):
            raise RuntimeError(
                f"sensor skipped synchronous frame {frame}; received {measurement.frame}"
            )


def _lidar_array(measurement: Any) -> np.ndarray:
    values = np.frombuffer(measurement.raw_data, dtype=np.float32)
    if values.size % 4:
        raise RuntimeError(f"CARLA lidar frame has invalid float count: {values.size}")
    return values.reshape((-1, 4)).copy()


def _crop_horizontal_fov(points: np.ndarray, horizontal_fov: float) -> np.ndarray:
    if float(horizontal_fov) >= 360.0:
        return points
    half_angle = math.radians(float(horizontal_fov) / 2.0)
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    return points[np.abs(azimuth) <= half_angle]


def _bbox_transform(actor: Any) -> np.ndarray:
    import carla

    bbox = actor.bounding_box
    bbox_local = carla.Transform(bbox.location, bbox.rotation)
    return _matrix(actor.get_transform()) @ _matrix(bbox_local)


def _ground_truth(
    world: Any,
    ego: Any,
    ego_canonical_to_world: np.ndarray,
    lidar_range: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    inverse_ego = np.linalg.inv(ego_canonical_to_world)
    centers = []
    actor_ids = []
    for actor in world.get_actors().filter("vehicle.*"):
        if int(actor.id) == int(ego.id) or not actor.is_alive:
            continue
        bbox_to_world = carla_transform_to_right_handed(_bbox_transform(actor))
        bbox_to_ego = inverse_ego @ bbox_to_world
        center = bbox_to_ego[:3, 3]
        if not (
            float(lidar_range[0]) <= center[0] <= float(lidar_range[3])
            and float(lidar_range[1]) <= center[1] <= float(lidar_range[4])
        ):
            continue
        bbox = actor.bounding_box
        yaw = math.atan2(float(bbox_to_ego[1, 0]), float(bbox_to_ego[0, 0]))
        centers.append(
            [
                float(center[0]),
                float(center[1]),
                float(center[2]),
                float(bbox.extent.x * 2.0),
                float(bbox.extent.y * 2.0),
                float(bbox.extent.z * 2.0),
                yaw,
            ]
        )
        actor_ids.append(int(actor.id))
    return boxes_to_corners(centers), np.asarray(actor_ids, dtype=np.int64)


def _collect_scene(
    client: Any,
    config: Dict[str, Any],
    scene: Dict[str, Any],
    output_root: Path,
    frames_per_scene: int,
    seed: int,
    static_ego: bool,
) -> Dict[str, Any]:
    simulation = config["simulation"]
    fixed_delta = float(simulation["fixed_delta_seconds"])
    sensor_tick = float(simulation["sensor_tick_seconds"])
    timeout = float(simulation.get("client_timeout_seconds", 30.0))
    scene_id = str(scene["id"])
    _progress(scene_id, "load_world")
    world = client.load_world(str(scene["map"]))
    world.set_weather(_weather(__import__("carla"), str(scene["weather"])))
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta
    # The server itself runs RenderOffScreen. CARLA 0.9.10's world-level
    # no_rendering_mode also suppresses ray-cast sensor updates.
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    _progress(scene_id, "synchronous_world_ready")

    # Keep the Traffic Manager port independent of the CARLA RPC port and
    # deterministic. CARLA 0.9.10 reports a Git hash, not a semantic version.
    traffic_manager_port = (
        20000
        + int(simulation.get("traffic_manager_port_offset", 1000))
        + int(seed % 5000)
    )
    traffic_manager = client.get_trafficmanager(traffic_manager_port)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(int(seed))
    traffic_manager.global_percentage_speed_difference(15.0)
    _progress(scene_id, "traffic_manager_ready")

    actors: List[Any] = []
    sensors: List[Any] = []
    vehicle_queue: "queue.Queue[Any]" = queue.Queue()
    infrastructure_queue: "queue.Queue[Any]" = queue.Queue()
    scene_dir = output_root / str(scene["id"])
    scene_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    rng = random.Random(int(seed))
    try:
        spawn_points = list(world.get_map().get_spawn_points())
        if not spawn_points:
            raise RuntimeError(f"map {scene['map']} has no vehicle spawn points")
        ego = _spawn_ego(world, spawn_points, int(scene.get("ego_spawn_index", 0)))
        actors.append(ego)
        if not static_ego:
            ego.set_autopilot(True, traffic_manager_port)
        _progress(scene_id, "ego_spawned")
        for _ in range(3):
            world.tick()
        _progress(scene_id, "ego_settled")
        traffic = _spawn_traffic(
            world,
            ego,
            spawn_points,
            int(scene.get("traffic_vehicles", 0)),
            traffic_manager_port,
            rng,
        )
        actors.extend(traffic)
        _progress(scene_id, f"traffic_spawned_{len(traffic)}")
        vehicle_profile = config["_profiles"]["vehicle"]
        infrastructure_profile = config["_profiles"]["infrastructure"]
        vehicle_sensor, infrastructure_sensor = _spawn_lidars(
            world,
            ego,
            scene,
            vehicle_profile,
            infrastructure_profile,
            sensor_tick,
        )
        sensors.extend([vehicle_sensor, infrastructure_sensor])
        _progress(scene_id, "lidars_spawned")
        vehicle_sensor.listen(vehicle_queue.put)
        infrastructure_sensor.listen(infrastructure_queue.put)
        _progress(scene_id, "lidar_callbacks_ready")

        for _ in range(int(simulation.get("warmup_ticks", 15))):
            frame = world.tick()
            _measurement_for_frame(vehicle_queue, frame, timeout)
            _measurement_for_frame(infrastructure_queue, frame, timeout)
        _progress(scene_id, "warmup_complete")

        lidar_range = config["model_contract"]["lidar_range"]
        for sample_index in range(int(frames_per_scene)):
            frame = world.tick()
            vehicle_measurement = _measurement_for_frame(vehicle_queue, frame, timeout)
            infrastructure_measurement = _measurement_for_frame(
                infrastructure_queue, frame, timeout
            )
            vehicle_transform = vehicle_sensor.get_transform()
            infrastructure_transform = infrastructure_sensor.get_transform()
            vehicle_height = _height_above_road(world, vehicle_transform)
            infrastructure_height = _height_above_road(world, infrastructure_transform)
            vehicle_offset = vehicle_height - vehicle_profile.canonical_height_m
            infrastructure_offset = (
                infrastructure_height - infrastructure_profile.canonical_height_m
            )
            vehicle_raw = _lidar_array(vehicle_measurement)
            infrastructure_raw = _lidar_array(infrastructure_measurement)
            vehicle_points = canonicalize_lidar(
                _crop_horizontal_fov(vehicle_raw, vehicle_profile.horizontal_fov),
                vehicle_offset,
            )
            infrastructure_points = canonicalize_lidar(
                _crop_horizontal_fov(
                    infrastructure_raw, infrastructure_profile.horizontal_fov
                ),
                infrastructure_offset,
            )
            vehicle_to_world = canonical_sensor_to_world(
                _matrix(vehicle_transform), vehicle_offset
            )
            infrastructure_to_world = canonical_sensor_to_world(
                _matrix(infrastructure_transform), infrastructure_offset
            )
            pairwise = pairwise_transforms([vehicle_to_world, infrastructure_to_world])
            gt_boxes, gt_actor_ids = _ground_truth(
                world, ego, vehicle_to_world, lidar_range
            )
            vehicle_hits = point_counts_in_boxes(vehicle_points[:, :3], gt_boxes)
            infrastructure_in_ego = transform_points(
                infrastructure_points, pairwise[1, 0]
            )
            infrastructure_hits = point_counts_in_boxes(
                infrastructure_in_ego, gt_boxes
            )
            gt_lidar_hits = vehicle_hits + infrastructure_hits
            name = f"{sample_index:06d}.npz"
            path = scene_dir / name
            np.savez_compressed(
                str(path),
                vehicle_points=vehicle_points,
                infrastructure_points=infrastructure_points,
                pairwise_t_matrix=pairwise,
                sensor_to_world=np.stack(
                    [vehicle_to_world, infrastructure_to_world], axis=0
                ).astype(np.float32),
                gt_boxes=gt_boxes.astype(np.float32),
                gt_actor_ids=gt_actor_ids,
                gt_lidar_hits=gt_lidar_hits,
                gt_vehicle_lidar_hits=vehicle_hits,
                gt_infrastructure_lidar_hits=infrastructure_hits,
                carla_frame=np.asarray([frame], dtype=np.int64),
            )
            rows.append(
                {
                    "sample_id": f"{scene['id']}/{sample_index:06d}",
                    "relative_path": f"{scene['id']}/{name}",
                    "carla_frame": int(frame),
                    "vehicle_point_count": int(vehicle_points.shape[0]),
                    "infrastructure_point_count": int(infrastructure_points.shape[0]),
                    "vehicle_raw_point_count": int(vehicle_raw.shape[0]),
                    "infrastructure_raw_point_count": int(infrastructure_raw.shape[0]),
                    "gt_count": int(gt_boxes.shape[0]),
                    "gt_visible_1plus_count": int((gt_lidar_hits >= 1).sum()),
                    "gt_visible_5plus_count": int((gt_lidar_hits >= 5).sum()),
                    "vehicle_height_above_road_m": round(vehicle_height, 6),
                    "infrastructure_height_above_road_m": round(
                        infrastructure_height, 6
                    ),
                    "vehicle_local_z_offset_m": round(vehicle_offset, 6),
                    "infrastructure_local_z_offset_m": round(
                        infrastructure_offset, 6
                    ),
                }
            )
            _progress(scene_id, f"frame_saved_{sample_index:06d}")
        report = {
            "scene_id": str(scene["id"]),
            "map": str(scene["map"]),
            "weather": str(scene["weather"]),
            "requested_traffic_vehicles": int(scene.get("traffic_vehicles", 0)),
            "spawned_traffic_vehicles": len(traffic),
            "evaluated_frames": len(rows),
            "frames": rows,
        }
        _write_json(scene_dir / "scene_manifest.json", report)
        _progress(scene_id, "complete")
        return report
    finally:
        for sensor in sensors:
            try:
                sensor.stop()
            except Exception:
                pass
        for actor in actors:
            try:
                if str(actor.type_id).startswith("vehicle."):
                    actor.set_autopilot(False, traffic_manager_port)
            except Exception:
                pass
        traffic_manager.set_synchronous_mode(False)
        current_settings = world.get_settings()
        current_settings.synchronous_mode = False
        current_settings.fixed_delta_seconds = None
        world.apply_settings(current_settings)
        for actor in reversed(sensors + actors):
            try:
                actor.destroy()
            except Exception:
                pass


def collect(args: argparse.Namespace) -> Dict[str, Any]:
    import carla

    config = copy.deepcopy(load_collection_config(args.config))
    simulation = config["simulation"]
    if args.warmup_ticks is not None:
        if int(args.warmup_ticks) < 0:
            raise ValueError("warmup_ticks must be non-negative")
        simulation["warmup_ticks"] = int(args.warmup_ticks)
    frames_per_scene = int(
        args.frames_per_scene
        if args.frames_per_scene is not None
        else simulation.get("frames_per_scene", 12)
    )
    if frames_per_scene <= 0:
        raise ValueError("frames_per_scene must be positive")
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected = set(args.scene or [])
    scenes = [scene for scene in config["scenes"] if not selected or scene["id"] in selected]
    missing = selected - {scene["id"] for scene in scenes}
    if missing:
        raise ValueError(f"unknown scene ids: {sorted(missing)}")
    if args.traffic_vehicles is not None:
        if int(args.traffic_vehicles) < 0:
            raise ValueError("traffic_vehicles must be non-negative")
        for scene in scenes:
            scene["traffic_vehicles"] = int(args.traffic_vehicles)
    if args.infrastructure_points_per_second is not None:
        points_per_second = int(args.infrastructure_points_per_second)
        if points_per_second <= 0:
            raise ValueError("infrastructure_points_per_second must be positive")
        profile = replace(
            config["_profiles"]["infrastructure"],
            points_per_second=points_per_second,
        )
        profile.validate()
        config["_profiles"]["infrastructure"] = profile
        config["lidar_profiles"]["infrastructure"][
            "points_per_second"
        ] = points_per_second

    client = carla.Client(str(args.host), int(args.port))
    client.set_timeout(float(simulation.get("client_timeout_seconds", 30.0)))
    seed = int(simulation.get("random_seed", 8848))
    reports = []
    for index, scene in enumerate(scenes):
        reports.append(
            _collect_scene(
                client,
                config,
                scene,
                output_root,
                frames_per_scene,
                seed + index,
                bool(args.static_ego),
            )
        )
    payload = {
        "schema_version": "heal-carla-cooperative-frames-v2",
        "success": len(reports) == len(scenes) and all(
            report["evaluated_frames"] == frames_per_scene for report in reports
        ),
        "carla_server_version": client.get_server_version(),
        "carla_client_version": client.get_client_version(),
        "config_path": config["_source"],
        "point_frontend": config["model_contract"]["point_frontend"],
        "tensorrt_inputs": config["model_contract"]["tensorrt_inputs"],
        "lidar_profiles": {
            name: asdict(profile) for name, profile in config["_profiles"].items()
        },
        "carla_compatibility": config.get("carla_0_9_10_compatibility", {}),
        "scene_count": len(reports),
        "frame_count": sum(report["evaluated_frames"] for report in reports),
        "scenes": reports,
    }
    _write_json(output_root / "manifest.json", payload)
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=27896)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames-per-scene", type=int)
    parser.add_argument("--warmup-ticks", type=int)
    parser.add_argument("--traffic-vehicles", type=int)
    parser.add_argument("--infrastructure-points-per-second", type=int)
    parser.add_argument("--static-ego", action="store_true")
    parser.add_argument("--scene", action="append", help="collect only this configured scene id")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    report = collect(parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
