"""Configuration parsing for DAIR-aligned CARLA collection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class LidarProfile:
    name: str
    channels: int
    rotation_frequency: float
    horizontal_fov: float
    upper_fov: float
    lower_fov: float
    range_m: float
    points_per_second: int
    physical_height_m: float
    canonical_height_m: float

    @classmethod
    def from_mapping(cls, name: str, values: Dict[str, Any]) -> "LidarProfile":
        profile = cls(
            name=str(name),
            channels=int(values["channels"]),
            rotation_frequency=float(values["rotation_frequency"]),
            horizontal_fov=float(values["horizontal_fov"]),
            upper_fov=float(values["upper_fov"]),
            lower_fov=float(values["lower_fov"]),
            range_m=float(values["range_m"]),
            points_per_second=int(values["points_per_second"]),
            physical_height_m=float(values["physical_height_m"]),
            canonical_height_m=float(values["canonical_height_m"]),
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if self.channels <= 0 or self.points_per_second <= 0:
            raise ValueError(f"{self.name}: channels and points_per_second must be positive")
        if self.rotation_frequency <= 0 or self.range_m <= 0:
            raise ValueError(f"{self.name}: frequency and range must be positive")
        if not 0 < self.horizontal_fov <= 360:
            raise ValueError(f"{self.name}: horizontal_fov must be in (0, 360]")
        if self.lower_fov >= self.upper_fov:
            raise ValueError(f"{self.name}: lower_fov must be below upper_fov")
        if self.physical_height_m <= 0 or self.canonical_height_m <= 0:
            raise ValueError(f"{self.name}: sensor heights must be positive")

    def blueprint_attributes(self, sensor_tick: float) -> Dict[str, str]:
        return {
            "channels": str(self.channels),
            "rotation_frequency": str(self.rotation_frequency),
            "horizontal_fov": str(self.horizontal_fov),
            "upper_fov": str(self.upper_fov),
            "lower_fov": str(self.lower_fov),
            "range": str(self.range_m),
            "points_per_second": str(self.points_per_second),
            "sensor_tick": str(float(sensor_tick)),
            "dropoff_general_rate": "0.0",
            "dropoff_intensity_limit": "1.0",
            "dropoff_zero_intensity": "0.0",
        }


def load_collection_config(path: str) -> Dict[str, Any]:
    import yaml

    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"collection config must be a mapping: {source}")
    simulation = payload.get("simulation", {})
    fixed_delta = float(simulation.get("fixed_delta_seconds", 0.1))
    sensor_tick = float(simulation.get("sensor_tick_seconds", fixed_delta))
    if fixed_delta <= 0 or sensor_tick <= 0:
        raise ValueError("simulation tick values must be positive")
    if abs(round(sensor_tick / fixed_delta) * fixed_delta - sensor_tick) > 1e-6:
        raise ValueError("sensor_tick_seconds must be a multiple of fixed_delta_seconds")
    profiles = payload.get("lidar_profiles", {})
    if set(profiles) != {"vehicle", "infrastructure"}:
        raise ValueError("lidar_profiles must contain vehicle and infrastructure")
    parsed_profiles = {
        key: LidarProfile.from_mapping(key, value) for key, value in profiles.items()
    }
    scenes: List[Dict[str, Any]] = list(payload.get("scenes", []))
    if len(scenes) < 5:
        raise ValueError("at least five CARLA scenes are required")
    scene_ids = [str(scene.get("id", "")) for scene in scenes]
    if not all(scene_ids) or len(scene_ids) != len(set(scene_ids)):
        raise ValueError("scene ids must be non-empty and unique")
    payload["_source"] = str(source)
    payload["_profiles"] = parsed_profiles
    return payload
