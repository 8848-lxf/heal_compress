"""Build and apply a no-leak CARLA-to-DAIR LiDAR intensity calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np


ENDPOINT_FIELDS = {
    "vehicle": "vehicle_pointcloud_path",
    "infrastructure": "infrastructure_pointcloud_path",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _range_mask(points: np.ndarray, lidar_range: Sequence[float]) -> np.ndarray:
    limits = np.asarray(lidar_range, dtype=np.float32)
    return (
        (points[:, 0] > limits[0])
        & (points[:, 0] < limits[3])
        & (points[:, 1] > limits[1])
        & (points[:, 1] < limits[4])
        & (points[:, 2] > limits[2])
        & (points[:, 2] < limits[5])
    )


def _ego_mask(points: np.ndarray) -> np.ndarray:
    return ~(
        (points[:, 0] >= -1.95)
        & (points[:, 0] <= 2.95)
        & (points[:, 1] >= -1.1)
        & (points[:, 1] <= 1.1)
    )


def filtered_intensity(
    points: np.ndarray, lidar_range: Sequence[float], *, remove_ego: bool
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 4:
        raise ValueError(f"expected Nx4 points, got {values.shape}")
    mask = _range_mask(values, lidar_range)
    if remove_ego:
        mask &= _ego_mask(values)
    result = values[mask, 3]
    return result[np.isfinite(result)]


class _Histogram:
    def __init__(self, bins: int) -> None:
        if int(bins) < 256:
            raise ValueError("histogram requires at least 256 bins")
        self.bins = int(bins)
        self.counts = np.zeros(self.bins, dtype=np.uint64)
        self.point_count = 0
        self.value_sum = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def add(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        if not array.size:
            return
        if float(array.min()) < 0.0 or float(array.max()) > 1.0:
            raise ValueError(
                f"intensity outside [0,1]: min={array.min()} max={array.max()}"
            )
        indices = np.minimum((array * self.bins).astype(np.int64), self.bins - 1)
        self.counts += np.bincount(indices, minlength=self.bins).astype(np.uint64)
        self.point_count += int(array.size)
        self.value_sum += float(array.sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(array.min()))
        self.maximum = max(self.maximum, float(array.max()))

    def quantiles(self, probabilities: np.ndarray) -> np.ndarray:
        if self.point_count <= 0:
            raise RuntimeError("cannot extract quantiles from an empty histogram")
        cumulative = np.cumsum(self.counts, dtype=np.uint64)
        ranks = np.floor(probabilities * (self.point_count - 1)).astype(np.uint64) + 1
        indices = np.searchsorted(cumulative, ranks, side="left")
        result = (indices.astype(np.float64) + 0.5) / float(self.bins)
        result[probabilities <= 0.0] = self.minimum
        result[probabilities >= 1.0] = self.maximum
        return result.astype(np.float32)

    def summary(self) -> Dict[str, Any]:
        if self.point_count <= 0:
            raise RuntimeError("empty intensity histogram")
        return {
            "point_count": self.point_count,
            "mean": self.value_sum / self.point_count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "histogram_bins": self.bins,
        }


def _manifest_frames(manifest: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for scene in manifest["scenes"]:
        yield from scene["frames"]


def _source_histograms(
    data_root: Path,
    lidar_range: Sequence[float],
    bins: int,
) -> tuple[Dict[str, _Histogram], Mapping[str, Any]]:
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    histograms = {name: _Histogram(bins) for name in ENDPOINT_FIELDS}
    frame_count = 0
    for frame in _manifest_frames(manifest):
        path = data_root / str(frame["relative_path"])
        with np.load(path) as payload:
            histograms["vehicle"].add(
                filtered_intensity(
                    payload["vehicle_points"], lidar_range, remove_ego=True
                )
            )
            histograms["infrastructure"].add(
                filtered_intensity(
                    payload["infrastructure_points"], lidar_range, remove_ego=False
                )
            )
        frame_count += 1
    if frame_count <= 0:
        raise RuntimeError(f"CARLA source manifest has no frames: {manifest_path}")
    maps = sorted({str(scene["map"]) for scene in manifest["scenes"]})
    return histograms, {
        "data_root": str(data_root),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "frame_count": frame_count,
        "maps": maps,
    }


def _target_histograms(
    dair_root: Path,
    train_split: Path,
    cooperative_info: Path,
    lidar_range: Sequence[float],
    bins: int,
) -> tuple[Dict[str, _Histogram], Mapping[str, Any]]:
    from pypcd import pypcd

    frame_ids = json.loads(train_split.read_text(encoding="utf-8"))
    cooperative_rows = json.loads(cooperative_info.read_text(encoding="utf-8"))
    by_vehicle = {
        Path(row["vehicle_pointcloud_path"]).stem: row for row in cooperative_rows
    }
    histograms = {name: _Histogram(bins) for name in ENDPOINT_FIELDS}
    pairs = []
    missing = []
    for vehicle_id in frame_ids:
        row = by_vehicle.get(str(vehicle_id))
        if row is None:
            missing.append(str(vehicle_id))
            continue
        pair = {"vehicle_frame_id": str(vehicle_id)}
        loaded = {}
        for endpoint, field in ENDPOINT_FIELDS.items():
            relative = str(row[field])
            path = dair_root / relative
            if not path.is_file():
                missing.append(f"{vehicle_id}:{relative}")
                break
            cloud = pypcd.PointCloud.from_path(str(path))
            points = np.empty((cloud.points, 4), dtype=np.float32)
            for index, name in enumerate(("x", "y", "z", "intensity")):
                points[:, index] = np.asarray(cloud.pc_data[name], dtype=np.float32)
            points[:, 3] /= 256.0
            loaded[endpoint] = filtered_intensity(
                points, lidar_range, remove_ego=endpoint == "vehicle"
            )
            pair[f"{endpoint}_frame_id"] = Path(relative).stem
        else:
            for endpoint, values in loaded.items():
                histograms[endpoint].add(values)
            pairs.append(pair)
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"DAIR train pairing failed closed for {len(missing)} entries: {preview}"
        )
    pair_bytes = json.dumps(pairs, sort_keys=True, separators=(",", ":")).encode()
    return histograms, {
        "dair_root": str(dair_root),
        "train_split": str(train_split),
        "train_split_sha256": _sha256(train_split),
        "cooperative_info": str(cooperative_info),
        "cooperative_info_sha256": _sha256(cooperative_info),
        "train_pair_count": len(pairs),
        "pair_ids_sha256": hashlib.sha256(pair_bytes).hexdigest(),
        "normalization": "HEAL read_pcd intensity / 256.0",
    }


def build_calibration(arguments: argparse.Namespace) -> Mapping[str, Any]:
    lidar_range = [float(value) for value in arguments.lidar_range]
    probabilities = np.linspace(0.0, 1.0, int(arguments.quantiles), dtype=np.float64)
    source, source_provenance = _source_histograms(
        arguments.carla_data.expanduser().resolve(), lidar_range, arguments.histogram_bins
    )
    held_out_maps = sorted(set(arguments.held_out_map or []))
    overlap = set(source_provenance["maps"]) & set(held_out_maps)
    if overlap:
        raise RuntimeError(
            f"CARLA calibration/test map leakage detected: {sorted(overlap)}"
        )
    target, target_provenance = _target_histograms(
        arguments.dair_root.expanduser().resolve(),
        arguments.train_split.expanduser().resolve(),
        arguments.cooperative_info.expanduser().resolve(),
        lidar_range,
        arguments.histogram_bins,
    )
    endpoints: MutableMapping[str, Any] = {}
    for endpoint in ENDPOINT_FIELDS:
        endpoints[endpoint] = {
            "probabilities": probabilities.tolist(),
            "source_quantiles": source[endpoint].quantiles(probabilities).tolist(),
            "target_quantiles": target[endpoint].quantiles(probabilities).tolist(),
            "source": source[endpoint].summary(),
            "target": target[endpoint].summary(),
        }
    report = {
        "schema_version": "heal-carla-intensity-calibration-v1",
        "method": "endpoint_empirical_quantile_map",
        "leakage_policy": (
            "DAIR train target; CARLA calibration maps disjoint from held-out "
            "test maps"
        ),
        "lidar_range": lidar_range,
        "held_out_test_maps": held_out_maps,
        "source_provenance": source_provenance,
        "target_provenance": target_provenance,
        "endpoints": endpoints,
    }
    _write_json(arguments.output.expanduser().resolve(), report)
    return report


class IntensityCalibration:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        if payload.get("schema_version") != "heal-carla-intensity-calibration-v1":
            raise ValueError("unsupported intensity calibration schema")
        self.payload = payload

    @classmethod
    def from_path(cls, path: Path) -> "IntensityCalibration":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def apply(self, points: np.ndarray, endpoint: str) -> np.ndarray:
        if endpoint not in ENDPOINT_FIELDS:
            raise ValueError(f"unknown endpoint: {endpoint}")
        config = self.payload["endpoints"][endpoint]
        source = np.asarray(config["source_quantiles"], dtype=np.float32)
        target = np.asarray(config["target_quantiles"], dtype=np.float32)
        unique, inverse = np.unique(source, return_inverse=True)
        sums = np.zeros(unique.shape, dtype=np.float64)
        counts = np.zeros(unique.shape, dtype=np.int64)
        np.add.at(sums, inverse, target)
        np.add.at(counts, inverse, 1)
        mapped_target = (sums / counts).astype(np.float32)
        result = np.asarray(points, dtype=np.float32).copy()
        result[:, 3] = np.interp(
            result[:, 3], unique, mapped_target, left=mapped_target[0], right=mapped_target[-1]
        ).astype(np.float32)
        return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carla-data", required=True, type=Path)
    parser.add_argument("--dair-root", required=True, type=Path)
    parser.add_argument("--train-split", required=True, type=Path)
    parser.add_argument("--cooperative-info", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--held-out-map", action="append")
    parser.add_argument("--quantiles", type=int, default=257)
    parser.add_argument("--histogram-bins", type=int, default=1048576)
    parser.add_argument(
        "--lidar-range",
        type=float,
        nargs=6,
        default=[-102.4, -51.2, -3.5, 102.4, 51.2, 1.5],
    )
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = build_calibration(arguments)
    print(
        json.dumps(
            {
                "output": str(arguments.output.expanduser().resolve()),
                "train_pair_count": report["target_provenance"]["train_pair_count"],
                "held_out_test_maps": report["held_out_test_maps"],
                "endpoint_means": {
                    name: {
                        "source": values["source"]["mean"],
                        "target": values["target"]["mean"],
                    }
                    for name, values in report["endpoints"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
