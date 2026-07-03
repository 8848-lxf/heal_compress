from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEPLOY_MODE = "single_engine_maxK"
FIXED_K = 29696
PRECISION_TO_PROFILE = {
    "FP32": "TRT_FP32",
    "FP16": "TRT_FP16",
    "INT8": "TRT_INT8_QDQ",
    "TRT_FP32": "TRT_FP32",
    "TRT_FP16": "TRT_FP16",
    "TRT_INT8_QDQ": "TRT_INT8_QDQ",
}
PROFILE_TO_WEIGHT = {
    "TRT_FP32": "FP32",
    "TRT_FP16": "FP16",
    "TRT_INT8_QDQ": "INT8",
}


def now_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def precision_to_profile(precision: str) -> str:
    value = str(precision).upper()
    if value not in PRECISION_TO_PROFILE:
        raise ValueError(f"unsupported precision/profile: {precision}")
    return PRECISION_TO_PROFILE[value]


def profile_to_weight_precision(profile: str) -> str:
    return PROFILE_TO_WEIGHT.get(precision_to_profile(profile), str(profile).upper())


def _norm_shape(value: Any) -> int | tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return int(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class LatencyLUTKey:
    deploy_mode: str
    fixed_K: int
    module_name: str
    block_name: str
    block_type: str
    H: int | None = None
    W: int | None = None
    C_in: int | None = None
    C_mid: int | None = None
    C_out: int | None = None
    kernel_size: int | tuple[int, ...] | None = None
    stride: int | tuple[int, ...] | None = None
    padding: int | tuple[int, ...] | None = None
    dilation: int | tuple[int, ...] | None = None
    groups: int | None = None
    batch_size: int = 1
    precision_profile: str = "TRT_FP16"
    weight_precision: str = "FP16"
    activation_precision: str = "FP16"
    compute_precision: str = "FP16"
    plugin_flag: bool = False
    plugin_name: str | None = None
    plugin_version: str | None = None
    trt_version: str | None = None
    cuda_version: str | None = None
    gpu_name: str | None = None
    builder_config_hash: str | None = None
    src_precision: str | None = None
    dst_precision: str | None = None
    tensor_dtype_before: str | None = None
    tensor_dtype_after: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.deploy_mode != DEPLOY_MODE:
            raise ValueError(f"Latency LUT only supports deploy_mode={DEPLOY_MODE}, got {self.deploy_mode}")
        if int(self.fixed_K) != FIXED_K:
            raise ValueError(f"Latency LUT only supports fixed_K={FIXED_K}, got {self.fixed_K}")
        object.__setattr__(self, "fixed_K", int(self.fixed_K))
        object.__setattr__(self, "precision_profile", precision_to_profile(self.precision_profile))
        if self.precision_profile == "TRT_INT8_QDQ":
            object.__setattr__(self, "weight_precision", "INT8")
            object.__setattr__(self, "activation_precision", "INT8")
            object.__setattr__(self, "compute_precision", "INT8")
        object.__setattr__(self, "kernel_size", _norm_shape(self.kernel_size))
        object.__setattr__(self, "stride", _norm_shape(self.stride))
        object.__setattr__(self, "padding", _norm_shape(self.padding))
        object.__setattr__(self, "dilation", _norm_shape(self.dilation))
        for name in ("H", "W", "C_in", "C_mid", "C_out", "groups", "batch_size"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, int(value))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["key_hash"] = self.stable_hash()
        return _jsonable(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LatencyLUTKey":
        payload = dict(data)
        payload.pop("key_hash", None)
        return cls(**payload)

    def canonical_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return _jsonable(data)

    def stable_json(self) -> str:
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def stable_hash(self) -> str:
        return hashlib.sha256(self.stable_json().encode("utf-8")).hexdigest()

    def comparable_tuple(self) -> tuple[Any, ...]:
        return (
            self.deploy_mode,
            self.fixed_K,
            self.module_name,
            self.block_name,
            self.block_type,
            self.H,
            self.W,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.batch_size,
            self.precision_profile,
            self.plugin_flag,
            self.plugin_name,
            self.src_precision,
            self.dst_precision,
        )


@dataclass
class LatencyRecord:
    key: LatencyLUTKey
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    latency_std_ms: float
    num_warmup: int
    num_repeat: int
    timing_method: str
    engine_hash: str | None = None
    onnx_hash: str | None = None
    created_at: str = field(default_factory=now_timestamp)
    status: str = "success"
    error: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key_hash(self) -> str:
        return self.key.stable_hash()

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_hash": self.key_hash,
            "key": self.key.to_dict(),
            "latency_p50_ms": float(self.latency_p50_ms),
            "latency_p90_ms": float(self.latency_p90_ms),
            "latency_p95_ms": float(self.latency_p95_ms),
            "latency_p99_ms": float(self.latency_p99_ms),
            "latency_mean_ms": float(self.latency_mean_ms),
            "latency_std_ms": float(self.latency_std_ms),
            "num_warmup": int(self.num_warmup),
            "num_repeat": int(self.num_repeat),
            "timing_method": self.timing_method,
            "engine_hash": self.engine_hash,
            "onnx_hash": self.onnx_hash,
            "created_at": self.created_at,
            "status": self.status,
            "error": self.error_message or self.error,
            "error_message": self.error_message or self.error,
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LatencyRecord":
        key_data = data.get("key")
        if not isinstance(key_data, dict):
            key_data = {k: data[k] for k in LatencyLUTKey.__dataclass_fields__ if k in data}
        return cls(
            key=LatencyLUTKey.from_dict(key_data),
            latency_p50_ms=float(data.get("latency_p50_ms", 0.0)),
            latency_p90_ms=float(data.get("latency_p90_ms", data.get("latency_p50_ms", 0.0))),
            latency_p95_ms=float(data.get("latency_p95_ms", data.get("latency_p50_ms", 0.0))),
            latency_p99_ms=float(data.get("latency_p99_ms", data.get("latency_p50_ms", 0.0))),
            latency_mean_ms=float(data.get("latency_mean_ms", data.get("latency_p50_ms", 0.0))),
            latency_std_ms=float(data.get("latency_std_ms", 0.0)),
            num_warmup=int(data.get("num_warmup", 0)),
            num_repeat=int(data.get("num_repeat", 0)),
            timing_method=str(data.get("timing_method", "")),
            engine_hash=data.get("engine_hash"),
            onnx_hash=data.get("onnx_hash"),
            created_at=str(data.get("created_at") or now_timestamp()),
            status=str(data.get("status", "success")),
            error=data.get("error"),
            error_message=data.get("error_message") or data.get("error"),
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def from_latencies(
        cls,
        key: LatencyLUTKey,
        values_ms: list[float],
        *,
        num_warmup: int,
        num_repeat: int,
        timing_method: str,
        engine_hash: str | None = None,
        onnx_hash: str | None = None,
        status: str = "success",
        **metadata: Any,
    ) -> "LatencyRecord":
        values = [float(v) for v in values_ms]
        if not values:
            values = [0.0]
        ordered = sorted(values)

        def percentile(pct: float) -> float:
            idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
            return float(ordered[idx])

        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) if values else 0.0
        return cls(
            key=key,
            latency_p50_ms=percentile(50),
            latency_p90_ms=percentile(90),
            latency_p95_ms=percentile(95),
            latency_p99_ms=percentile(99),
            latency_mean_ms=mean,
            latency_std_ms=std,
            num_warmup=num_warmup,
            num_repeat=num_repeat,
            timing_method=timing_method,
            engine_hash=engine_hash,
            onnx_hash=onnx_hash,
            status=status,
            metadata=metadata,
        )


def write_jsonl(records: list[LatencyRecord] | list[LatencyLUTKey], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def read_key_jsonl(path: str | Path) -> list[LatencyLUTKey]:
    keys: list[LatencyLUTKey] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            keys.append(LatencyLUTKey.from_dict(data.get("key", data)))
    return keys


def read_record_jsonl(path: str | Path) -> list[LatencyRecord]:
    records: list[LatencyRecord] = []
    if not Path(path).is_file():
        return records
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(LatencyRecord.from_dict(json.loads(line)))
    return records


def write_records_csv(records: list[LatencyRecord], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        row = record.to_dict()
        key = row.pop("key")
        for key_name, value in key.items():
            if key_name != "metadata":
                row[f"key.{key_name}"] = value
        rows.append(row)
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({field for row in rows for field in row.keys()})
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
