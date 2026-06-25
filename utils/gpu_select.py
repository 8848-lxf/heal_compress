"""GPU selection helpers for experiment runners."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class GPUInfo:
    index: int
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_gpu: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class GPUSelectionError(RuntimeError):
    def __init__(
        self,
        *,
        error: str,
        candidate_gpus: Sequence[GPUInfo],
        excluded_gpus: Sequence[int],
        required_min_free_memory_mb: int,
        required_max_gpu_util: int,
        message: str | None = None,
    ):
        self.error = error
        self.candidate_gpus = list(candidate_gpus)
        self.excluded_gpus = list(excluded_gpus)
        self.required_min_free_memory_mb = int(required_min_free_memory_mb)
        self.required_max_gpu_util = int(required_max_gpu_util)
        super().__init__(message or json.dumps(self.to_dict(), ensure_ascii=False))

    def to_dict(self) -> dict:
        return {
            "error": self.error,
            "candidate_gpus": [gpu.to_dict() for gpu in self.candidate_gpus],
            "excluded_gpus": list(self.excluded_gpus),
            "required_min_free_memory_mb": self.required_min_free_memory_mb,
            "required_max_gpu_util": self.required_max_gpu_util,
        }


def parse_nvidia_smi_csv(text: str) -> list[GPUInfo]:
    gpus: list[GPUInfo] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        index, total, used, free, util = [int(float(part)) for part in parts]
        gpus.append(
            GPUInfo(
                index=index,
                memory_total_mb=total,
                memory_used_mb=used,
                memory_free_mb=free,
                utilization_gpu=util,
            )
        )
    return gpus


def query_nvidia_smi() -> list[GPUInfo]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise GPUSelectionError(
            error="nvidia_smi_not_available",
            candidate_gpus=[],
            excluded_gpus=[],
            required_min_free_memory_mb=0,
            required_max_gpu_util=0,
            message="nvidia-smi is not available; pass --gpu-id <id> or --gpu-id cpu explicitly",
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise GPUSelectionError(
            error="nvidia_smi_failed",
            candidate_gpus=[],
            excluded_gpus=[],
            required_min_free_memory_mb=0,
            required_max_gpu_util=0,
            message=exc.stderr or str(exc),
        ) from exc
    return parse_nvidia_smi_csv(result.stdout)


def select_gpu(
    gpu_id: str,
    gpus: Sequence[GPUInfo],
    *,
    exclude_gpu_ids: Sequence[int],
    min_free_memory_mb: int,
    max_gpu_util: int,
) -> GPUInfo:
    """Select one GPU.

    Explicit numeric GPU ids bypass exclusion and threshold checks. Auto mode
    applies exclusions and thresholds, then sorts by free memory desc,
    utilization asc, and used memory asc.
    """

    gpu_arg = str(gpu_id).strip().lower()
    if gpu_arg not in ("auto", ""):
        if gpu_arg.startswith("cuda:"):
            gpu_arg = gpu_arg.split(":", 1)[1]
        if not gpu_arg.isdigit():
            raise ValueError(f"gpu_id must be auto, cpu, cuda:N, or an integer id; got {gpu_id!r}")
        target = int(gpu_arg)
        for gpu in gpus:
            if int(gpu.index) == target:
                return gpu
        return GPUInfo(index=target, memory_total_mb=0, memory_used_mb=0, memory_free_mb=0, utilization_gpu=0)

    excluded = {int(v) for v in exclude_gpu_ids}
    candidates = [gpu for gpu in gpus if int(gpu.index) not in excluded]
    available = [
        gpu for gpu in candidates
        if int(gpu.memory_free_mb) >= int(min_free_memory_mb)
        and int(gpu.utilization_gpu) <= int(max_gpu_util)
    ]
    if not available:
        raise GPUSelectionError(
            error="no_available_gpu",
            candidate_gpus=candidates,
            excluded_gpus=sorted(excluded),
            required_min_free_memory_mb=min_free_memory_mb,
            required_max_gpu_util=max_gpu_util,
        )
    return sorted(
        available,
        key=lambda gpu: (-int(gpu.memory_free_mb), int(gpu.utilization_gpu), int(gpu.memory_used_mb), int(gpu.index)),
    )[0]


def resolve_gpu(
    gpu_id: str,
    *,
    exclude_gpu_ids: Sequence[int],
    min_free_memory_mb: int,
    max_gpu_util: int,
    queried_gpus: Sequence[GPUInfo] | None = None,
) -> tuple[str, GPUInfo | None, list[GPUInfo]]:
    gpu_arg = str(gpu_id).strip().lower()
    if gpu_arg == "cpu":
        return "cpu", None, []
    if gpu_arg not in ("auto", ""):
        normalized = gpu_arg.split(":", 1)[1] if gpu_arg.startswith("cuda:") else gpu_arg
        if not normalized.isdigit():
            raise ValueError(f"gpu_id must be auto, cpu, cuda:N, or an integer id; got {gpu_id!r}")
        index = int(normalized)
        return f"cuda:{index}", GPUInfo(
            index=index,
            memory_total_mb=0,
            memory_used_mb=0,
            memory_free_mb=0,
            utilization_gpu=0,
        ), []
    gpus = list(queried_gpus) if queried_gpus is not None else query_nvidia_smi()
    selected = select_gpu(
        gpu_id,
        gpus,
        exclude_gpu_ids=exclude_gpu_ids,
        min_free_memory_mb=min_free_memory_mb,
        max_gpu_util=max_gpu_util,
    )
    return f"cuda:{selected.index}", selected, gpus
