"""Opt-in TensorRT engine build execution."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from ..artifacts.io import atomic_write_bytes, file_sha256
from ..config import TensorRTBuildConfig
from ..exceptions import TensorRTBuildError
from ..types import CanonicalPrecisionMappingResult, TensorRTBuildResult
from .command import build_trt_command


def build_trt_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    precision_mapping: CanonicalPrecisionMappingResult,
    *,
    config: TensorRTBuildConfig,
    layer_info_path: str | Path | None = None,
    log_path: str | Path | None = None,
    raise_on_failure: bool = True,
) -> TensorRTBuildResult:
    """Execute a build only when this function is explicitly called."""

    command = build_trt_command(
        onnx_path,
        engine_path,
        precision_mapping,
        config=config,
        layer_info_path=layer_info_path,
    )
    engine = Path(engine_path)
    engine.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command.command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(config.timeout_seconds),
            check=False,
        )
        output = completed.stdout or ""
        returncode: int | None = int(completed.returncode)
        failure = "" if completed.returncode == 0 and engine.is_file() else f"trtexec_failed_rc_{completed.returncode}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        output = str(exc)
        returncode = None
        failure = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - started
    log = Path(log_path) if log_path is not None else engine.with_suffix(".build.log")
    atomic_write_bytes(log, output.encode("utf-8", errors="replace"))
    result = TensorRTBuildResult(
        success=not failure,
        command=command,
        returncode=returncode,
        elapsed_seconds=elapsed,
        engine_hash=file_sha256(engine),
        log_path=str(log),
        failure_reason=failure,
    )
    if failure and raise_on_failure:
        raise TensorRTBuildError(failure)
    return result
