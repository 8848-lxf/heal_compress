from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_trt_build_worker_runtime_provenance_uses_loaded_modelopt_runtime() -> None:
    from search.stage2.trt_build_worker import _runtime_provenance

    class FakeCuda:
        @staticmethod
        def get_device_capability(_index: int) -> tuple[int, int]:
            return (8, 9)

        @staticmethod
        def get_device_name(_index: int) -> str:
            return "NVIDIA GeForce RTX 4090"

    fake_torch = SimpleNamespace(
        __version__="2.3.1",
        version=SimpleNamespace(cuda="11.8"),
        cuda=FakeCuda(),
    )
    fake_trt = SimpleNamespace(__version__="10.9.0.34")

    assert _runtime_provenance(fake_trt, fake_torch) == {
        "tensorrt_version": "10.9.0.34",
        "cuda_version": "11.8",
        "torch_version": "2.3.1",
        "gpu_architecture": "8.9",
        "gpu_name": "NVIDIA GeForce RTX 4090",
    }


def test_candidate_deployment_signature_joins_physical_qdq_and_runtime_lineage() -> None:
    from search.stage2.lidar_pyramid_real_evaluator import _candidate_deployment_signature

    context = SimpleNamespace(
        code_commit="commit-a",
        search_space=SimpleNamespace(plugin_hashes={"scatter.so": "plugin-a"}),
    )
    signature = _candidate_deployment_signature(
        context=context,
        physical={"physical_hash": "physical-a"},
        qdq={
            "base_onnx_hash": "onnx-a",
            "qdq_topology_hash": "topology-a",
            "canonical_mapping_hash": "mapping-a",
            "legalized_precision_profile_hash": "legalized-a",
            "realized_precision_profile_hash": "realized-a",
            "calibration_manifest_hash": "manifest-a",
            "calibration_recipe_hash": "recipe-a",
        },
        trt={
            "runtime_provenance": {
                "tensorrt_version": "10.9.0.34",
                "cuda_version": "11.8",
                "gpu_architecture": "8.9",
            }
        },
    )

    assert signature["code_commit"] == "commit-a"
    assert signature["physical_model_hash"] == "physical-a"
    assert signature["plugin_binary_hash"] == "plugin-a"
    assert signature["gpu_architecture"] == "8.9"


def test_search_context_repository_commit_matches_git_head() -> None:
    from search.integration.lidar_pyramid_context import _repository_commit

    repo = Path(__file__).resolve().parents[1]
    expected = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()

    assert _repository_commit(repo) == expected


def test_qdq_deployment_lineage_hashes_explicit_production_inputs(tmp_path: Path) -> None:
    from search.hashing import canonical_json_hash
    from search.stage2.lidar_pyramid_real_evaluator import _file_hash, _qdq_deployment_lineage

    base_onnx = tmp_path / "base.onnx"
    base_onnx.write_bytes(b"physical-base-onnx")
    mapping = {"entries": [{"module_path": "conv", "realized_request_precision": "int8"}]}
    legalized = {"quant::conv": "INT8"}
    realized = {"conv": "INT8"}
    calibration_recipe = {"backend": "tensorrt_entropy_calibration2", "batches": 200}
    qdq_result = SimpleNamespace(
        calibration_metadata={
            "qdq_topology_hash": "topology-a",
            "calibration_manifest_hash": "manifest-a",
        }
    )

    lineage = _qdq_deployment_lineage(
        base_onnx_path=base_onnx,
        canonical_mapping=mapping,
        legalized_precision_profile=legalized,
        realized_precision_profile=realized,
        calibration_recipe=calibration_recipe,
        qdq_result=qdq_result,
    )

    assert lineage == {
        "base_onnx_hash": _file_hash(base_onnx),
        "qdq_topology_hash": "topology-a",
        "canonical_mapping_hash": canonical_json_hash(mapping),
        "legalized_precision_profile_hash": canonical_json_hash(legalized),
        "realized_precision_profile_hash": canonical_json_hash(realized),
        "calibration_manifest_hash": "manifest-a",
        "calibration_recipe_hash": canonical_json_hash(calibration_recipe),
    }


def test_modelopt_python_command_uses_resolved_4090_conda_root() -> None:
    from search.integration.runtime_environment import (
        modelopt_python_command,
        resolve_conda_env_prefix,
    )

    prefix = resolve_conda_env_prefix("modelopt")
    conda_sh = prefix.parents[1] / "etc" / "profile.d" / "conda.sh"
    command = modelopt_python_command("modelopt")

    assert conda_sh.is_file()
    assert f"source {conda_sh}" in command[2]
    assert "/home/lixingfeng/miniconda3/etc/profile.d/conda.sh" not in command[2]


def test_trt_build_worker_deserializes_written_engine_after_loading_plugin(tmp_path: Path) -> None:
    from search.stage2.trt_build_worker import _engine_deserialization_audit

    events: list[str] = []
    engine_path = tmp_path / "engine.plan"
    engine_path.write_bytes(b"serialized-engine")
    plugin_path = tmp_path / "scatter.so"
    plugin_path.write_bytes(b"plugin")

    class FakeEngine:
        num_io_tensors = 4

    class FakeRuntime:
        def __init__(self, _logger: object) -> None:
            events.append("runtime")

        def deserialize_cuda_engine(self, payload: bytes) -> FakeEngine:
            events.append(f"deserialize:{payload.decode()}")
            return FakeEngine()

    fake_trt = SimpleNamespace(
        Logger=lambda _severity: object(),
        Runtime=FakeRuntime,
    )
    fake_trt.Logger.ERROR = object()

    report = _engine_deserialization_audit(
        fake_trt,
        engine_path,
        plugin_path=plugin_path,
        plugin_loader=lambda path: events.append(f"plugin:{Path(path).name}"),
    )

    assert report == {
        "passed": True,
        "engine_path": str(engine_path),
        "engine_bytes": len(b"serialized-engine"),
        "num_io_tensors": 4,
        "plugin_path": str(plugin_path),
        "plugin_loaded_before_deserialize": True,
    }
    assert events == ["plugin:scatter.so", "runtime", "deserialize:serialized-engine"]


def test_trt_build_worker_fails_closed_when_engine_deserialization_returns_none(tmp_path: Path) -> None:
    from search.stage2.trt_build_worker import _engine_deserialization_audit

    engine_path = tmp_path / "engine.plan"
    engine_path.write_bytes(b"invalid-engine")

    class FakeLogger:
        ERROR = object()

        def __init__(self, _severity: object) -> None:
            pass

    fake_trt = SimpleNamespace(
        Logger=FakeLogger,
        Runtime=lambda _logger: SimpleNamespace(deserialize_cuda_engine=lambda _payload: None),
    )

    with pytest.raises(RuntimeError, match="engine_deserialize_returned_none"):
        _engine_deserialization_audit(fake_trt, engine_path)


def test_gpu_isolation_audit_rejects_foreign_compute_process() -> None:
    from search.integration.runtime_environment import audit_gpu_isolation

    report = audit_gpu_isolation(
        [
            {
                "index": 3,
                "uuid": "GPU-4090",
                "utilization_gpu_pct": 0,
                "processes": [
                    {"pid": 101, "process_name": "search", "used_memory_mib": 1200},
                    {"pid": 202, "process_name": "training", "used_memory_mib": 7000},
                ],
            }
        ],
        gpu_index=3,
        allowed_pids={101},
    )

    assert report["passed"] is False
    assert report["gpu_uuid"] == "GPU-4090"
    assert [row["pid"] for row in report["foreign_compute_processes"]] == [202]
    assert report["issues"] == ["foreign_compute_processes_present"]


def test_gpu_isolation_audit_accepts_only_current_process() -> None:
    from search.integration.runtime_environment import audit_gpu_isolation

    report = audit_gpu_isolation(
        [
            {
                "index": 1,
                "uuid": "GPU-OWN",
                "utilization_gpu_pct": 8,
                "processes": [
                    {"pid": 303, "process_name": "search", "used_memory_mib": 1200}
                ],
            }
        ],
        gpu_index=1,
        allowed_pids={303},
    )

    assert report["passed"] is True
    assert report["foreign_compute_processes"] == []
