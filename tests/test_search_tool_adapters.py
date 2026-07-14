from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search.adapters.pruning_adapter import FormalPruningAdapter
from search.adapters.quantization_adapter import FormalQuantizationAdapter
from search.adapters.tracer_adapter import FormalTracerAdapter
from search.candidate import CandidatePhenotype, PrecisionDecision
from search.stage2 import trt_modelopt


def test_tracer_adapter_calls_trace_model() -> None:
    calls: list[str] = []

    def trace_model(model: Any, example_inputs: Any, **kwargs: Any) -> str:
        calls.append("trace_model")
        return "trace-result"

    adapter = FormalTracerAdapter(trace_model_fn=trace_model)

    assert adapter.trace(object(), object()) == "trace-result"
    assert calls == ["trace_model"]


def test_pruning_adapter_calls_plan_first_pipeline() -> None:
    calls: list[str] = []

    @dataclass
    class Result:
        model: str = "pruned-model"
        plan: str = "legal-plan"
        ledger: str = "ledger"
        snapshot: str = "snapshot"

    adapter = FormalPruningAdapter(
        build_plan_fn=lambda model, request: calls.append("build_physical_pruning_plan") or "plan",
        legalize_plan_fn=lambda model, plan, **kwargs: calls.append("legalize_pruning_plan") or "legal-plan",
        materialize_fn=lambda model, plan, **kwargs: calls.append("materialize_pruning") or Result(),
        snapshot_fn=lambda model: calls.append("build_physical_structure_snapshot") or "snapshot",
        hash_fn=lambda snapshot: calls.append("compute_physical_hashes") or "hashes",
        validate_fn=lambda model, **kwargs: calls.append("validate_physical_model") or "validation",
    )

    result = adapter.materialize_from_request("model", "request")

    assert result["model"] == "pruned-model"
    assert calls == [
        "build_physical_pruning_plan",
        "legalize_pruning_plan",
        "materialize_pruning",
        "compute_physical_hashes",
        "validate_physical_model",
    ]


def test_pruning_adapter_aggregates_selected_atomic_units_by_module_axis() -> None:
    @dataclass
    class Member:
        module_path: str
        axis: str
        indices: list[int]
        dependency_type: str
        index_map: dict[int, list[int]]

        def to_dict(self) -> dict[str, Any]:
            return {
                "module_path": self.module_path,
                "axis": self.axis,
                "indices": self.indices,
                "dependency_type": self.dependency_type,
                "index_map": self.index_map,
            }

    @dataclass
    class Unit:
        stable_id: str
        scope_id: str
        root_module_path: str
        root_axis: str
        root_indices: list[int]
        members: list[Member]
        protected: bool = False
        constraints: dict[str, Any] | None = None
        source_coupled_unit_ids: list[str] | None = None
        channel_cost: int = 1
        parameter_cost: int = 0
        group_keep_map: dict[int, list[int]] | None = None
        group_prune_map: dict[int, list[int]] | None = None
        protection_reason: str = ""

    units = [
        Unit(
            stable_id="apu_a",
            scope_id="scope_a",
            root_module_path="conv",
            root_axis="out",
            root_indices=[0],
            members=[Member("conv", "out", [0], "root_out", {0: [0]})],
        ),
        Unit(
            stable_id="apu_b",
            scope_id="scope_a",
            root_module_path="conv",
            root_axis="out",
            root_indices=[1],
            members=[Member("conv", "out", [1], "root_out", {1: [1]})],
        ),
    ]
    phenotype = CandidatePhenotype(
        pruned_unit_ids=["apu_a", "apu_b"],
        precision_profile={},
    )

    request = FormalPruningAdapter().request_from_phenotype(phenotype, units)

    assert len(request.entries) == 1
    entry = request.entries[0]
    assert entry.module_path == "conv"
    assert entry.axis == "out"
    assert entry.prune_indices == [0, 1]
    assert entry.source_atomic_unit_ids == ["apu_a", "apu_b"]
    assert entry.metadata["closure_index_map"] == {0: [0], 1: [1]}


def test_quantization_adapter_calls_explicit_qdq_pipeline(tmp_path: Path) -> None:
    calls: list[str] = []
    phenotype = CandidatePhenotype(
        pruned_unit_ids=[],
        precision_profile={"m": PrecisionDecision("INT8", "INT8", "")},
    )

    @dataclass
    class Export:
        onnx_path: str = "origin.onnx"
        origin_map: str = "origin-map"

    @dataclass
    class Mapping:
        entries: list[Any]

    adapter = FormalQuantizationAdapter(
        export_onnx_fn=lambda *args, **kwargs: calls.append("export_pruned_signal_maxk_onnx") or Export(),
        profile_builder_fn=lambda origin_map, phenotype: calls.append("precision_profile_from_phenotype") or "profile",
        canonical_mapping_fn=lambda origin_map, profile, **kwargs: calls.append("build_canonical_precision_mapping") or Mapping(entries=[]),
        calibration_fn=lambda *args, **kwargs: calls.append("collect_or_validate_calibration_scales") or {"m": 1.0},
        qdq_fn=lambda *args, **kwargs: calls.append("insert_explicit_qdq") or "qdq-result",
    )

    result = adapter.export_qdq(
        model="model",
        example_inputs=(),
        physical_snapshot="snapshot",
        phenotype=phenotype,
        output_dir=tmp_path,
    )

    assert result["qdq"] == "qdq-result"
    assert calls == [
        "export_pruned_signal_maxk_onnx",
        "precision_profile_from_phenotype",
        "build_canonical_precision_mapping",
        "collect_or_validate_calibration_scales",
        "insert_explicit_qdq",
    ]


def test_quantization_adapter_does_not_use_native_int8() -> None:
    adapter = FormalQuantizationAdapter()
    assert "native" not in adapter.pipeline_name.lower()
    assert "explicit_qdq" in adapter.pipeline_name


def test_modelopt_trt_build_worker_uses_conda_run_python(tmp_path: Path, monkeypatch: Any) -> None:
    calls: dict[str, Any] = {}

    monkeypatch.setattr(
        trt_modelopt,
        "modelopt_python_command",
        lambda conda_env: ["bash", "-lc", "activate-modelopt", "modelopt-python", conda_env],
    )

    def fake_env(**kwargs: Any) -> dict[str, str]:
        calls["env_kwargs"] = kwargs
        return {"PYTHONPATH": "", "LD_LIBRARY_PATH": "", "CUDA_VISIBLE_DEVICES": str(kwargs.get("cuda_visible_devices", ""))}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls["cmd"] = cmd
        calls["run_kwargs"] = kwargs
        request_path = Path(cmd[-1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        Path(request["output_path"]).write_text(json.dumps({"status": "ok"}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="worker-ok")

    monkeypatch.setattr(trt_modelopt, "modelopt_subprocess_env", fake_env)
    monkeypatch.setattr(trt_modelopt.subprocess, "run", fake_run)

    result = trt_modelopt.build_engine_modelopt(
        qdq_onnx=tmp_path / "model.onnx",
        engine_path=tmp_path / "engine.plan",
        precision_mapping={"entries": []},
        build_config={},
        physical_snapshot={},
        output_dir=tmp_path / "build",
        tensorrt_root=tmp_path / "trt",
        conda_env="modelopt",
        gpu_id=2,
    )

    assert result["status"] == "ok"
    assert calls["cmd"][:6] == ["bash", "-lc", "activate-modelopt", "modelopt-python", "modelopt", "-m"]
    assert calls["env_kwargs"]["cuda_visible_devices"] == 2
    assert calls["run_kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == "2"


def test_modelopt_subprocess_env_pins_conda_cuda_and_compilers(tmp_path: Path, monkeypatch: Any) -> None:
    from search.integration import runtime_environment

    prefix = tmp_path / "modelopt"
    trt_root = tmp_path / "TensorRT-10.9"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "lib").mkdir()
    (trt_root / "bin").mkdir(parents=True)
    (trt_root / "lib").mkdir()
    monkeypatch.setattr(runtime_environment, "resolve_conda_env_prefix", lambda _name: prefix)
    monkeypatch.setenv("PATH", "/usr/local/cuda/bin:/usr/bin")
    monkeypatch.setenv("CMAKE_PREFIX_PATH", "/legacy")

    env = runtime_environment.modelopt_subprocess_env(
        tensorrt_root=trt_root,
        conda_env="modelopt",
        cuda_visible_devices=7,
    )

    assert env["PATH"].split(":", 1)[0] == str(prefix / "bin")
    assert env["CUDA_HOME"] == str(prefix)
    assert env["CC"] == str(prefix / "bin" / "gcc")
    assert env["CXX"] == str(prefix / "bin" / "g++")
    assert env["CUDACXX"] == str(prefix / "bin" / "nvcc")
    assert env["CMAKE_PREFIX_PATH"] == f"{prefix}:/legacy"
    assert env["CUDA_VISIBLE_DEVICES"] == "7"


def test_modelopt_python_command_removes_system_cuda_runtime_precedence() -> None:
    from search.integration.runtime_environment import modelopt_python_command

    command = modelopt_python_command("modelopt")
    script = command[2]

    assert "requested_ld_library_path" in script
    assert "requested_path" in script
    assert "/usr/local/cuda*" in script
    assert "*/envs/univ2x-opt/*" in script
    assert 'export PATH="$clean_path"' in script
    assert 'export LD_LIBRARY_PATH="$clean_ld_library_path"' in script


def test_search_cli_records_effective_process_arguments(tmp_path: Path, monkeypatch: Any) -> None:
    from search import cli

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"model": {"checkpoint": "/model.pth"}}),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run with spaces"
    run_dir.mkdir()

    class FakeSearch:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def run(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["baseline_only"] is True
            return {"run_dir": str(run_dir)}

    monkeypatch.setattr(cli, "LidarPyramidTwoStageSearch", FakeSearch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "search.cli",
            "--config",
            str(config_path),
            "--output-root",
            str(tmp_path / "output with spaces"),
            "--baseline-only",
        ],
    )

    assert cli.main() == 0
    recorded = (run_dir / "commands.sh").read_text(encoding="utf-8")
    assert recorded == (
        "python -m search.cli --config "
        + str(config_path)
        + " --output-root '"
        + str(tmp_path / "output with spaces")
        + "' --baseline-only\n"
    )


def test_search_cli_preserves_configured_gpu_when_flag_is_omitted(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from search import cli

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {"checkpoint": "/model.pth"},
                "runtime": {"gpu_id": "5"},
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    captured: dict[str, Any] = {}

    class FakeSearch:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def run(self, **_kwargs: Any) -> dict[str, Any]:
            return {"run_dir": str(run_dir)}

    monkeypatch.setattr(cli, "LidarPyramidTwoStageSearch", FakeSearch)

    assert (
        cli.main(
            [
                "--config",
                str(config_path),
                "--output-root",
                str(tmp_path / "output"),
                "--baseline-only",
            ]
        )
        == 0
    )
    assert captured["config"]["runtime"]["gpu_id"] == "5"


def test_search_cli_explicit_gpu_overrides_configured_gpu(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from search import cli

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {"checkpoint": "/model.pth"},
                "runtime": {"gpu_id": "5"},
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    captured: dict[str, Any] = {}

    class FakeSearch:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def run(self, **_kwargs: Any) -> dict[str, Any]:
            return {"run_dir": str(run_dir)}

    monkeypatch.setattr(cli, "LidarPyramidTwoStageSearch", FakeSearch)

    assert (
        cli.main(
            [
                "--config",
                str(config_path),
                "--output-root",
                str(tmp_path / "output"),
                "--gpu-id",
                "6",
                "--baseline-only",
            ]
        )
        == 0
    )
    assert captured["config"]["runtime"]["gpu_id"] == "6"
