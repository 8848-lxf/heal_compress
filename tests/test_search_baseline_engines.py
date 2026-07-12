from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_strict_fp32_build_config_disables_tf32_fp16_and_int8() -> None:
    from search.baselines.original_engines import make_baseline_trt_build_config

    config = make_baseline_trt_build_config(
        "strict_fp32",
        trtexec_path=Path("/opt/trtexec"),
        plugin_path=Path("/tmp/plugin.so"),
        shape_profiles={"x": {"min": (1,), "opt": (1,), "max": (1,)}},
    )

    assert config.no_tf32 is True
    assert config.enable_fp16 is False
    assert config.enable_int8 is False
    assert config.precision_constraints == "obey"


def test_strict_fp16_build_config_enables_fp16_without_int8() -> None:
    from search.baselines.original_engines import make_baseline_trt_build_config

    config = make_baseline_trt_build_config(
        "strict_fp16",
        trtexec_path=None,
        plugin_path=None,
        shape_profiles={},
    )

    assert config.no_tf32 is True
    assert config.enable_fp16 is True
    assert config.enable_int8 is False


def test_maximal_legal_int8_build_config_uses_explicit_qdq_and_int8_builder() -> None:
    from search.baselines.original_engines import make_baseline_trt_build_config

    config = make_baseline_trt_build_config(
        "maximal_legal_int8",
        trtexec_path=None,
        plugin_path=None,
        shape_profiles={},
    )

    assert config.enable_fp16 is True
    assert config.enable_int8 is True
    assert "explicit-qdq" in config.policy_version


def test_baseline_precision_assignments_use_coupled_groups_not_synthetic() -> None:
    from quantization.types import CanonicalMappingEntry, OnnxOriginMapResult
    from search.baselines.original_engines import build_baseline_precision_profile
    from search.quantization_space.types import QuantizationSearchGroup

    groups = [
        QuantizationSearchGroup(
            "pg_scope_a",
            ("block.conv1", "block.conv2"),
            ("node1", "node2"),
            ("FP32", "FP16", "INT8"),
            False,
            "",
            0,
            12,
            100.0,
            {},
        )
    ]
    origin = OnnxOriginMapResult(
        entries=[
            CanonicalMappingEntry("block.conv1", "Conv2d", 0, "Conv", "Conv_0", "__canonical__conv1", "w1"),
            CanonicalMappingEntry("block.conv2", "Conv2d", 1, "Conv", "Conv_1", "__canonical__conv2", "w2"),
        ]
    )

    profile = build_baseline_precision_profile(
        "maximal_legal_int8",
        origin_map=origin,
        groups=groups,
    )

    assert len(profile.assignments) == 2
    assert {row.precision_group for row in profile.assignments} == {"pg_scope_a"}
    assert all(not row.precision_group.startswith("search::") for row in profile.assignments)
    assert all(not row.precision_group.startswith("pg::") for row in profile.assignments)
    assert all(row.requested_precision == "int8" for row in profile.assignments)
