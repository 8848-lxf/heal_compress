from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIAD = ROOT.parent
for path in (UNIAD, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _toy_layer_specs() -> list[dict[str, object]]:
    return [
        {"module_name": "dense_12", "module_type": "Conv2d", "in_channels": 12, "out_channels": 20, "groups": 1},
        {"module_name": "dense_28", "module_type": "Conv2d", "in_channels": 28, "out_channels": 28, "groups": 1},
        {"module_name": "grouped_safe", "module_type": "Conv2d", "in_channels": 32 * 32, "out_channels": 32 * 32, "groups": 32},
        {"module_name": "grouped_per16", "module_type": "Conv2d", "in_channels": 32 * 16, "out_channels": 32 * 16, "groups": 32},
        {"module_name": "grouped_per8", "module_type": "Conv2d", "in_channels": 32 * 8, "out_channels": 32 * 8, "groups": 32},
        {"module_name": "grouped_per4", "module_type": "Conv2d", "in_channels": 32 * 4, "out_channels": 32 * 4, "groups": 32},
    ]


def test_random_sampler_is_reproducible_and_does_not_use_taylor_ranking(tmp_path: Path) -> None:
    from tools.latency_lut.random_deployment_aware_subnet_sampler import (
        RANDOM_SAMPLING_METHOD,
        generate_random_deployment_aware_subnets,
    )

    first = generate_random_deployment_aware_subnets(
        layer_specs=_toy_layer_specs(),
        output_dir=tmp_path / "run_a",
        num_subnets=4,
        seed=123,
        max_channel_prune_ratio=0.80,
        min_channel_keep_ratio=0.20,
        ordinary_conv_round_to=4,
        grouped_conv_safe_per_group={4, 8, 16, 32},
    )
    second = generate_random_deployment_aware_subnets(
        layer_specs=_toy_layer_specs(),
        output_dir=tmp_path / "run_b",
        num_subnets=4,
        seed=123,
        max_channel_prune_ratio=0.80,
        min_channel_keep_ratio=0.20,
        ordinary_conv_round_to=4,
        grouped_conv_safe_per_group={4, 8, 16, 32},
    )

    assert RANDOM_SAMPLING_METHOD == "deployment_aware_random_without_taylor_ranking"
    assert [row["structure_hash"] for row in first["subnets"]] == [row["structure_hash"] for row in second["subnets"]]
    assert all(row["random_sampling_method"] == RANDOM_SAMPLING_METHOD for row in first["subnets"])
    assert "taylor_importance" not in json.dumps(first["subnets"]).lower()


def test_random_sampler_different_seed_changes_structure_hashes(tmp_path: Path) -> None:
    from tools.latency_lut.random_deployment_aware_subnet_sampler import generate_random_deployment_aware_subnets

    first = generate_random_deployment_aware_subnets(
        layer_specs=_toy_layer_specs(),
        output_dir=tmp_path / "run_a",
        num_subnets=4,
        seed=123,
        max_channel_prune_ratio=0.80,
        min_channel_keep_ratio=0.20,
        ordinary_conv_round_to=4,
        grouped_conv_safe_per_group={4, 8, 16, 32},
    )
    second = generate_random_deployment_aware_subnets(
        layer_specs=_toy_layer_specs(),
        output_dir=tmp_path / "run_b",
        num_subnets=4,
        seed=124,
        max_channel_prune_ratio=0.80,
        min_channel_keep_ratio=0.20,
        ordinary_conv_round_to=4,
        grouped_conv_safe_per_group={4, 8, 16, 32},
    )

    assert {row["structure_hash"] for row in first["subnets"]} != {row["structure_hash"] for row in second["subnets"]}


def test_random_sampler_respects_deployment_channel_constraints(tmp_path: Path) -> None:
    from tools.latency_lut.random_deployment_aware_subnet_sampler import generate_random_deployment_aware_subnets

    result = generate_random_deployment_aware_subnets(
        layer_specs=_toy_layer_specs(),
        output_dir=tmp_path / "run",
        num_subnets=6,
        seed=777,
        max_channel_prune_ratio=0.80,
        min_channel_keep_ratio=0.20,
        ordinary_conv_round_to=4,
        grouped_conv_safe_per_group={4, 8, 16, 32},
    )

    assert len(result["subnets"]) == 6
    assert result["diversity_report"]["duplicate_reject_count"] == 0
    for subnet in result["subnets"]:
        manifest = json.loads(Path(subnet["manifest_path"]).read_text(encoding="utf-8"))
        assert manifest["disable_taylor_ranking"] is True
        assert manifest["max_single_layer_channel_prune_ratio"] <= 0.80 + 1e-12
        for row in manifest["module_channel_before_after"]:
            before = row["before"]
            after = row["after"]
            if int(before["groups"]) == 1:
                assert int(after["in_channels"]) % 4 == 0
                assert int(after["out_channels"]) % 4 == 0
            else:
                assert int(after["in_channels"]) // int(after["groups"]) in {4, 8, 16, 32}
                assert int(after["out_channels"]) // int(after["groups"]) in {4, 8, 16, 32}
                if int(before["in_channels"]) // int(before["groups"]) == 4:
                    assert int(after["in_channels"]) == int(before["in_channels"])
                    assert int(after["out_channels"]) == int(before["out_channels"])


def test_random_sampler_writes_required_dryrun_reports(tmp_path: Path) -> None:
    from tools.latency_lut.random_deployment_aware_subnet_sampler import generate_random_deployment_aware_subnets

    result = generate_random_deployment_aware_subnets(
        layer_specs=_toy_layer_specs(),
        output_dir=tmp_path / "run",
        num_subnets=2,
        seed=9,
        max_channel_prune_ratio=0.80,
        min_channel_keep_ratio=0.20,
        ordinary_conv_round_to=4,
        grouped_conv_safe_per_group={4, 8, 16, 32},
    )

    assert (tmp_path / "run/diversity_report.json").is_file()
    assert (tmp_path / "run/random_subnet_sampling_report.md").is_file()
    for row in result["subnets"]:
        subnet_dir = Path(row["subnet_dir"])
        assert (subnet_dir / "pruning_manifest.json").is_file()
        assert (subnet_dir / "grouped_conv_int8_eligibility_report.json").is_file()
        report = json.loads((subnet_dir / "grouped_conv_int8_eligibility_report.json").read_text(encoding="utf-8"))
        assert all(item["int8_shape_supported"] for item in report)


def test_random_sampler_covers_global_target_prune_bins_and_records_global_metrics(tmp_path: Path) -> None:
    from tools.latency_lut.random_deployment_aware_subnet_sampler import generate_random_deployment_aware_subnets

    result = generate_random_deployment_aware_subnets(
        layer_specs=_toy_layer_specs(),
        output_dir=tmp_path / "run",
        num_subnets=8,
        seed=20260709,
        max_channel_prune_ratio=0.80,
        min_channel_keep_ratio=0.20,
        ordinary_conv_round_to=4,
        grouped_conv_safe_per_group={4, 8, 16, 32},
        global_target_prune_bins=[(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8)],
    )

    bins = [row["target_global_prune_bin"] for row in result["subnets"]]
    assert bins.count("0.00:0.20") == 2
    assert bins.count("0.20:0.40") == 2
    assert bins.count("0.40:0.60") == 2
    assert bins.count("0.60:0.80") == 2
    assert len({round(float(row["sampled_target_global_prune_ratio"]), 2) for row in result["subnets"]}) > 4
    assert any(float(row["max_single_layer_channel_prune_ratio"]) < 0.5 for row in result["subnets"])
    for row in result["subnets"]:
        manifest = json.loads(Path(row["manifest_path"]).read_text(encoding="utf-8"))
        for key in (
            "achieved_global_channel_prune_ratio",
            "achieved_global_param_prune_ratio",
            "achieved_global_bops_prune_ratio",
            "mean_layer_prune_ratio",
            "median_layer_prune_ratio",
        ):
            assert key in manifest
            assert 0.0 <= float(manifest[key]) <= 1.0


def test_grouped_conv_snap_report_records_supported_transitions(tmp_path: Path) -> None:
    from tools.latency_lut.random_deployment_aware_subnet_sampler import generate_random_deployment_aware_subnets

    result = generate_random_deployment_aware_subnets(
        layer_specs=_toy_layer_specs(),
        output_dir=tmp_path / "run",
        num_subnets=12,
        seed=42,
        max_channel_prune_ratio=0.80,
        min_channel_keep_ratio=0.20,
        ordinary_conv_round_to=4,
        grouped_conv_safe_per_group={4, 8, 16, 32},
        global_target_prune_bins=[(0.6, 0.8)],
    )

    rows = []
    for subnet in result["subnets"]:
        rows.extend(json.loads((Path(subnet["subnet_dir"]) / "grouped_conv_int8_eligibility_report.json").read_text(encoding="utf-8")))
    assert all(row["after_cin_per_group"] in {4, 8, 16, 32} for row in rows if int(row["groups"]) > 1)
    assert all(row["after_cout_per_group"] in {4, 8, 16, 32} for row in rows if int(row["groups"]) > 1)
    assert any(row["whether_8_to_4"] for row in rows)
    assert any(row["whether_16_to_8_or_4"] for row in rows)
    for row in rows:
        assert "snapping_policy" in row
        assert "unsupported_reason" in row
        assert "before_cin_per_group" in row
        assert "after_cin_per_group" in row


def test_source_layer_spec_audit_uses_before_channels_not_old_after_channels(tmp_path: Path) -> None:
    from tools.latency_lut.random_deployment_aware_subnet_sampler import load_layer_specs_for_cli, write_source_layer_spec_audit

    source = tmp_path / "subnet_999"
    source.mkdir()
    (source / "pruning_manifest.json").write_text(
        json.dumps(
            {
                "module_channel_before_after": [
                    {
                        "module_name": "dense",
                        "module_type": "Conv2d",
                        "before": {"in_channels": 64, "out_channels": 128, "groups": 1},
                        "after": {"in_channels": 32, "out_channels": 64, "groups": 1},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    specs, source_path = load_layer_specs_for_cli(source, tmp_path)
    audit = write_source_layer_spec_audit(tmp_path / "out", source_path, specs)

    assert specs[0]["in_channels"] == 64
    assert specs[0]["out_channels"] == 128
    assert audit["uses_original_before_channels"] is True
    assert audit["uses_old_subnet_after_channels"] is False
    assert audit["verdict"] == "pass"


def test_random_sampler_protects_deblock_convtranspose_output_by_default(tmp_path: Path) -> None:
    from tools.latency_lut.random_deployment_aware_subnet_sampler import generate_random_deployment_aware_subnets

    layer_specs = [
        {
            "module_name": "pyramid_backbone.deblocks.0.0",
            "module_type": "ConvTranspose2d",
            "in_channels": 256,
            "out_channels": 128,
            "groups": 1,
        },
        {
            "module_name": "pyramid_backbone.shrink_conv.0",
            "module_type": "Conv2d",
            "in_channels": 384,
            "out_channels": 128,
            "groups": 1,
        },
    ]

    result = generate_random_deployment_aware_subnets(
        layer_specs=layer_specs,
        output_dir=tmp_path / "run",
        num_subnets=2,
        seed=20260709,
        max_channel_prune_ratio=0.80,
        min_channel_keep_ratio=0.20,
        ordinary_conv_round_to=4,
        grouped_conv_safe_per_group={4, 8, 16, 32},
        global_target_prune_bins=[(0.6, 0.8)],
    )

    for subnet in result["subnets"]:
        manifest = json.loads(Path(subnet["manifest_path"]).read_text(encoding="utf-8"))
        row = next(item for item in manifest["module_channel_before_after"] if item["module_name"] == "pyramid_backbone.deblocks.0.0")
        assert row["after"]["out_channels"] == 128
        assert row["after"]["in_channels"] <= row["before"]["in_channels"]
        assert row["protected_axes"] == ["out"]
        assert row["protection_reason"] == "protected_convtranspose_deblock_or_fpn_output_contract"
        assert manifest["allow_deblock_output_pruning"] is False
        assert manifest["protect_deblock_output"] is True
