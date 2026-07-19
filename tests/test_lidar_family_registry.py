from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_registry_exposes_pyramid_disco_and_fcooper() -> None:
    from search.integration.lidar_family_registry import (
        get_lidar_family_spec,
        registered_lidar_families,
    )

    assert registered_lidar_families() == (
        "lidar_disco",
        "lidar_fcooper",
        "lidar_pyramid",
    )
    pyramid = get_lidar_family_spec("lidar_pyramid")
    disco = get_lidar_family_spec("lidar_disco")
    fcooper = get_lidar_family_spec("lidar_fcooper")
    assert pyramid.export_recipe == "pyramid_fixed_k"
    assert disco.fusion_kind == "disconet"
    assert disco.export_recipe == "soft_fusion_fixed_k"
    assert disco.weighted_fusion is True
    assert disco.compatibility_module == (
        "opencood.models.fuse_modules.disco_fuse"
    )
    assert fcooper.fusion_kind == "max"
    assert fcooper.weighted_fusion is False
    assert fcooper.export_recipe == "soft_fusion_fixed_k"


def test_family_aliases_are_canonicalized() -> None:
    from search.integration.lidar_family_registry import get_lidar_family_spec

    assert get_lidar_family_spec("disconet").name == "lidar_disco"
    assert get_lidar_family_spec("f-cooper").name == "lidar_fcooper"
    assert get_lidar_family_spec("pyramid").name == "lidar_pyramid"


def test_unknown_family_fails_closed() -> None:
    from search.integration.lidar_family_registry import get_lidar_family_spec

    with pytest.raises(KeyError, match="unknown_heal_lidar_family"):
        get_lidar_family_spec("unknown")


def test_all_families_keep_fixed_k_scatter_outside_precision_gene() -> None:
    from search.integration.lidar_family_registry import (
        get_lidar_family_spec,
        registered_lidar_families,
    )

    for name in registered_lidar_families():
        spec = get_lidar_family_spec(name)
        assert spec.fixed_k == 29696
        assert spec.plugin_boundary_dtype == "fp32"
        assert spec.scatter_is_precision_gene is False
        assert spec.output_names == ("cls_preds", "reg_preds", "dir_preds")

