"""Fail-closed registry for supported HEAL LiDAR model families."""

from __future__ import annotations

from pathlib import Path

from .lidar_family import HEALLidarFamilySpec


_MODEL_ROOT = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/"
    "dairv2s/LiDAROnly"
)

_FAMILIES = {
    "lidar_pyramid": HEALLidarFamilySpec(
        name="lidar_pyramid",
        aliases=("pyramid", "lidar-pyramid"),
        fusion_kind="pyramid",
        export_recipe="pyramid_fixed_k",
        weighted_fusion=True,
        default_config=_MODEL_ROOT / "lidar_pyramid" / "config.yaml",
        default_checkpoint=(
            _MODEL_ROOT
            / "lidar_pyramid"
            / "net_epoch_bestval_at17.pth"
        ),
    ),
    "lidar_disco": HEALLidarFamilySpec(
        name="lidar_disco",
        aliases=("disco", "disconet", "lidar-disconet"),
        fusion_kind="disconet",
        export_recipe="soft_fusion_fixed_k",
        weighted_fusion=True,
        default_config=_MODEL_ROOT / "lidar_disco" / "config.yaml",
        default_checkpoint=(
            _MODEL_ROOT
            / "lidar_disco"
            / "net_epoch_bestval_at35.pth"
        ),
        compatibility_module="opencood.models.fuse_modules.disco_fuse",
    ),
    "lidar_fcooper": HEALLidarFamilySpec(
        name="lidar_fcooper",
        aliases=("fcooper", "f-cooper", "lidar-fcooper"),
        fusion_kind="max",
        export_recipe="soft_fusion_fixed_k",
        weighted_fusion=False,
        default_config=_MODEL_ROOT / "lidar_fcooper" / "config.yaml",
        default_checkpoint=None,
    ),
}

_ALIASES = {
    alias.lower(): name
    for name, spec in _FAMILIES.items()
    for alias in (name, *spec.aliases)
}


def registered_lidar_families() -> tuple[str, ...]:
    return tuple(sorted(_FAMILIES))


def get_lidar_family_spec(name: str) -> HEALLidarFamilySpec:
    normalized = str(name).strip().lower()
    canonical = _ALIASES.get(normalized)
    if canonical is None:
        raise KeyError(
            f"unknown_heal_lidar_family:{name}:"
            f"registered={','.join(registered_lidar_families())}"
        )
    return _FAMILIES[canonical]


__all__ = ["get_lidar_family_spec", "registered_lidar_families"]

