"""Canonical model-family registry for the public search entrypoint."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    display_name: str
    aliases: tuple[str, ...]
    runner_kind: str
    config_template: str
    supports_activation_taylor: bool = True
    supports_beam_recovery: bool = True
    supports_greedy_anchor_gate: bool = True


_FAMILIES = (
    FamilySpec(
        family_id="lidar_pyramid",
        display_name="LiDAR Pyramid",
        aliases=("pyramid", "lidar-pyramid"),
        runner_kind="cnn_strict_stage12_v3",
        config_template="search/configs/unified/lidar_pyramid_ga.yaml",
    ),
    FamilySpec(
        family_id="heal_lidar_disco",
        display_name="DiscoNet",
        aliases=("disco", "disconet", "lidar_disco"),
        runner_kind="cnn_strict_stage12_v3",
        config_template="search/configs/unified/lidar_disco_ga.yaml",
    ),
    FamilySpec(
        family_id="heal_lidar_fcooper",
        display_name="F-Cooper",
        aliases=("f-cooper", "fcooper", "lidar_fcooper"),
        runner_kind="cnn_strict_stage12_v3",
        config_template="search/configs/unified/lidar_fcooper_ga.yaml",
    ),
    FamilySpec(
        family_id="heal_lidar_v2xvit",
        display_name="V2X-ViT",
        aliases=("v2x-vit", "v2xvit", "lidar_v2xvit"),
        runner_kind="v2xvit_framework",
        config_template="search/configs/unified/lidar_v2xvit_ga.yaml",
    ),
)

_BY_NAME = {
    name.lower(): spec
    for spec in _FAMILIES
    for name in (spec.family_id, *spec.aliases)
}


def get_family(value: str) -> FamilySpec:
    try:
        return _BY_NAME[str(value).strip().lower()]
    except KeyError as exc:
        supported = ", ".join(spec.family_id for spec in _FAMILIES)
        raise KeyError(f"unsupported_model_family:{value}; supported={supported}") from exc


def registered_families() -> tuple[FamilySpec, ...]:
    return _FAMILIES


__all__ = ["FamilySpec", "get_family", "registered_families"]
