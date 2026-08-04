from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.latency_lut.save_reload_stage0_variants_v1061 import (  # noqa: E402
    artifact_paths_for_variant,
    build_model_artifact_manifest,
    load_model_object_artifact,
    save_model_artifacts,
    smoke_reloaded_model,
)


class TinyReloadModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_artifact_path_generation(tmp_path: Path) -> None:
    paths = artifact_paths_for_variant(tmp_path / "out", "stage0_reblock8_no_prune")

    assert paths["model_object"].name == "stage0_reblock8_no_prune_model_object.pth"
    assert paths["state_dict_manifest"].name == "stage0_reblock8_no_prune_state_dict_with_manifest.pth"
    assert paths["model_object"].parent.name == "models"


def test_manifest_contains_required_fields() -> None:
    manifest = build_model_artifact_manifest(
        variant="stage0_reblock8_prune_pergroup8_all_blocks",
        modified_blocks=[{"block_name": "layer0.0", "before": {}, "after": {}}],
        parameter_count_before=100,
        parameter_count_after=75,
    )

    for key in [
        "variant",
        "modified_blocks",
        "groups_old",
        "groups_new",
        "hidden_width_old",
        "hidden_width_new",
        "per_group_old",
        "per_group_new",
        "hidden_keep_indices",
        "keep_subgroups_per_new_group",
        "parameter_count_before",
        "parameter_count_after",
    ]:
        assert key in manifest
    assert manifest["requires_architecture_patch"] is True


def test_saved_model_object_reload_smoke_helper_works_on_toy_model(tmp_path: Path) -> None:
    model = TinyReloadModel().eval()
    manifest = build_model_artifact_manifest(
        variant="toy",
        modified_blocks=[],
        parameter_count_before=sum(p.numel() for p in model.parameters()),
        parameter_count_after=sum(p.numel() for p in model.parameters()),
    )

    paths = save_model_artifacts(
        model=model,
        variant="toy",
        models_dir=tmp_path / "models",
        model_config="config.yaml",
        checkpoint_source="checkpoint.pth",
        manifest=manifest,
        architecture_changed=True,
    )
    reloaded = load_model_object_artifact(paths["model_object"], device=torch.device("cpu"))
    report = smoke_reloaded_model(reloaded, torch.randn(1, 3, 5, 7))

    assert paths["model_object"].is_file()
    assert paths["state_dict_manifest"].is_file()
    assert report["reload_forward_smoke_passed"] is True
    assert report["output_finite"] is True
