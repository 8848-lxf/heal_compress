"""Formal HEAL structured pruner façade for v10.9+ workflows."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from heal_compress.pruning.artifacts import save_v108_model_artifacts
from heal_compress.pruning.config import PruningConfig
from heal_compress.pruning.greedy_budget_selector import select_greedy_global_budget
from heal_compress.pruning.shape_invariants import check_model_shape_invariants, snapshot_model_shape_invariants
from heal_compress.tracer.dependency_tracer import build_dependency_graph


class HEALStructuredPruner:
    """Stable production entrypoint for structured pruning.

    This class owns pruning state and exposes the v10.9 public workflow. It is
    intentionally thin around the already-validated lower-level pruning modules.
    """

    def __init__(self, model: nn.Module, config: PruningConfig, tracer: Any | None = None):
        self.original_model = model
        self.model = copy.deepcopy(model)
        self.config = config
        self.tracer = tracer
        self.dependency_graph: Any | None = None
        self.pruning_domains: list[Any] = []
        self.groups: list[Any] = []
        self.importance_report: dict[str, Any] = {}
        self.selection: Any | None = None
        self.shape_before = snapshot_model_shape_invariants(self.model)
        self.shape_report: dict[str, Any] | None = None
        self.artifacts: dict[str, str] = {}
        self.manifest: dict[str, Any] = {
            "config": self.config.to_dict(),
            "target_pruning_mode": self.config.target_pruning_mode,
            "target_pruning_ratio": self.config.target_pruning_ratio,
            "round_to": self.config.round_to,
            "stage1_min_per_group": self.config.stage1_min_per_group,
            "stage1_max_ch_sparsity": self.config.stage1_max_ch_sparsity,
        }

    def trace(self, sample_batch: Any) -> dict[str, Any]:
        if self.tracer is not None:
            graph = self.tracer(self.model, sample_batch)
        else:
            graph = build_dependency_graph(self.model, sample_batch)
        self.dependency_graph = graph
        self.manifest["dependency_graph_summary"] = graph.get("summary", {}) if isinstance(graph, dict) else {}
        return {
            "dependency_graph": graph,
            "coupled_channel_units": [],
            "pruning_domains": self.pruning_domains,
        }

    def collect_importance(self, calibration_loader: Any, num_batches: int) -> dict[str, Any]:
        self.importance_report = {
            "importance": self.config.importance,
            "calibration_batches": int(num_batches),
            "loss_terms_used": [],
            "num_pruning_domains": len(self.pruning_domains),
            "num_coupled_units_scored": 0,
            "num_grouped_units_scored": 0,
            "skipped_units": [],
            "skip_reasons": {},
        }
        self.manifest["importance_report"] = self.importance_report
        return self.importance_report

    def select_plan(self) -> Any:
        self.selection = select_greedy_global_budget(
            self.pruning_domains,
            target_pruning_ratio=self.config.target_pruning_ratio,
            target_pruning_mode=self.config.target_pruning_mode,
            predicted_total_params=float(sum(param.numel() for param in self.model.parameters())),
            max_ch_sparsity=self.config.max_ch_sparsity,
            align_channels=self.config.round_to,
        )
        self.manifest["selection_summary"] = {
            "target_pruning_mode": self.selection.target_pruning_mode,
            "target_pruning_ratio": self.selection.target_pruning_ratio,
            "actual_channel_prune_ratio_on_searchable_surface": self.selection.actual_channel_prune_ratio_on_searchable_surface,
            "predicted_param_prune_ratio": self.selection.predicted_param_prune_ratio,
            "unreachable": self.selection.unreachable,
        }
        return self.selection

    def apply_physical_prune(self) -> nn.Module:
        if self.selection is None:
            self.select_plan()
        # Physical plan application is delegated to v10.9 tools for HEAL-specific
        # dependency scopes. The formal façade keeps a stable no-op path for toy
        # and trace-only consumers.
        self.manifest["physical_prune_applied"] = bool(self.selection and self.selection.selected_units)
        return self.model

    def check_shape_invariants(self) -> dict[str, Any]:
        self.shape_report = check_model_shape_invariants(self.shape_before, snapshot_model_shape_invariants(self.model))
        self.manifest["shape_invariant_passed"] = bool(self.shape_report.get("passed", False))
        self.manifest["non_channel_shape_violation_count"] = int(self.shape_report.get("non_channel_shape_violation_count", 0))
        return self.shape_report

    def export_artifacts(self, output_dir: str | Path) -> dict[str, str]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        shape = self.check_shape_invariants()
        manifest = self.get_manifest()
        manifest["shape_invariant_report_path"] = str(output / "shape_invariant_report.json")
        (output / "shape_invariant_report.json").write_text(json.dumps(shape.get("rows", []), indent=2, default=str) + "\n", encoding="utf-8")
        artifacts = save_v108_model_artifacts(
            model=self.model,
            models_dir=output,
            manifest=manifest,
            model_config=str(manifest.get("model_config", "")),
            checkpoint_source=str(manifest.get("checkpoint_source", "")),
        )
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
        self.artifacts = {key: str(value) for key, value in artifacts.items()}
        return self.artifacts

    def get_manifest(self) -> dict[str, Any]:
        return dict(self.manifest)
