"""Unified orchestration facade over the audited family-specific backends."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .artifacts import BestEnginePublisher
from .config import ResolvedSearchConfig
from .progress import SearchProgress


Backend = Callable[[ResolvedSearchConfig, Path], Mapping[str, Any]]


class UnifiedSearchRunner:
    def __init__(
        self,
        config: ResolvedSearchConfig,
        *,
        output_root: str | Path,
        progress: SearchProgress | None = None,
        backends: Mapping[str, Backend] | None = None,
    ) -> None:
        self.config = config
        self.output_root = Path(output_root).expanduser().resolve()
        self.progress = progress or SearchProgress(
            enabled=bool(config.payload.get("search", {}).get("show_progress", True))
        )
        self.backends = dict(backends or {})

    def plan(self) -> dict[str, Any]:
        proxy = dict(self.config.payload.get("proxy", {}) or {})
        stage2 = dict(self.config.payload.get("stage2", {}) or {})
        gate = dict(stage2.get("greedy_anchor_accuracy_gate", {}) or {})
        search = dict(self.config.payload.get("search", {}) or {})
        return {
            "family_id": self.config.family.family_id,
            "family_name": self.config.family.display_name,
            "runner_kind": self.config.family.runner_kind,
            "search_method": str(search.get("method", "ga")),
            "activation_taylor_included": bool(
                proxy.get("include_activation_taylor", False)
            ),
            "greedy_beam_recovery": {
                "beam_width": int(search.get("budget_recovery_beam_width", 8)),
                "seed_pool_size": int(search.get("budget_recovery_seed_pool_size", 32)),
                "max_depth": int(search.get("budget_recovery_max_depth", 64)),
            },
            "stage2_greedy_anchor_gate": gate,
            "unresolved_placeholders": list(self.config.unresolved_placeholders),
            "full_search_executed": False,
        }

    def run(self, *, dry_run: bool = False) -> dict[str, Any]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.progress.phase(1, 5, f"解析 {self.config.family.display_name} 搜索配置")
        plan = self.plan()
        plan_path = self.output_root / "unified_search_plan.json"
        plan_path.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if dry_run:
            self.progress.phase(5, 5, "最小流程检查完成，未执行 GA 或构建引擎")
            return {**plan, "status": "dry_run_ok", "plan_path": str(plan_path)}

        backend = self.backends.get(self.config.family.runner_kind)
        if backend is None:
            backend = self._default_backend
        self.progress.phase(2, 5, "加载模型、校准清单与搜索空间")
        self.progress.phase(3, 5, "执行 Stage1 GA/greedy 搜索")
        result = dict(backend(self.config, self.output_root))
        self.progress.phase(4, 5, "执行 Stage2 精度门、部署评估与候选选择")
        output = dict(self.config.payload.get("output", {}) or {})
        best_dir = Path(str(output.get("best_engine_dir", "best_engines")))
        if not best_dir.is_absolute():
            best_dir = self.output_root / best_dir
        publication = BestEnginePublisher(
            best_dir,
            mode=str(output.get("best_engine_publish_mode", "hardlink")),
        ).publish(result, family_id=self.config.family.family_id)
        self.progress.phase(5, 5, "搜索结果与最佳引擎归档完成")
        return {**result, "best_engine_publication": asdict(publication)}

    @staticmethod
    def _default_backend(
        config: ResolvedSearchConfig, output_root: Path
    ) -> Mapping[str, Any]:
        payload = config.payload
        model = dict(payload.get("model", {}) or {})
        family_id = config.family.family_id
        if config.family.runner_kind == "v2xvit_framework":
            from ..orchestration.v2xvit_formal_search import V2XViTFormalSearch

            return V2XViTFormalSearch(
                config=config,
                output_root=output_root,
            ).run()
        if config.family.runner_kind == "heal_lidar_baseline_two_stage":
            from ..orchestration.heal_lidar_baseline_search import (
                HealLidarBaselineTwoStageSearch,
            )

            runner_type = HealLidarBaselineTwoStageSearch
        else:
            from ..orchestration.lidar_pyramid_search import LidarPyramidTwoStageSearch

            runner_type = LidarPyramidTwoStageSearch
        runner = runner_type(
            config=payload,
            checkpoint=str(model["checkpoint"]),
            output_root=output_root,
            resume=None,
        )
        result = runner.run()
        return {**dict(result), "family_id": family_id}


__all__ = ["Backend", "UnifiedSearchRunner"]
