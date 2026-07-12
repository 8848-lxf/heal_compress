"""Stage-2 physical pruning export orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from ..adapters.pruning_adapter import FormalPruningAdapter
from ..candidate import CandidatePhenotype


class PhysicalExportStage:
    """Run the formal pruning adapter and persist core artifacts."""

    def __init__(self, adapter: FormalPruningAdapter | None = None) -> None:
        self.adapter = adapter or FormalPruningAdapter()

    def run(
        self,
        *,
        model: Any,
        phenotype: CandidatePhenotype,
        atomic_units: Sequence[Any],
        output_dir: str | Path,
        example_inputs: Any | None = None,
    ) -> dict[str, Any]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        request = self.adapter.request_from_phenotype(phenotype, atomic_units)
        result = self.adapter.materialize_from_request(model, request, example_inputs=example_inputs)
        validation = result["validation"]
        if hasattr(validation, "passed") and not validation.passed:
            return {"stage2_status": "prune_failed", "failure_reason": getattr(validation, "issues", [])}
        model_path = destination / "pruned_model.pth"
        torch.save({"model": result["model"].state_dict()}, model_path)
        return {**result, "stage2_status": "ok", "pruned_model_path": str(model_path), "request": request}
