from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class DeblockConcatToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.deblock_a = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.deblock_b = nn.ConvTranspose2d(16, 12, 2, stride=2)
        self.consumer = nn.Conv2d(20, 10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.deblock_a(x)
        b = self.deblock_b(x)
        return self.consumer(torch.cat([a, b], dim=1))


class UnsupportedDeblockConsumerToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.deblock = nn.ConvTranspose2d(16, 8, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.deblock(x)
        grid = torch.zeros(y.shape[0], y.shape[2], y.shape[3], 2, device=y.device, dtype=y.dtype)
        return F.grid_sample(y, grid, align_corners=False)


def test_deblock_concat_offset_pruning_syncs_downstream_consumer() -> None:
    from heal_compress.pruning.physical_prune_plan import GlobalPhysicalPrunePlan, ModuleAxisPruneRequest
    from tools.latency_lut.audit_dependency_graph_v99 import build_deblock_concat_dependency_proof

    model = DeblockConcatToy().eval()
    sample = torch.randn(2, 16, 8, 8)
    rows = build_deblock_concat_dependency_proof(model, sample, apply_smoke=True)

    b_row = next(row for row in rows if row["deblock_module"] == "deblock_b")
    assert b_row["branch_offset"] == 8
    assert b_row["branch_channels_before"] == 12
    assert b_row["downstream_consumer"] == "consumer"
    assert b_row["downstream_input_indices_synced"] is True
    assert b_row["offset_proof_available"] is True
    assert b_row["physical_prune_supported"] is True
    assert b_row["simulator_legal"] is True
    assert b_row["forward_smoke_status"] == "forward_passed"

    # Direct physical contract check for deblock_b.out[j] -> consumer.in[8+j].
    plan = GlobalPhysicalPrunePlan()
    plan.add_request(ModuleAxisPruneRequest("deblock_b", "out", [2], source_recipe_id="toy"))
    plan.add_request(ModuleAxisPruneRequest("consumer", "in", [10], source_recipe_id="toy"))
    plan.apply_one_shot(model)
    assert model.deblock_b.out_channels == 11
    assert model.consumer.in_channels == 19
    assert model(sample).shape == (2, 10, 16, 16)


def test_deblock_unsupported_downstream_consumer_remains_protected() -> None:
    from tools.latency_lut.audit_dependency_graph_v99 import build_deblock_concat_dependency_proof

    rows = build_deblock_concat_dependency_proof(
        UnsupportedDeblockConsumerToy().eval(),
        torch.randn(1, 16, 8, 8),
        apply_smoke=False,
    )

    assert rows
    assert rows[0]["physical_prune_supported"] is False
    assert rows[0]["failure_reason"] == "protected_deblock_unsupported_downstream_consumer"
