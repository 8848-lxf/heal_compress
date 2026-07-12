from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_gpu_batch_proxy_matches_cpu_reference_for_small_population() -> None:
    import torch

    from search.proxy.gpu_batch_proxy import BatchProxyTables, score_candidates_batched, score_candidates_cpu

    tables = BatchProxyTables(
        prune_loss=torch.tensor([0.0, 1.0, 2.0]),
        sqnr_by_group_precision=torch.tensor(
            [
                [0.0, 0.1, 0.2],
                [0.0, 0.3, 0.6],
            ],
            dtype=torch.float32,
        ),
        size_by_group_precision=torch.tensor(
            [
                [32.0, 16.0, 8.0],
                [64.0, 32.0, 16.0],
            ],
            dtype=torch.float32,
        ),
        bops_by_group_precision=torch.tensor(
            [
                [320.0, 80.0, 20.0],
                [640.0, 160.0, 40.0],
            ],
            dtype=torch.float32,
        ),
        fp16_size_baseline=48.0,
        fp16_bops_baseline=240.0,
    )
    prune_choices = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.int64)
    precision_choices = torch.tensor([[1, 1], [2, 1]], dtype=torch.int64)

    cpu = score_candidates_cpu(tables, prune_choices, precision_choices, bops_target=0.95)
    gpu = score_candidates_batched(tables, prune_choices, precision_choices, bops_target=0.95, device="cpu")

    assert torch.allclose(cpu["F1"], gpu["F1"])
    assert torch.allclose(cpu["R_bops_vs_fp16"], gpu["R_bops_vs_fp16"])
    assert gpu["candidate_count"] == 2
