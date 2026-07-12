from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_bops_breakdown_contains_runtime_fields() -> None:
    from search.candidate import CandidatePhenotype, PrecisionDecision
    from search.proxy.bops_proxy import BOPSProxy
    from search.proxy.runtime_shape_profiler import RuntimeLayerShape

    shape = RuntimeLayerShape(
        module_path="conv",
        call_index=2,
        module_type="Conv2d",
        input_shape=(1, 4, 6, 5),
        output_shape=(1, 8, 6, 5),
        c_in=4,
        c_out=8,
        h_out=6,
        w_out=5,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        weight_shape=(8, 4, 3, 3),
        precision_group_id="pg_conv",
        macs=8640.0,
    )
    phenotype = CandidatePhenotype(precision_profile={"conv": PrecisionDecision("INT8", "INT8")})

    metrics = BOPSProxy(runtime_shapes=[shape]).evaluate_breakdown(phenotype)

    row = metrics["breakdown"][0]
    assert row["H_out"] == 6
    assert row["W_out"] == 5
    assert row["precision_group_id"] == "pg_conv"
    assert row["weight_bits"] == 8
    assert row["activation_bits"] == 8
    assert metrics["R_bops_vs_fp16_deploy"] == 0.25
