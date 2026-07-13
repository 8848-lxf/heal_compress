from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def _shape(module: str, group: str, macs: float):
    from search.proxy.runtime_shape_profiler import RuntimeLayerShape

    return RuntimeLayerShape(
        module_path=module,
        call_index=0,
        module_type="Conv2d",
        input_shape=(1, 8, 4, 4),
        output_shape=(1, 8, 4, 4),
        c_in=8,
        c_out=8,
        h_out=4,
        w_out=4,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        weight_shape=(8, 8, 1, 1),
        precision_group_id=group,
        macs=macs,
    )


def test_bops_uses_fp16_deploy_reference_and_int8_reduces_ratio() -> None:
    from search.candidate import CandidatePhenotype, PrecisionDecision
    from search.proxy.bops_proxy import BOPSProxy

    proxy = BOPSProxy(runtime_shapes=[_shape("a", "pg_a", 100.0), _shape("b", "pg_b", 100.0)])
    fp16 = CandidatePhenotype(precision_profile={
        "a": PrecisionDecision("FP16", "FP16"),
        "b": PrecisionDecision("FP16", "FP16"),
    })
    mixed = CandidatePhenotype(precision_profile={
        "a": PrecisionDecision("INT8", "INT8"),
        "b": PrecisionDecision("FP16", "FP16"),
    })

    fp16_metrics = proxy.evaluate_breakdown(fp16)
    mixed_metrics = proxy.evaluate_breakdown(mixed)

    assert fp16_metrics["R_bops_vs_fp16_deploy"] == 1.0
    assert mixed_metrics["R_bops_vs_fp16_deploy"] < 1.0
    assert mixed_metrics["int8_macs_ratio"] == 0.5


def test_bops_schedule_and_feasibility_first_sorting() -> None:
    from search.proxy.objective import bops_target_for_generation, feasibility_first_key

    assert bops_target_for_generation(0, 5, {"start": 0.95, "end": 0.80}) == 0.95
    assert bops_target_for_generation(4, 5, {"start": 0.95, "end": 0.80}) == 0.80

    feasible = {"F1": 10.0, "R_bops_vs_fp32": 0.79, "R_bops_vs_fp16_deploy": 1.58}
    infeasible = {"F1": 1.0, "R_bops_vs_fp32": 0.81, "R_bops_vs_fp16_deploy": 0.40}

    assert feasibility_first_key(feasible, bops_target=0.80) < feasibility_first_key(infeasible, bops_target=0.80)
