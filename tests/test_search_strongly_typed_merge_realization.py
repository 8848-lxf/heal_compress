from __future__ import annotations

import json
from types import SimpleNamespace


def test_strongly_typed_fused_add_is_resolved_by_exact_branch_tensors(tmp_path) -> None:
    from search.stage2.lidar_pyramid_real_evaluator import _engine_merge_precision_realization

    left = "__merge_fp16__unit__input00__Cast__output"
    right = "__merge_fp16__unit__input01__Cast__output"
    layer_info = tmp_path / "layers.json"
    layer_info.write_text(
        json.dumps(
            {
                "Layers": [
                    {
                        "Name": "__myl_CastCastAddMaxCast_fused",
                        "LayerType": "kgen",
                        "Inputs": [
                            {"Name": left, "Format/Datatype": "Half"},
                            {"Name": right, "Format/Datatype": "Half"},
                        ],
                        "Outputs": [
                            {"Name": "/block/relu/Relu_output_0", "Format/Datatype": "Half"}
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    qdq = SimpleNamespace(
        calibration_metadata={
            "merge_quantization_audit": [
                {
                    "merge_op_name": "/block/Add",
                    "merge_op_type": "Add",
                    "input_branches": [
                        {"tensor": left, "cast_to_fp16": True},
                        {"tensor": right, "cast_to_fp16": True},
                    ],
                    "output_tensors": ["/block/Add_output_0"],
                }
            ]
        }
    )

    report = _engine_merge_precision_realization(layer_info, qdq)

    assert report["passed"] is True
    assert report["merges"][0]["realized_merge_precision"] == "FP16"
    assert report["merges"][0]["engine_optimization"] == "fused_by_exact_graph_input_tensor_set"


def test_strongly_typed_concat_can_be_proven_by_explicit_fp16_graph_contract(tmp_path) -> None:
    from search.stage2.lidar_pyramid_real_evaluator import _engine_merge_precision_realization

    layer_info = tmp_path / "layers.json"
    layer_info.write_text(json.dumps({"Layers": []}), encoding="utf-8")
    qdq = SimpleNamespace(
        calibration_metadata={
            "merge_quantization_audit": [
                {
                    "merge_op_name": "/Concat_9",
                    "merge_op_type": "Concat",
                    "input_branches": [
                        {
                            "tensor": f"__merge_fp16__concat__input{index:02d}__Cast__output",
                            "cast_to_fp16": True,
                        }
                        for index in range(3)
                    ],
                    "downstream": [
                        {"consumer": "downstream_fp32_cast", "op_type": "Cast"}
                    ],
                    "output_tensors": ["/Concat_9_output_0"],
                }
            ]
        }
    )

    report = _engine_merge_precision_realization(layer_info, qdq)

    assert report["passed"] is True
    assert report["merges"][0]["realized_merge_precision"] == "FP16"
    assert (
        report["merges"][0]["engine_optimization"]
        == "graph_constrained_fp16_concat_fused_with_downstream"
    )
