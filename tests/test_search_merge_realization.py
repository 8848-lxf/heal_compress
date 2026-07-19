from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_concat_merge_matches_trt_compiler_backend_output_tensor_name(tmp_path: Path) -> None:
    from search.stage2.lidar_pyramid_real_evaluator import (
        _engine_merge_precision_realization,
    )

    layer_info = tmp_path / "engine_layer_info.json"
    layer_info.write_text(
        json.dumps(
            {
                "Layers": [
                    {
                        "Name": f"__myl_Move_myl{index}_0",
                        "LayerType": "kgen",
                        "Inputs": [
                            {
                                "Name": f"/deblocks.{index}/Relu_output_0",
                                "Format/Datatype": "Half",
                            }
                        ],
                        "Outputs": [
                            {
                                "Name": "/Concat_9_output_0",
                                "Format/Datatype": "Half",
                            }
                        ],
                        "Metadata": "",
                    }
                    for index in range(3)
                ]
            }
        ),
        encoding="utf-8",
    )
    qdq = SimpleNamespace(
        calibration_metadata={
            "merge_quantization_audit": [
                {
                    "merge_op_name": "/Concat_9",
                    "merge_op_type": "Concat",
                    "output_tensors": ["/Concat_9_output_0"],
                    "input_branches": [
                        {"cast_to_fp16": True},
                        {"cast_to_fp16": True},
                        {"cast_to_fp16": True},
                    ],
                    "downstream": [],
                }
            ]
        }
    )

    report = _engine_merge_precision_realization(layer_info, qdq)

    assert report["passed"] is True
    merge = report["merges"][0]
    assert merge["realized_merge_precision"] == "FP16"
    assert merge["engine_optimization"] == "compiler_backend_merge_tensor_match"
    assert merge["engine_layer_names"] == [
        "__myl_Move_myl0_0",
        "__myl_Move_myl1_0",
        "__myl_Move_myl2_0",
    ]


def test_concat_merge_accepts_strongly_typed_metadata_fusion_with_downstream_cast(
    tmp_path: Path,
) -> None:
    from search.stage2.lidar_pyramid_real_evaluator import (
        _engine_merge_precision_realization,
    )

    layer_info = tmp_path / "engine_layer_info.json"
    layer_info.write_text(
        json.dumps(
            {
                "Layers": [
                    {
                        "Name": "__myl_GridConcatCast_myl202_6",
                        "LayerType": "kgen",
                        "Inputs": [
                            {
                                "Name": "warped",
                                "Format/Datatype": "Half",
                            },
                            {
                                "Name": "ego",
                                "Format/Datatype": "Half",
                            },
                        ],
                        "Outputs": [
                            {
                                "Name": "pixel_weight_input",
                                "Format/Datatype": "Float",
                            }
                        ],
                        "Metadata": (
                            "[ONNX Layer: /GridSample]\u001f"
                            "[ONNX Layer: /Concat_7]\u001f"
                            "[ONNX Layer: __typed__pixel_weight_input__Cast]"
                        ),
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
                    "merge_op_name": "/Concat_7",
                    "merge_op_type": "Concat",
                    "output_tensors": ["/Concat_7_output_0"],
                    "input_branches": [
                        {"cast_to_fp16": True},
                        {"cast_to_fp16": True},
                    ],
                    "downstream": [],
                }
            ]
        }
    )

    report = _engine_merge_precision_realization(layer_info, qdq)

    assert report["passed"] is True
    merge = report["merges"][0]
    assert merge["realized_merge_precision"] == "FP16"
    assert merge["engine_optimization"] == (
        "graph_constrained_fp16_concat_fused_with_downstream_cast"
    )
    assert merge["engine_layer_names"] == ["__myl_GridConcatCast_myl202_6"]
