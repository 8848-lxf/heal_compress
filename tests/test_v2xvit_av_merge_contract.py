from __future__ import annotations

import pytest


def test_av_profiles_use_probability_and_value_operand_bits() -> None:
    from search.quantization_space.v2xvit_av_merge import av_bops, av_contract

    assert av_bops(10, "AV32") == 10 * 32 * 32
    assert av_bops(10, "AV16") == 10 * 16 * 16
    assert av_bops(10, "AV8") == 10 * 8 * 8
    assert av_contract("AV8").softmax_compute == "FP32"
    assert av_contract("AV8").probability_precision == "INT8"
    assert av_contract("AV8").value_precision == "INT8"


@pytest.mark.parametrize(
    "op,inputs,scales,capability,expected",
    [
        ("Add", ["INT8", "INT8"], [0.1, 0.1], {"add_int8": True}, "INT8"),
        ("Add", ["INT8", "INT8"], [0.1, 0.2], {"add_int8": True}, "FP16"),
        ("Add", ["INT8", "FP16"], [0.1, None], {"add_int8": True}, "FP16"),
        ("Add", ["FP16", "FP32"], [None, None], {}, "FP32"),
        ("Add", ["FP32", "FP32", "FP32"], [], {}, "FP32"),
        ("Concat", ["INT8", "INT8"], [0.1, 0.1], {"concat_int8": True}, "INT8"),
        ("Concat", ["INT8", "INT8"], [0.1, 0.2], {"concat_int8": False}, "FP16"),
        ("Reshape", ["FP16"], [], {}, "FP16"),
    ],
)
def test_derived_merge_precision_cases(op, inputs, scales, capability, expected) -> None:
    from search.quantization_space.v2xvit_av_merge import derive_merge_precision

    result = derive_merge_precision(op, inputs, scales, trt_capability=capability)
    assert result["derived_precision"] == expected
    assert result["independent_chromosome_gene"] is False


def test_shape_merge_rejects_mixed_input_dtypes() -> None:
    from search.quantization_space.v2xvit_av_merge import derive_merge_precision

    with pytest.raises(ValueError, match="shape_merge_mixed_inputs"):
        derive_merge_precision("reshape", ["INT8", "FP16"])
