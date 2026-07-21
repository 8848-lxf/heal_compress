from __future__ import annotations


def test_cobevt_scatter_symbolic_reuses_registered_plugin_contract():
    from quantization.export.heal_lidar_cobevt import CobevtPointPillarScatterTRT

    calls = []

    class Graph:
        def op(self, name, *inputs, **attributes):
            calls.append((name, inputs, attributes))
            return "spatial"

    result = CobevtPointPillarScatterTRT.symbolic(
        Graph(), "pillar", "coords", "mask", "pairwise", 256, 512
    )

    assert result == "spatial"
    assert calls[0][0] == "trt::PointPillarScatterTRT"
    assert calls[0][2]["height_i"] == 256
    assert calls[0][2]["width_i"] == 512
    assert calls[0][2]["plugin_version_s"] == "1"


def test_cobevt_scatter_is_not_a_precision_gene():
    from quantization.export.heal_lidar_cobevt import scatter_export_capability

    report = scatter_export_capability(boundary_dtype="fp32")

    assert report["op_type"] == "PointPillarScatterTRT"
    assert report["boundary_dtype"] == "FP32"
    assert report["quantization_gene"] is False
    assert report["int8_allowed"] is False
