# PointPillarScatterTRT Plugin LUT Benchmark TODO

Current status: `skipped_plugin_not_available`.

The latency LUT contains schema keys for the `PointPillarScatterTRT` deployment
unit under the fixed deployment mode:

- `deploy_mode = single_engine_maxK`
- `fixed_K = 29696`
- plugin name: `PointPillarScatterTRT`

This pass does not generate fake plugin latency. A plugin key is written with
`status=skipped_plugin_not_available` until a real benchmark runner is connected.

Required runner contract:

1. Load the built plugin shared library, for example
   `quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so`.
2. Construct a minimal TensorRT network containing only the plugin wrapper with
   fixed input shape for `fixed_K=29696`.
3. Use representative BEV dimensions and channel count from the LUT key:
   `H`, `W`, `C_in/C_out`.
4. Build a TensorRT engine with the requested profile:
   `TRT_FP32`, `TRT_FP16`, or later `TRT_INT8_QDQ`.
5. Benchmark with CUDA events or `trtexec` and write p50/p90/p95/p99/mean/std
   into `outputs/latency_lut/lut_records.jsonl`.

Until this runner exists, the plugin benchmark path must stay skipped and
`LatencyProxy` will either use a configured default/uncertainty or report a
missing key for candidates that require the plugin latency.

