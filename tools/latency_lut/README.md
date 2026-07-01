# TensorRT Latency LUT Tools

This toolchain builds and queries a TensorRT latency LUT for the fixed default
deployment shape:

- deploy mode: `single_engine_maxK`
- fixed K: `29696`
- TensorRT engine shape: single engine

It does not implement or depend on dynamic bucket routing, bucket switching, or
bucket padding.

## Collect Keys

```bash
python tools/latency_lut/collect_keys.py \
  --config configs/latency_lut/lidar_pyramid_single_engine_maxK29696.yaml \
  --output outputs/latency_lut/keys.jsonl \
  --deploy-mode single_engine_maxK \
  --fixed-k 29696
```

## Benchmark Subgraphs

Real TensorRT benchmarking is the default. Use `--dry-run` only to validate key
and export paths without building engines.

```bash
python tools/latency_lut/benchmark_subgraph.py \
  --keys outputs/latency_lut/keys.jsonl \
  --output outputs/latency_lut/lut_records.jsonl \
  --failed-output outputs/latency_lut/failed_keys.jsonl \
  --onnx-dir outputs/latency_lut/subgraphs_onnx \
  --engine-dir outputs/latency_lut/subgraphs_engine \
  --trtexec /home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118/targets/x86_64-linux-gnu/bin/trtexec \
  --device 0 \
  --module-filter backbone \
  --precision-filter TRT_FP16 TRT_FP32 \
  --warmup 50 \
  --repeat 200 \
  --resume
```

The benchmarker exports minimal ONNX subgraphs and parses trtexec `GPU Compute
Time` summaries into p50/p90/p95/p99/mean/std records.

Implemented real subgraphs:

- `conv_block`
- `residual_block`
- `compression_1x1`
- `head_branch`
- `fusion_block` surrogate with ego/infrastructure inputs, concat/add,
  compression conv, and fusion conv
- `pfn_block` surrogate with fixed `fixed_K=29696` point dimension

Explicit non-success statuses:

- `skipped_int8_not_implemented`: INT8 QDQ benchmark is not available unless
  `--allow-int8-fallback-to-fp16` is explicitly passed.
- `skipped_plugin_not_available`: PointPillarScatterTRT plugin runner is not
  connected yet.
- `skipped_subgraph_not_implemented`: a deployment unit has a schema key but no
  minimal ONNX exporter yet.
- `failed`: TensorRT or export failed unexpectedly; inspect `error_message` and
  `metadata.log_path`.

`LatencyLUTDatabase` only loads successful records (`success`, plus legacy
`ok`), so skipped/failed records can coexist in `lut_records.jsonl` without
polluting search-time latency estimates.

Missing INT8 LUT keys are returned as `match_type=unavailable` with high
uncertainty unless real `TRT_INT8_QDQ` records are present.

## Full-Engine Calibration Samples

The helper below samples candidate configs and predicts `T_lut_pred`. It only
writes calibration samples when a real full-engine build/benchmark command is
provided. Without `--command-template`, it writes a skipped report and refuses
to fabricate `T_real_*` values.

```bash
python tools/latency_lut/build_full_engine_calibration_samples.py \
  --lut outputs/latency_lut/lut_records.jsonl \
  --output outputs/latency_lut/full_engine_samples.jsonl \
  --num-samples 20
```

To connect a real runner, pass a command template that writes a JSON result with
`T_real_p50` or `real_engine_p50_ms`:

```bash
python tools/latency_lut/build_full_engine_calibration_samples.py \
  --lut outputs/latency_lut/lut_records.jsonl \
  --output outputs/latency_lut/full_engine_samples.jsonl \
  --num-samples 20 \
  --command-template "python your_runner.py --candidate {candidate_json} --output {output_json}"
```

## Validate

```bash
python tools/latency_lut/validate_lut.py \
  --samples outputs/latency_lut/full_engine_samples.jsonl \
  --lut outputs/latency_lut/lut_records.jsonl
```
