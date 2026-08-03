# dynamic_agent_single_engine_maxK

This deployment path tests a single TensorRT engine with true dynamic agent count `N` and fixed LiDAR voxel capacity `maxK=24064`. It is intended to reduce deployment complexity compared with the historical `dynamic_agent_dim` N/bucket router, while preserving the same PointPillarScatterTRT and `valid_voxel_mask` semantics.

## Why This Path

The historical dynamic path uses multiple engines:

- `record_len=1` -> N1 engine
- `record_len=2` -> N2 engine
- voxel bucket engines for K: 9728, 23040, 23552, 24064

`dynamic_agent_single_engine_maxK` instead exports one ONNX and builds one engine per precision. Runtime `N` is set by TensorRT input shapes, while all voxel tensors are padded to `maxK=24064`.

## Inputs

Fixed voxel inputs:

- `voxel_features`: `[24064, 32, 4]`, float32
- `voxel_coords`: `[24064, 4]`, int32
- `voxel_num_points`: `[24064]`, int32
- `valid_voxel_mask`: `[24064]`, float32

Dynamic agent input:

- `pairwise_t_matrix`: `[1, N, N, 4, 4]`, float32

If real `K > 24064`, the sample is skipped and recorded. Invalid padded voxels are masked by `valid_voxel_mask` and must not affect spatial features.

## PointPillarScatterTRT Dynamic N

The plugin now supports both the legacy 3-input ABI and the new 4-input dynamic-N ABI:

- legacy: `pillar_features, voxel_coords, valid_voxel_mask`
- dynamic N: `pillar_features, voxel_coords, valid_voxel_mask, pairwise_t_matrix`

For the dynamic-N ABI, `getOutputDimensions()` derives output batch `N` from `pairwise_t_matrix` dimension 1, and `enqueue()` reads runtime `N` from `outputDesc[0]`. The plugin still supports FP32/FP16 feature IO only; in INT8 engines it is an FP16 precision island. `voxel_coords` is int32 and `valid_voxel_mask` supports the stable float path used here.

## Commands

Environment:

```bash
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate modelopt
export CUDA_VISIBLE_DEVICES=0
export TRT_ROOT=${TENSORRT_ROOT}
export LD_LIBRARY_PATH=$TRT_ROOT/lib:$TRT_ROOT/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH
```

Plugin check:

```bash
python tests/quant_deploy/pointpillar_scatter_dynamic_n_plugin_check.py \
  --output_root tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare \
  --trt_root $TRT_ROOT \
  --device cuda:0 \
  --precision fp32
```

Export ONNX:

```bash
python tests/quant_deploy/export_dynamic_single_engine_maxk_onnx.py \
  --output_root tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare \
  --trt_root $TRT_ROOT \
  --device cuda:0 \
  --overwrite
```

Dump train calibration NPZ:

```bash
python tests/quant_deploy/dump_dynamic_single_engine_maxk_calibration.py \
  --output_root tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare \
  --num_calib_frames 50 \
  --calib_split train \
  --ensure_agent_coverage 1,2 \
  --device cuda:0

python tests/quant_deploy/dump_dynamic_single_engine_maxk_calibration.py \
  --output_root tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare \
  --num_calib_frames 200 \
  --calib_split train \
  --ensure_agent_coverage 1,2 \
  --device cuda:0
```

Build engines:

```bash
python tests/quant_deploy/build_dynamic_single_engine_maxk_trt_engine.py \
  --output_root tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare \
  --plugin_path tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/plugins/pointpillar_scatter_trt_build/libpointpillar_scatter_trt.so \
  --trt_root $TRT_ROOT \
  --precisions fp32 fp16 int8 \
  --calibration_frames 50 200 \
  --profile_calibration_frames 200
```

Evaluate on val split:

```bash
python tests/quant_deploy/run_dynamic_single_engine_maxk.py \
  --output_root tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare \
  --plugin_path tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/plugins/pointpillar_scatter_trt_build/libpointpillar_scatter_trt.so \
  --device cuda:0 \
  --precision all \
  --calibration_frames 50 200 \
  --eval_frames 50 200
```

## TensorRT Profile

The profile is derived from train calibration NPZ shapes:

| input | min | opt | max |
| --- | --- | --- | --- |
| voxel_features | `[24064,32,4]` | `[24064,32,4]` | `[24064,32,4]` |
| voxel_coords | `[24064,4]` | `[24064,4]` | `[24064,4]` |
| voxel_num_points | `[24064]` | `[24064]` | `[24064]` |
| valid_voxel_mask | `[24064]` | `[24064]` | `[24064]` |
| pairwise_t_matrix | `[1,1,1,4,4]` | `[1,2,2,4,4]` | `[1,2,2,4,4]` |

The new INT8 calibration split is `train`; evaluation split is `val`; `calibration_eval_overlap=false`.

## Results

50 val frames:

| precision | calibration | AP@0.30 | AP@0.50 | AP@0.70 | mAP | mAP drop vs FP16 | execute p50 ms | forward p50 ms | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP32 | n/a | 0.8721 | 0.8534 | 0.7464 | 0.8240 | -0.0054 | 10.84 | 25.72 | 38.88 |
| FP16 | n/a | 0.8723 | 0.8523 | 0.7312 | 0.8186 | 0.0000 | 1.81 | 16.69 | 59.90 |
| INT8 | train_calib50 | 0.7058 | 0.6911 | 0.5240 | 0.6403 | 0.1783 | 1.49 | 16.38 | 61.05 |
| INT8 | train_calib200 | 0.7843 | 0.7692 | 0.5992 | 0.7176 | 0.1010 | 1.52 | 16.39 | 61.03 |

200 val frames:

| precision | calibration | AP@0.30 | AP@0.50 | AP@0.70 | mAP | mAP drop vs FP16 | execute p50 ms | forward p50 ms | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP32 | n/a | 0.8102 | 0.7706 | 0.5965 | 0.7258 | -0.0012 | 10.88 | 25.75 | 38.84 |
| FP16 | n/a | 0.8103 | 0.7701 | 0.5935 | 0.7246 | 0.0000 | 1.82 | 16.65 | 60.07 |
| INT8 | train_calib50 | 0.6575 | 0.6199 | 0.4193 | 0.5656 | 0.1590 | 1.50 | 16.37 | 61.07 |
| INT8 | train_calib200 | 0.7255 | 0.6777 | 0.4690 | 0.6241 | 0.1005 | 1.53 | 16.41 | 60.95 |

The 200-frame val runs skipped 218 samples with `K > 24064` while collecting 200 valid frames.

## Comparison

Compared with the historical dynamic bucket FP16 200-frame result, single-engine FP16 keeps AP (`mAP 0.7246` vs `0.7251`) but forward p50 is much slower (`16.65 ms` vs `3.36 ms`). This fails the replacement criterion because latency is not within 10%.

Single-engine INT8 runs many INT8 layers, but the plugin remains FP16. The best native INT8 result here is train_calib200 with `mAP drop 0.1005` on 200 val frames and `0.1010` on 50 val frames, just above the 0.1 candidate threshold. AP@0.70 is the most sensitive metric.

## Recommendation

Do not replace the current default deployment path yet. Keep:

`dynamic_agent_dim + fixed-K bucket router + PointPillarScatterTRT FP16`

Use `dynamic_agent_single_engine_maxK` as a validated research path, not the default deployment path. For INT8, try a mixed precision whitelist before full Q/DQ or ModelOpt: preserve plugin/PFN/heads in FP16 and quantize BEV backbone/shrink where possible.

`padded_agent_static` remains only an optional baseline.

## Known Limits

- `fixed_K=24064`; samples with `K > 24064` are skipped.
- Single-engine runner currently pays large fixed maxK input-copy overhead in forward latency.
- PointPillarScatterTRT INT8 IO is not supported; the plugin is FP16/FP32 island in INT8 engines.
- AP@0.70 is sensitive to INT8 score and regression drift.
- Calibration split must be `train`.
- Evaluation split must be `val`.
