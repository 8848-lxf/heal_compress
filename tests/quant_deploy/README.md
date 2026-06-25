# LiDAR Pyramid TensorRT Quant Deployment

This directory contains the LiDAROnly / `lidar_pyramid` deployment test framework for HEAL/OpenCOOD.

The source scripts live directly under `tests/quant_deploy/`. Runtime outputs must be written under:

```text
tests/quant_deploy/outputs/<run_name>/
```

No ONNX, TensorRT engine, benchmark, calibration, evaluation, log, or summary output should be written to `tests/outputs/` or the project root.

## Supported Precision Modes

- `fp32`: TensorRT baseline from the exported FP32 ONNX. Uses `--noTF32` by default.
- `fp16`: TensorRT practical FP16 from the exported FP32 ONNX. Uses `--fp16`; no Q/DQ or calibration.
- `int8`: CLI and directory interface are reserved. Q/DQ insertion and calibration are not implemented in this stage.

`--strict_fp16` optionally attempts an additional strict FP16 build with TensorRT precision constraints. A strict build failure is recorded separately and does not change the practical FP16 result.

## Environment

Use the server `modelopt` conda environment and TensorRT 10.9 installation:

```bash
cd /home/lixingfeng/UniAD_examine/heal_compress
conda activate modelopt
source tests/quant_deploy/env_modelopt_trt.sh
```

The environment script sets:

```text
TRT_ROOT=/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118
PATH=${TRT_ROOT}/bin:${TRT_ROOT}/targets/x86_64-linux-gnu/bin:${PATH}
LD_LIBRARY_PATH=<nvidia pip libs>:${TRT_ROOT}/lib:${TRT_ROOT}/targets/x86_64-linux-gnu/lib:<torch lib>:${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${LD_LIBRARY_PATH}
```

It also prints the active Python, `trtexec`, PyTorch CUDA status, TensorRT import status, and ModelOpt import status.

`find_trtexec()` resolves TensorRT in this order:

```text
1. --trtexec_path
2. --trt_root/bin/trtexec
3. --trt_root/targets/x86_64-linux-gnu/bin/trtexec
4. PATH
```

Every run writes `debug/env_report.json` with Python, conda, CUDA, TensorRT, ModelOpt, TRT_ROOT, LD_LIBRARY_PATH, CUDA_VISIBLE_DEVICES, GPU names, and trtexec discovery details.

## BEV Warp Export

`lidar_pyramid` uses pyramid BEV warp. The original HEAL function calls `torch.nn.functional.affine_grid`, which fails PyTorch ONNX export as `aten::affine_grid_generator`.

The default export mode is:

```text
--bev_warp_export_mode exportable_grid
```

This mode monkey-patches only the ONNX export process:

```text
opencood.models.sub_modules.torch_transformation_utils.warp_affine_simple
opencood.models.fuse_modules.pyramid_fuse.warp_affine_simple
```

with an ONNX-exportable grid implementation that avoids `F.affine_grid` and keeps `F.grid_sample`. The original functions are restored after export. A small numerical check is written to:

```text
debug/bev_warp_equivalence_report.json
```

The target after export is:

```text
AffineGrid / aten::affine_grid_generator: absent
GridSample: may remain
```

If TensorRT later rejects `GridSample`, the build log and debug reports should be used to decide whether a BEV warp plugin is needed.

## PillarVFE Squeeze Export Fix

HEAL `opencood/models/sub_modules/pillar_vfe.py` ends `PillarVFE.forward()` with:

```python
features = features.squeeze()
```

The final PFN layer returns `[M, 1, C]`, so deployment only needs `[M, 1, C] -> [M, C]`. A dimensionless ONNX `Squeeze` fails TensorRT dynamic-shape parsing. The default export-only fix is:

```text
--pillar_vfe_export_fix explicit_squeeze
```

It temporarily monkey-patches `PillarVFE.forward` during ONNX export and changes only the final squeeze to `features.squeeze(1)`. The original HEAL function is restored after export. The numerical check is written to:

```text
debug/pillar_vfe_export_fix_report.json
```

After export, `debug/special_ops_report.json` records every `Squeeze` node and whether it has an axes input. The expected result for this fix is:

```text
/model/encoder_m1/pillar_vfe/Squeeze: has_axes_input=true
Squeeze_without_axes: 0
```

## Fixed Pyramid Export Forward

The original HEAL pyramid fusion path uses Python list operations in `regroup`, `forward_collab`, and multiscale feature decoding. PyTorch ONNX export can lower those paths to `SequenceEmpty`, `SequenceInsert`, and `SequenceAt`, which TensorRT does not accept for this deployment graph.

The default export-only fix is:

```text
--pyramid_forward_export_mode fixed_static
```

This reuses the fixed-forward idea from:

```text
/home/lixingfeng/UniAD_examine/HEAL/prune_model/pyramid-trt/export_dynamic_onnx.py
```

but keeps only the LiDAR path. It statically expands the 3 pyramid levels, avoids list/dict/Sequence outputs, and returns a fixed Tensor tuple. It does not modify HEAL/OpenCOOD source and does not affect PyTorch evaluation. Numerical comparison with the original model outputs is written to:

```text
debug/fixed_pyramid_forward_report.json
```

Expected ONNX result:

```text
SequenceEmpty: 0
SequenceInsert: 0
SequenceAt: 0
```

## One-Key Run

```bash
python tests/quant_deploy/run_lidar_pyramid_deploy.py \
  --checkpoint /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth \
  --hypes_yaml /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml \
  --output_dir tests/quant_deploy/outputs \
  --run_name lidar_pyramid_fp32_fp16_baseline \
  --precisions fp32 fp16 \
  --num_frames 50 \
  --device cuda:0 \
  --opset 17 \
  --trt_root /home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118 \
  --bev_warp_export_mode exportable_grid \
  --pillar_vfe_export_fix explicit_squeeze \
  --pyramid_forward_export_mode fixed_static
```

Recommended current command:

```bash
python tests/quant_deploy/run_lidar_pyramid_deploy.py \
  --hypes_yaml /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml \
  --checkpoint /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth \
  --output_dir tests/quant_deploy/outputs \
  --run_name lidar_pyramid_fp32_fp16_exportable_warp \
  --precisions fp32 fp16 \
  --device cuda:0 \
  --num_frames 50 \
  --opset 17 \
  --trt_root /home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118 \
  --bev_warp_export_mode exportable_grid \
  --pillar_vfe_export_fix explicit_squeeze \
  --pyramid_forward_export_mode fixed_static
```

## Separate Steps

```bash
python tests/quant_deploy/export_lidar_pyramid_onnx.py \
  --hypes_yaml /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml \
  --checkpoint /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth \
  --output_dir tests/quant_deploy/outputs \
  --run_name lidar_pyramid_onnx_exportable_warp \
  --device cuda:0 \
  --opset 17 \
  --bev_warp_export_mode exportable_grid \
  --pillar_vfe_export_fix explicit_squeeze \
  --pyramid_forward_export_mode fixed_static

python tests/quant_deploy/build_lidar_pyramid_trt_engine.py \
  --onnx_path tests/quant_deploy/outputs/lidar_pyramid_fp32_fp16_exportable_warp/artifacts/onnx/fp32/lidar_pyramid_fp32_dynamic.onnx \
  --output_root tests/quant_deploy/outputs/lidar_pyramid_fp32_fp16_exportable_warp \
  --precision fp16 \
  --trt_root /home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118

python tests/quant_deploy/benchmark_lidar_pyramid_trt_engine.py \
  --output_root tests/quant_deploy/outputs/<run_name> \
  --precision fp16 \
  --num_frames 50 \
  --warmup_frames 10
```

## Output Layout

Each run creates:

```text
configs/
artifacts/onnx/fp32/
artifacts/onnx/qdq_int8/
artifacts/engines/fp32/
artifacts/engines/fp16/
artifacts/engines/int8/
calibration/
benchmark/fp32/
benchmark/fp16/
benchmark/int8/
evaluation/fp32/
evaluation/fp16/
evaluation/int8/
logs/export/
logs/build/
logs/benchmark/
logs/evaluation/
summary/
debug/
```

`summary/summary_all.json` and `summary/summary_all.md` record the directory map, export status, build status, benchmark status, latency metrics, special ONNX ops, and failure reasons.

## INT8 Entry Points

The reserved INT8 parameters are:

```text
--calib_dir
--calib_num_frames
--calib_cache
--qdq_onnx_path
--int8_mode qdq
--allow_fp16_fallback
```

The future INT8 path should be:

```text
FP32 ONNX -> calibration data -> explicit ONNX Q/DQ -> TensorRT INT8 engine -> benchmark/evaluation
```

INT8 Q/DQ should be based on the successfully exported FP32 ONNX and then use ModelOpt for explicit Q/DQ quantization. Do not start ModelOpt INT8 from an FP16 ONNX.

Calibration outputs should go only under `calibration/`; Q/DQ ONNX should go under `artifacts/onnx/qdq_int8/`.
