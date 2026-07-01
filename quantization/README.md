# Formal Quantization Deployment Tools

Default deployment path:

- `strategy`: `single_engine_maxK`
- `fixed_K`: `29696`
- `precision`: `fp16`
- calibration split: `train`
- evaluation split: `val`

INT8 is available only as an optional build/evaluation path. `dynamic_bucket`
and `padded_agent_static` are legacy comparison baselines, not defaults.

## CLIs

```bash
python -m quantization.export.export_single_engine_maxk_onnx \
  --config /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml \
  --checkpoint /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth \
  --fixed-k 29696 \
  --precision fp16 \
  --output-dir tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/onnx/fixedK29696/single_engine_maxK/

python -m quantization.build.build_single_engine_maxk_engine \
  --onnx <onnx_path> \
  --precision fp16 \
  --fixed-k 29696 \
  --trt-root /home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118 \
  --plugin quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so \
  --output-dir tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/engines/fixedK29696/dynamic_agent_single_engine_maxK/fp16/

python -m quantization.calibrate.dump_train_calibration_npz \
  --config <config.yaml> \
  --checkpoint <net_epoch_bestval_at17.pth> \
  --fixed-k 29696 \
  --num-frames 200 \
  --split train \
  --strategy single_engine_maxK \
  --output-dir tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/calibration/train_calib_single_engine_maxK29696_200/

python -m quantization.eval.evaluate_single_engine_maxk \
  --config <config.yaml> \
  --checkpoint <net_epoch_bestval_at17.pth> \
  --engine <engine_path> \
  --precision fp16 \
  --fixed-k 29696 \
  --split val \
  --output-dir tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/evaluation/single_engine_maxK_fixedK29696_fp16_formal_tool/

python -m quantization.reports.summarize_deployment \
  --root tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/ \
  --strategy single_engine_maxK \
  --fixed-k 29696
```

The PointPillarScatterTRT source mirror is under
`quantization/plugins/pointpillar_scatter_trt/`. This migration does not add an
external dynamic-K frontend or a `DynamicPillarVFEAndScatterTRT` plugin.
