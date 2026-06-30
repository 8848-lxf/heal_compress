# Dynamic Fixed-K INT8 Deployment Report

mode | precision strategy | calibration frames | eval frames | AP@0.30 | AP@0.50 | AP@0.70 | mAP | mAP drop vs FP16 | execute p50 | forward p50 | FPS | notes
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
dynamic fixed-K plugin | fp32 | - | 50 | 0.8721 | 0.8534 | 0.7476 | 0.8244 | -0.0055 | 4.985856056213379 | 6.470624357461929 | 154.54459179766775 | per-N fixed-K bucket router
dynamic fixed-K plugin | fp16 | - | 50 | 0.8723 | 0.8523 | 0.7322 | 0.8189 | 0.0 | 2.148128032684326 | 3.6728251725435257 | 272.2699701242448 | per-N fixed-K bucket router
dynamic fixed-K plugin | native int8 | 50 | 50 | 0.8076 | 0.7873 | 0.6412 | 0.7454 | 0.0735 | 1.533087968826294 | 2.5472305715084076 | 392.58322791243216 | per-N fixed-K bucket router
dynamic fixed-K plugin | native int8 | 200 | 50 | 0.8257 | 0.8042 | 0.6327 | 0.7542 | 0.0647 | 1.5381120443344116 | 2.5267787277698517 | 395.760811585827 | per-N fixed-K bucket router
dynamic fixed-K plugin | int8 mixed heads-FP16 | 200 | 50 | 0.825 | 0.8024 | 0.6422 | 0.7565 | 0.0624 | 1.622912049293518 | 2.6664920151233673 | 375.024561981947 | per-N fixed-K bucket router
dynamic fixed-K plugin | fp32 | - | 200 | 0.8102 | 0.7712 | 0.598 | 0.7265 | -0.0014 | 4.987904071807861 | 6.449433043599129 | 155.05238882857623 | per-N fixed-K bucket router
dynamic fixed-K plugin | fp16 | - | 200 | 0.8105 | 0.7704 | 0.5944 | 0.7251 | 0.0 | 2.1432321071624756 | 3.358054906129837 | 297.7914381848215 | per-N fixed-K bucket router
dynamic fixed-K plugin | native int8 | 50 | 200 | 0.6855 | 0.6441 | 0.4513 | 0.5936 | 0.1315 | 1.5535039901733398 | 2.55439430475235 | 391.48223832927414 | per-N fixed-K bucket router
dynamic fixed-K plugin | native int8 | 200 | 200 | 0.7604 | 0.7154 | 0.5107 | 0.6622 | 0.0629 | 1.5445760488510132 | 2.5555528700351715 | 391.3047590309635 | per-N fixed-K bucket router
dynamic fixed-K plugin | int8 mixed heads-FP16 | 200 | 200 | 0.7605 | 0.7157 | 0.5079 | 0.6614 | 0.0637 | 1.6270079612731934 | 2.682555466890335 | 372.77887161797156 | per-N fixed-K bucket router

## Answers

- INT8 engine 是否成功构建: True
- INT8 engine 是否真的运行了 INT8 层: True
- PointPillarScatterTRT 在 INT8 engine 中以什么精度执行: [['fp16'], ['fp16']]
- INT8 相比 FP16 是否有 latency 收益: True
- INT8 相比 FP16 的 AP 损失是多少: 0.0629
- AP 损失主要发生在哪个 IoU: AP@0.70 drop=0.0837
- native INT8 是否可接受: False
- mixed precision INT8 是否优于 native INT8: False
- 是否建议进入 Q/DQ / ModelOpt 进一步优化: True
- 是否建议当前就把 INT8 纳入默认部署路径: False
- 当前默认部署路径是否仍为 dynamic_agent_dim + fixed-K bucket router + PointPillarScatterTRT: True
- padded_agent_static 是否仍只作为 baseline: True
