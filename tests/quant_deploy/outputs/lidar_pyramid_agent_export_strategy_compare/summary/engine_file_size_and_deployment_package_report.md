# Engine File Size and Deployment Package Report

- output_root: /home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare
- fixed_K: 29696
- full_val_tag: full_val_fixedK29696_trainCalib
- existing full-val AP/latency reports were reused; no K scan, calibration dump, or trusted full-val re-evaluation was run.

scheme | engine_strategy | precision | calibration | engine_count | total_engine_size_MB | ratio_vs_single_same_precision | AP@0.30 | AP@0.50 | AP@0.70 | mAP | forward_p50 | FPS | notes
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
padded_agent_static | padded_agent_static_fixed_k_plugin | fp32 |  | 4 | 120.3061 | 2.862 | 0.8263 | 0.7817 | 0.6029 | 0.737 | 6.3414 | 157.6944 | 
padded_agent_static | padded_agent_static_fixed_k_plugin | fp16 |  | 4 | 88.974 | 2.9286 | 0.826 | 0.782 | 0.601 | 0.7363 | 3.3626 | 297.3853 | 
dynamic_agent_dim | dynamic_agent_dim_bucket_fixed_k_plugin | fp32 |  | 8 | 241.7984 | 5.7522 | 0.8263 | 0.7817 | 0.6026 | 0.7369 | 6.1406 | 162.8506 | deployment package engine_count=8 differs from full-val evaluated engine_count=5
dynamic_agent_dim | dynamic_agent_dim_bucket_fixed_k_plugin | fp16 |  | 8 | 176.1661 | 5.7985 | 0.826 | 0.7818 | 0.6013 | 0.7364 | 3.2826 | 304.6342 | deployment package engine_count=8 differs from full-val evaluated engine_count=5
dynamic_agent_dim | dynamic_agent_dim_bucket_int8 | int8 | train_calib50 | 5 | 91.904 | 3.4527 | 0.3299 | 0.3104 | 0.2104 | 0.2836 | 2.7093 | 369.0979 | 
dynamic_agent_dim | dynamic_agent_dim_bucket_int8 | int8 | train_calib200 | 5 | 91.8705 | 3.4306 | 0.7117 | 0.6683 | 0.4783 | 0.6194 | 2.7009 | 370.2481 | 
dynamic_agent_single_engine_maxK | single TensorRT engine | fp32 |  | 1 | 42.0362 | 1 | 0.8262 | 0.7815 | 0.6027 | 0.7368 | 6.4862 | 154.1741 | 
dynamic_agent_single_engine_maxK | single TensorRT engine | fp16 |  | 1 | 30.3816 | 1 | 0.8259 | 0.7818 | 0.6012 | 0.7363 | 3.4408 | 290.6283 | 
dynamic_agent_single_engine_maxK | single TensorRT engine | int8 | train_calib50 | 1 | 26.6178 | 1 | 0.6416 | 0.6062 | 0.4239 | 0.5572 | 2.8794 | 347.2993 | 
dynamic_agent_single_engine_maxK | single TensorRT engine | int8 | train_calib200 | 1 | 26.78 | 1 | 0.7048 | 0.6632 | 0.4691 | 0.6124 | 2.8709 | 348.318 | 
padded_agent_static | padded_agent_static_fixed_k_plugin | int8 | train_calib200 | 4 | 71.1101 | 2.6553 | 0.6594 | 0.6184 | 0.4303 | 0.5694 | 2.815 | 355.2456 | 

## Answers

- dynamic bucket FP16 total engine size MB: 176.1661
- single_engine_maxK FP16 engine size MB: 30.3816
- dynamic bucket FP16 size ratio vs single_engine_maxK FP16: 5.7985x
- dynamic bucket INT8 train_calib200 total engine size MB: 91.8705
- single_engine_maxK INT8 train_calib200 engine size MB: 26.78
- dynamic bucket INT8 train_calib200 size ratio vs single_engine_maxK INT8: 3.4306x
- padded static FP16 total engine size MB: 88.974
- multi bucket / route engines significantly increase package size: True
- dynamic bucket FP16 forward p50 speedup vs single_engine_maxK FP16: 4.6%
- dynamic bucket INT8 train_calib200 forward p50 speedup vs single_engine_maxK INT8: 5.92%
- padded_agent_static INT8 train_calib200 evaluated: True
- padded_agent_static INT8 train_calib200 skip reason: None
- final recommended default path: dynamic_agent_single_engine_maxK FP16 fixedK29696 + PointPillarScatterTRT
- dynamic bucket FP16 role: speed upper-bound / optional low-latency route when package size is acceptable
- INT8 recommended as default: False
- recommend mixed precision whitelist / Q-DQ / ModelOpt next: True
