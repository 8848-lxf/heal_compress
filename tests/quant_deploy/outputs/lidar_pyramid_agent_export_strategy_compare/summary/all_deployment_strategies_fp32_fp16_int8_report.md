# All Deployment Strategies FP32/FP16/INT8

scheme | engine_strategy | precision | calibration_split | evaluation_split | calibration | frames | engine_count | dynamic_N | bucket_router | maxK | AP@0.30 | AP@0.50 | AP@0.70 | mAP | mAP_drop_vs_FP16 | execute_p50 | forward_p50 | FPS | notes
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
padded_agent_static fixed-K plugin | bucket_router_baseline | fp32 | None | None | None | 50 | 4 | False | True | 24064 | 0.8721 | 0.8534 | 0.745 | 0.8235 | None | 5.133279800415039 | 6.905348971486092 | 144.81527351177317 | optional baseline, not default
padded_agent_static fixed-K plugin | bucket_router_baseline | fp32 | None | None | None | 200 | 4 | False | True | 24064 | 0.8102 | 0.7712 | 0.598 | 0.7265 | None | 5.1609601974487305 | 6.964974105358124 | 143.57555173546223 | optional baseline, not default
padded_agent_static fixed-K plugin | bucket_router_baseline | fp16 | None | None | None | 50 | 4 | False | True | 24064 | 0.8723 | 0.8523 | 0.7312 | 0.8186 | None | 2.1809918880462646 | 3.9627663791179657 | 252.3489664365681 | optional baseline, not default
padded_agent_static fixed-K plugin | bucket_router_baseline | fp16 | None | None | None | 200 | 4 | False | True | 24064 | 0.8102 | 0.7704 | 0.5942 | 0.7249 | None | 2.200608015060425 | 3.955904394388199 | 252.78669560836423 | optional baseline, not default
dynamic_agent_dim multi-engine bucket fixed-K plugin | N1/N2 x bucket router | fp32 | None | None | None | 50 | 8 | False | True | 24064 | 0.8721 | 0.8534 | 0.7476 | 0.8244 | None | 4.985856056213379 | 6.470624357461929 | 154.54459179766775 | historical baseline
dynamic_agent_dim multi-engine bucket fixed-K plugin | N1/N2 x bucket router | fp16 | None | None | None | 50 | 8 | False | True | 24064 | 0.8723 | 0.8523 | 0.7322 | 0.8189 | None | 2.148128032684326 | 3.6728251725435257 | 272.2699701242448 | historical baseline
dynamic_agent_dim multi-engine bucket fixed-K plugin | N1/N2 x bucket router | int8 | None | None | calib50 | 50 | 8 | False | True | 24064 | 0.8076 | 0.7873 | 0.6412 | 0.7454 | None | 1.533087968826294 | 2.5472305715084076 | 392.58322791243216 | historical_calibration_split_confirmed=false
dynamic_agent_dim multi-engine bucket fixed-K plugin | N1/N2 x bucket router | int8 | None | None | calib200 | 50 | 8 | False | True | 24064 | 0.8257 | 0.8042 | 0.6327 | 0.7542 | None | 1.5381120443344116 | 2.5267787277698517 | 395.760811585827 | historical_calibration_split_confirmed=false
dynamic_agent_dim multi-engine bucket fixed-K plugin | N1/N2 x bucket router | int8_mixed_heads_fp16 | None | None | calib200 | 50 | 8 | False | True | 24064 | 0.825 | 0.8024 | 0.6422 | 0.7565 | None | 1.622912049293518 | 2.6664920151233673 | 375.024561981947 | historical_calibration_split_confirmed=false
dynamic_agent_dim multi-engine bucket fixed-K plugin | N1/N2 x bucket router | fp32 | None | None | None | 200 | 8 | False | True | 24064 | 0.8102 | 0.7712 | 0.598 | 0.7265 | None | 4.987904071807861 | 6.449433043599129 | 155.05238882857623 | historical baseline
dynamic_agent_dim multi-engine bucket fixed-K plugin | N1/N2 x bucket router | fp16 | None | None | None | 200 | 8 | False | True | 24064 | 0.8105 | 0.7704 | 0.5944 | 0.7251 | None | 2.1432321071624756 | 3.358054906129837 | 297.7914381848215 | historical baseline
dynamic_agent_dim multi-engine bucket fixed-K plugin | N1/N2 x bucket router | int8 | None | None | calib50 | 200 | 8 | False | True | 24064 | 0.6855 | 0.6441 | 0.4513 | 0.5936 | None | 1.5535039901733398 | 2.55439430475235 | 391.48223832927414 | historical_calibration_split_confirmed=false
dynamic_agent_dim multi-engine bucket fixed-K plugin | N1/N2 x bucket router | int8 | None | None | calib200 | 200 | 8 | False | True | 24064 | 0.7604 | 0.7154 | 0.5107 | 0.6622 | None | 1.5445760488510132 | 2.5555528700351715 | 391.3047590309635 | historical_calibration_split_confirmed=false
dynamic_agent_dim multi-engine bucket fixed-K plugin | N1/N2 x bucket router | int8_mixed_heads_fp16 | None | None | calib200 | 200 | 8 | False | True | 24064 | 0.7605 | 0.7157 | 0.5079 | 0.6614 | None | 1.6270079612731934 | 2.682555466890335 | 372.77887161797156 | historical_calibration_split_confirmed=false
dynamic_agent_single_engine_maxK | single TensorRT engine | fp32 | None | val | None | 50 | 1 | True | False | 24064 | 0.8721 | 0.8534 | 0.7464 | 0.824 | -0.0054 | 10.836799621582031 | 25.716926902532578 | 38.8848949094894 | calibration_split=n/a evaluation_split=val calibration_eval_overlap=false
dynamic_agent_single_engine_maxK | single TensorRT engine | fp32 | None | val | None | 200 | 1 | True | False | 24064 | 0.8102 | 0.7706 | 0.5965 | 0.7258 | -0.0012 | 10.877792358398438 | 25.74703097343445 | 38.83942972033517 | calibration_split=n/a evaluation_split=val calibration_eval_overlap=false
dynamic_agent_single_engine_maxK | single TensorRT engine | fp16 | None | val | None | 50 | 1 | True | False | 24064 | 0.8723 | 0.8523 | 0.7312 | 0.8186 | None | 1.8143359422683716 | 16.694746911525726 | 59.899081148071765 | calibration_split=n/a evaluation_split=val calibration_eval_overlap=false
dynamic_agent_single_engine_maxK | single TensorRT engine | fp16 | None | val | None | 200 | 1 | True | False | 24064 | 0.8103 | 0.7701 | 0.5935 | 0.7246 | None | 1.818943977355957 | 16.6485495865345 | 60.065292463002855 | calibration_split=n/a evaluation_split=val calibration_eval_overlap=false
dynamic_agent_single_engine_maxK | single TensorRT engine | int8 | train | val | train_calib50 | 50 | 1 | True | False | 24064 | 0.7058 | 0.6911 | 0.524 | 0.6403 | 0.1783 | 1.4941760301589966 | 16.380228102207184 | 61.04921089989298 | calibration_split=train evaluation_split=val calibration_eval_overlap=false
dynamic_agent_single_engine_maxK | single TensorRT engine | int8 | train | val | train_calib50 | 200 | 1 | True | False | 24064 | 0.6575 | 0.6199 | 0.4193 | 0.5656 | 0.159 | 1.4981440305709839 | 16.374271363019943 | 61.071419779839765 | calibration_split=train evaluation_split=val calibration_eval_overlap=false
dynamic_agent_single_engine_maxK | single TensorRT engine | int8 | train | val | train_calib200 | 50 | 1 | True | False | 24064 | 0.7843 | 0.7692 | 0.5992 | 0.7176 | 0.101 | 1.524224042892456 | 16.385603696107864 | 61.02918260115945 | calibration_split=train evaluation_split=val calibration_eval_overlap=false
dynamic_agent_single_engine_maxK | single TensorRT engine | int8 | train | val | train_calib200 | 200 | 1 | True | False | 24064 | 0.7255 | 0.6777 | 0.469 | 0.6241 | 0.1005 | 1.5275839567184448 | 16.406603157520294 | 60.95106893236642 | calibration_split=train evaluation_split=val calibration_eval_overlap=false

## Answers

- fastest_FP32_scheme: dynamic_agent_dim multi-engine bucket fixed-K plugin
- fastest_FP16_scheme: dynamic_agent_dim multi-engine bucket fixed-K plugin
- fastest_INT8_scheme: dynamic_agent_dim multi-engine bucket fixed-K plugin
- best_AP_scheme: dynamic_agent_dim multi-engine bucket fixed-K plugin
- single_engine_maxK_success: True
- single_engine_maxK_fp16_keeps_AP: True
- single_engine_maxK_fp16_forward_within_10pct: False
- single_engine_maxK_should_replace_multi_engine_bucket: False
- INT8_single_engine_has_acceptable_accuracy: False
- recommended_deployment_path: dynamic_agent_dim fixed-K bucket router PointPillarScatterTRT FP16
- new_scheme_calibration_split_train: True
- new_scheme_evaluation_split_val: True
