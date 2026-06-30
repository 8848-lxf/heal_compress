# Agent Export Strategy Comparison

- recommended_strategy: padded_agent_static
- int8_qdq_modelopt_plugin: not run

strategy | ONNX | semantics | ORT FP32 mAP | TRT FP32 mAP | TRT FP16 mAP | record_len=1 | record_len=2 | FP32 p50 | FP16 p50 | plugin | failure
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
dynamic_agent_dim | yes | yes | 0.8238 | 0.8239 | 0.8201 | True | True | 401.9781 | 375.0135 | no | None
padded_agent_static | yes | yes | 0.8243 | 0.8240 | 0.8185 | True | True | 160.4637 | 157.4891 | no | None
