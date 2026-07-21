# CoBEVT Attention Operand and Accumulator Precision Audit Design

## Objective

Independently control and audit operand, multiplication, accumulator, and output precision for CoBEVT QK and AV matrix multiplications on RTX 4090 with TensorRT 10.9.0.34. Results are admissible only when requested and realized phenotypes are separated and accumulator claims satisfy the configured evidence level.

## Evidence Architecture

The audit has three non-interchangeable layers:

1. Installed TensorRT capability: local Python APIs, headers, libraries, samples, release notes, and EngineInspector evidence determine what native TensorRT 10.9 can express and prove.
2. Exact oracle: cuBLASLt is the primary implementation and CUTLASS is the fallback. Its descriptors/templates explicitly bind operand, compute, accumulator, scale, and output types and provide Level-A evidence.
3. Deployment realization: native TensorRT micro-engines are tried first. If native TensorRT cannot implement or directly prove F16A32, a separately labelled cuBLASLt/CUTLASS TensorRT plugin is used as a `plugin_oracle`, never reported as a native tactic.

The phenotype schema contains `storage_precision`, `operand_precision`, `multiplication_precision`, `accumulator_precision`, `output_precision`, `requested_precision`, `realized_precision`, `evidence_source`, `evidence_level`, and Cast materialization state. F32A32 after a materialized Cast, INT8 projection followed by FP32 QK, and unknown-accumulator fused MHA are deliberately distinct.

## Data and Experiments

Fresh hooks capture Q, K, V, scaled Q, logits, masked logits, probability, AV, output-projection input, and residual update for all six CoBEVT Attention blocks over the accepted smoke10 manifest. Exact oracles and TensorRT micro-engines consume those tensors and accumulator-discriminative synthetic probes.

Micro-engine profiles cover QK M0-M5 and AV N0-N5 for all six real shapes. Full-model profiles are gated: F3 is the fresh reference; QK F16A32, F16A16, native I8A32I, AV F16A32/F16A16/I8A32I and joint profiles proceed only when build, runtime, finite output, and phenotype evidence pass. No unsupported or unknown-accumulator profile is promoted.

## Safety and Search Contract

Numerical thresholds are calibrated from accepted F3 noise and historical R1 failure distributions. Full-model fixed500 classification uses the specified `0.003` safe and `0.010` unsafe mAP boundaries. An independent precision gene requires Level-A accumulator evidence, exact requested/realized identity, fixed500 safety, formal isolated latency gain, zero skips, and zero precision conflicts. Level-B evidence remains experimental; Level-C/unknown is not searchable.

All CUDA compilation uses `/home/lixingfeng/anaconda3/envs/modelopt/bin/nvcc` with SM89. The implementation fails closed on system nvcc, stale caches, Pyramid paths, silent precision fallback, and attempts to relabel plugin results as native TensorRT.

## Deliverables

The run writes machine-readable capability, capture, oracle, micro-engine, Cast, blockwise numerical, full-model, safety-boundary, latency, and search-contract artifacts under a fresh timestamped directory. Repository changes include focused precision-contract, oracle, micro-engine, capture, full-model, reporting, optional plugin, tests, a handoff report, and reproducible commands.
