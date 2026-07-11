# Quantization deployment capabilities

The implemented deployment mode is TensorRT layer-level **FP16 + INT8 explicit
Q/DQ**. It does not implement INT4, arbitrary bit width or weight-only
quantization.

`quantization.api` exposes signal-maxK ONNX export wrappers, origin mapping,
canonical names, four precision profiles, Q/DQ insertion/root validation,
TensorRT command/build/runtime interfaces, structure/precision/provenance
checks, metrics and latency summaries. ONNX and TensorRT are imported lazily.

Profiles request INT8 for 0%, 20%, 50% and 80% of deterministic precision
groups. Unsupported/grouped cases fall back explicitly to FP16 with a reason.
Every stage records hashes and policy versions. TensorRT build and runtime occur
only when their explicit functions are called.

