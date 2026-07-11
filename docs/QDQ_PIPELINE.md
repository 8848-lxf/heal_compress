# Explicit Q/DQ pipeline

`insert_explicit_qdq` targets exact canonical INT8 entries. It inserts
QuantizeLinear/DequantizeLinear pairs for activation input, weight and output,
using caller-supplied positive calibration scales and an explicit zero point.
The report records requested, inserted and fallback layers, node names, scales,
calibration metadata, policy version and output hash.

`trace_qdq_root_initializer` walks backward through QuantizeLinear,
DequantizeLinear, Cast, Identity, Transpose, Reshape, Squeeze and Unsqueeze. It
must end at one real initializer. Validation compares this root with snapshot
v2 using exact Conv/ConvTranspose layouts and Gemm/MatMul transpose/transB
semantics. Failure to reach an initializer raises `QDQValidationError`.

