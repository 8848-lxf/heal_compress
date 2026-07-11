# ONNX canonical mapping

The formal identity chain is:

```text
PyTorch weighted module -> traced call -> weighted ONNX node
-> root initializer -> canonical ONNX name -> Q/DQ target
-> TensorRT layer name -> requested precision -> realized precision
```

Names follow:

```text
__canonical__{normalized_module_path}__{onnx_op_type}__call{call_index:05d}
```

Long names use a deterministic SHA256 suffix. The origin map preserves original
name, canonical name, graph index, module path/type, call index, initializer,
groups and weight shape. Repeated calls are paired by trace call order and ONNX
graph order only when candidate counts match exactly.

Conv, ConvTranspose, Gemm and weighted MatMul are supported. A functional
MatMul is never invented as a module. Unresolved initializers, overlapping
candidates, non-one-to-one mappings and unresolved name collisions raise
`CanonicalMappingError`/`AmbiguousCanonicalMappingError`.

