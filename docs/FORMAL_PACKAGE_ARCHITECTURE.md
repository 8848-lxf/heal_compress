# Formal package architecture

The release surface is split into three packages with a one-way dependency
direction:

```text
tracer
  runtime/static graph -> dependency scopes -> coupled units -> atomic units
       |
       v
pruning
  importance -> global selector -> one-shot plan -> materialization -> artifacts
       |
       v
quantization
  ONNX/origin map -> canonical precision -> explicit Q/DQ -> TensorRT checks
```

`tracer` owns graph identities and directional protection metadata. `pruning`
owns all model mutation and physical structure truth. `quantization` consumes a
physical snapshot and canonical origin evidence; it never treats a sampling
estimate as a deployed shape.

The public modules are `tracer.api`, `pruning.api`, and `quantization.api`.
Configuration dataclasses contain only serializable values and support
dictionary/YAML round trips. Heavy optional dependencies (`onnx`, TensorRT and
HEAL adapters) are imported only inside the operation that needs them, so the
packages remain importable in a CPU-only release environment.

Default policy chain:

```text
first_order_taylor
  + coupled_dependency_mean_then_scope_mean_v1
  + global normalized ranking
  + global budget
  + dense alignment 4
  + allowed grouped channels per group {4,8,16,32,64,128,256,512}
  + independent_group_topk
  -> one frozen PhysicalPruningPlan
  -> one materialization transaction
  -> snapshot v2 + ledger + hashes
  -> FP16/INT8 explicit Q/DQ deployment metadata
```

Transformer/attention support is not advertised as validated. An unresolved
channel-changing operation, weighted ONNX node or initializer trace is a hard
error under the default fail-closed configuration.

