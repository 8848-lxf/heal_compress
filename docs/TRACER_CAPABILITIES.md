# Tracer capabilities

The canonical entry point is `tracer.api.trace_model`. It combines one
representative forward (temporary module hooks only) with a Torch FX operation
graph. The result records module calls, repeated-call indices, tensor
producers/consumers, shapes, weighted calls, dependency edges, scopes, coupled
units, atomic units, protection and coverage.

Proven channel rules cover Conv2d, BatchNorm, Linear, ConvTranspose2d,
residual Add, channel Concat with offsets, Split, view/reshape/flatten,
permute/transpose, interpolate, grouped/depthwise convolution and
dependency-driven head/deblock/FPN inputs. A channel-changing operation with no
registered mapping raises `UnsupportedOperationError`.

`TraceResult`, `DependencyScope` (`PruningGroup` alias),
`CoupledChannelUnit`, `AtomicPruneUnit` and
`ConcreteCoupledPruningGroup` are schema-versioned dataclasses. IDs and
`trace_hash` use canonical JSON plus SHA256. `serialize_trace_result` writes
atomically and `load_trace_result` validates the schema.

One example input covers only the executed branch. Coverage explicitly records
that limitation. Transformer/attention pruning is not claimed as validated.

