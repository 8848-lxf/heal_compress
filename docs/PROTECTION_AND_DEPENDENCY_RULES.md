# Protection and dependency rules

Protection is directional. Every policy records:

- `root_pruning_allowed`;
- `input_dependency_pruning_allowed`;
- `output_dependency_pruning_allowed`;
- `fixed_output_contract`;
- `protection_reason`.

Deblock, FPN and final classification/regression/direction head outputs are
fixed and cannot be selected as pruning roots. Their input axes remain eligible
for dependency-driven pruning. ConvTranspose2d input slicing uses weight axis 0
for layout `[C_in,C_out/groups,kH,kW]`; its output axis stays unchanged.

Residual Add creates a common channel identity across branches. Channel Concat
records branch offsets and maps downstream input indices in concatenated space.
Fixed PFN/scatter interfaces remain conservative unless an adapter proves the
mapping. An unresolved operation, protected-output request or incomplete
directional propagation is rejected rather than silently skipped.

