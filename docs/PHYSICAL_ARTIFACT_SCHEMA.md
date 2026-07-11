# Physical artifact schema

Three artifacts have distinct meanings:

- `sampling_structure_request.json`: requested/search structure only;
- `physical_pruning_application_ledger.json`: terminal result for every
  request, including indices, repair/merge/skip reason, alignment, protection
  and closure;
- `physical_structure_snapshot_v2.json`: full scan of the live physical model
  and state dict, including unchanged weighted modules.

`physical_hash_v2.json` contains `structure_hash_v2`, `shape_hash_v2`, snapshot
hash, schema version, and optional model/config hashes. Canonical JSON
(`sort_keys=True`, compact separators, ASCII) and SHA256 are used consistently.

Snapshot rows record module type, logical channel/features, groups,
kernel/stride/padding/dilation/output padding, weight/bias shape, parameter
counts, state-dict keys and protection policy. Conv2d weights use
`[C_out,C_in/groups,kH,kW]`; ConvTranspose2d uses
`[C_in,C_out/groups,kH,kW]`.

Legacy `before_after_shapes` is a sampling estimate and
`module_channel_before_after` is only a sparse physical delta. Neither can
replace snapshot v2 as physical truth.

