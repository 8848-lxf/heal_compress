# Grouped convolution pruning

The discrete legality set is explicitly defined as **channels per group**:

```text
allowed_channels_per_group = (4, 8, 16, 32, 64, 128, 256, 512)
```

Total input/output channels must remain divisible by `groups`, and every group
must retain the same count. The number of groups is not itself required to be a
multiple of the channel alignment.

The default `independent_group_topk` ranks each group's local scores
independently. Groups delete the same number of channels but can use different
local positions. Decisions record per-group raw and normalized scores,
`group_keep_map`, `group_prune_map`, selected locals, final width, repairs and
legality. Materialization verifies that expanded maps equal the frozen absolute
keep/prune indices. Replay without an exact map raises
`MissingGroupKeepMapError`.

`shared_local_mean` is a compatibility mode that shares local positions across
groups. `remove_groups` is opt-in and must remove complete group blocks with a
legal group update. Depthwise convolution is separate: input/output keep
indices must be identical and the resulting `groups`, input channels and output
channels are updated together.

