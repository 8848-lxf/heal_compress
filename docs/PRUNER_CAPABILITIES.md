# Pruner capabilities

The formal pipeline is:

```text
immutable model + dependency units
  -> normalized importance
  -> global budgeted selection
  -> SamplingPruningRequest
  -> complete closure merge/legalization
  -> one frozen PhysicalPruningPlan
  -> one materialization transaction
  -> ledger + snapshot v2 + hashes + validation
```

`pruning.api` exposes model loading, scoring, selection, planning,
legalization, materialization, replay, physical artifacts and validation. The
selector never mutates a model. The executor supports dense Conv2d,
ConvTranspose2d, Linear, BatchNorm, regular grouped Conv and coupled depthwise
group removal. Original indices are frozen before any tensor is sliced.

Defaults are normalized first-order Taylor, global one-shot selection, dense
alignment 4, channels-per-group in `{4,8,16,32,64,128,256,512}`, and
`independent_group_topk`. Deblock/FPN/detection-head outputs remain fixed while
their input axes can follow upstream pruning.

HEAL config/checkpoint paths and model construction are caller supplied.
Integration calibration, dataset evaluation and GPU execution are not side
effects of any formal API.

