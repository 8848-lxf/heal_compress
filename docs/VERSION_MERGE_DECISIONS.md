# Version merge decisions

Version numbers were treated as evidence chronology, not as an automatic
selection rule.

| Family | Decision | Reason |
|---|---|---|
| v8.1 grouped independent top-k | merged | establishes independent local ranking and equal per-group count |
| v8.4-v8.5 grouped audits | test-only | useful legality evidence; ambiguous `safe_shapes` terminology is not retained |
| v9.4 global physical plan | merged | original-index one-shot request merge and closure |
| v9.7-v9.9 grouped surgery variants | compatibility/optional | input balancing and explicit remove-groups remain opt-in; experimental reblock variants are not defaults |
| v10.2 coupled-unit/search audits | merged | richer coupled/atomic metadata and budget repair evidence |
| v10.5-v10.7 stage-specific reblock programs | test-only | HEAL experiment drivers, not generic production APIs |
| v10.8 Taylor greedy | merged | latest verified dependency-member mean followed by scope-mean normalization |
| v10.9 parameter-budget round4 | merged | dense alignment 4, exact budget accounting and one-shot plan wiring |
| v11 mixed-precision LUT builder | split and merged | canonical precision mapping, profiles, Q/DQ and TRT command logic moved to dedicated modules; worker/experiment orchestration remains test-only |
| v12 combined LUT/physical structure | merged | snapshot v2 hard truth, Q/DQ root tracing, provenance and canonical mismatch fixes |
| temporary v12 diagnosis/recovery scripts | obsolete/test-only | incident-specific recovery; no production import |

The formal defaults do not use the older Torch-Pruning-like shared local
position policy. `shared_local_mean` is available only when explicitly
configured. `remove_groups` likewise requires explicit selection and legality
checks.

No v13/v14 branch was created. Every production capability has one canonical
implementation under `tracer`, `pruning` or `quantization`.

