# Migration Artifact Audit

Generated from the current server on 2026-07-12. Historical handoff claims are not treated as completion evidence.

## Current-server conclusion

- Repository: `/home/lixingfeng/UniAD_examine/heal_compress`
- Branch/starting HEAD: `feature/pruning-quant-toolkit-cleanupv1` / `94c4ab6`
- Missing historical run: `tests/outputs/lidar_pyramid_joint_search_final_20260712_060922`
- Current regenerated run: `tests/outputs/lidar_pyramid_joint_search_final_20260712_115848`
- Historical absence classification: `missing_due_to_server_migration`, not `pipeline_generation_failed`.
- Current result: four outer rounds, 20 real Stage-2 candidates, four round winners, shared 1789-frame final validation, and final winner are present and verified on this server.

## A–L status

| Category | Artifact | Exists | Current-server evidence | Reuse |
|---|---|---:|---|---:|
| A | Search/GA/proxy/Stage-2 source | yes | Four-round CLI execution and completed resume passed | yes |
| B | Joint-search and original model configs | yes | Paths and hashes recorded in JSON audit | yes |
| C | Original checkpoint | yes | SHA256 `d20a0107…27caca` | yes |
| D | Fisher statistics | yes | Regenerated and reused on current server | yes |
| E | Stage-1 generation records | yes | All four rounds; CUDA batched; scalar calls 0 | yes |
| F | Repaired Top-5 | yes | Four unique repaired Top-5 manifests | yes |
| G | Physical checkpoints | yes | Nonzero candidate strict reload and real forward verified; 20 Stage-2 candidate directories exist | yes |
| H | Pruned FP32 ONNX | yes | Checker, shape inference, canonical mapping and physical shapes verified | yes |
| I | Explicit Q/DQ ONNX | yes | BN-fold-aware real calibration, finite scales, Q/DQ and no-fallback checks | yes |
| J | TensorRT engines | yes | Isolated modelopt build, deserialize, context, plugin, bindings, structure and precision checks | yes |
| K | Evaluations | yes | All 20 candidates: 300/300, zero skips; final references/winners: 1789/1789, zero skips | yes |
| L | Layered caches | partial | Physical/ONNX index and evaluation cache verified; completed resume left 174 artifacts unchanged; dedicated Q/DQ/engine index hit not verified | partial |

## Final winner

Round-0 candidate `ddd6a0f2ccdc3f37c8361c1cb090ee193100d63ff626d0f93018cdda267d0132` is the final full-validation winner. It is all-keep/all-FP16.

- Full-val strict FP32 mAP: `0.7368540734`.
- Full-val strict FP16 p50: `2.5024222753 ms`.
- Winner mAP: `0.7362839016`.
- Winner p50: `2.5152699867 ms`.
- Winner F2: `0.2238336957`.

The all-keep outcome does not erase physical-pruning validation: multiple nonzero candidates completed physical pruning, strict reload, real forward, ONNX, explicit Q/DQ, TensorRT, and real evaluation. They lost under the AP-prioritized objective.

## Synchronization needs

No file is required from the original server to reproduce the current completion state. Synchronizing the absent historical run is optional and useful only for forensic comparison.

TensorRT must continue to use `/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118` and explicit `modelopt` activation with environment-pinned CUDA/GCC/G++.

Exact sizes, hashes, regeneration dependencies, and the remaining cache caveat are recorded in `migration_artifact_audit.json`.
