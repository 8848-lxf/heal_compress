# Formal Structured Pruning Tools

The formal pruning surface is plan-first:

1. read or generate tracer `coupled_channel_groups.json`;
2. generate a schema-stable `prune_plan.json`;
3. run legality checks;
4. optionally execute the existing validated HEAL general pruner for physical
   pruning and checkpoint export;
5. evaluate the exported checkpoint with the existing full-val evaluator.

## CLIs

```bash
python -m pruning.planner.physical_prune_plan \
  --config <config.yaml> \
  --checkpoint <net_epoch_bestval_at17.pth> \
  --trace-report tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/tracer_reports/lidar_pyramid/coupled_channel_groups.json \
  --importance l1 \
  --target-prune-ratio 0.2 \
  --min-keep-ratio 0.5 \
  --output-dir tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/pruning_reports/lidar_pyramid_plan_only/

python -m pruning.export.export_pruned_model \
  --config <config.yaml> \
  --checkpoint <net_epoch_bestval_at17.pth> \
  --prune-plan <prune_plan.json> \
  --output-dir tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/pruned_models/lidar_pyramid/ \
  --execute-general-pruner

python -m pruning.eval.prune_and_eval \
  --config <config.yaml> \
  --checkpoint <net_epoch_bestval_at17.pth> \
  --prune-plan <prune_plan.json> \
  --split val \
  --output-dir tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/pruning_eval/lidar_pyramid/
```

Without `--execute-general-pruner`, export reports `skipped_requires_execute_general_pruner`
instead of pretending that physical surgery happened.
