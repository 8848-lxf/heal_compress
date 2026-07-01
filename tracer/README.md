# Formal Tracer Tools

The tracer package exports a formal report pipeline for HEAL/OpenCOOD
`lidar_pyramid` models:

1. load the model checkpoint;
2. build a tracing input from the real dataset when available;
3. run the generic forward tracer;
4. build a dependency graph;
5. build coupled channel groups;
6. write JSON and Markdown reports.

```bash
python -m tracer.export_trace_report \
  --config <config.yaml> \
  --checkpoint <net_epoch_bestval_at17.pth> \
  --output-dir tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/tracer_reports/lidar_pyramid/
```

Outputs:

- `trace_graph.json`
- `dependency_graph.json`
- `coupled_channel_groups.json`
- `trace_summary.json`
- `trace_summary.md`

If the checkpoint, dataset, or HEAL runtime is unavailable, the CLI writes the
same files with `success=false` and the concrete failure reason.
