#!/usr/bin/env python3
"""Search entry point: GA-driven joint pruning + quantization search.

Usage:
    python scripts/run_search.py \
        --config configs/default_config.yaml \
        --checkpoint /path/to/model.pth \
        --heal-config /path/to/heal_config.yaml \
        --export-script /path/to/export_dynamic_onnx.py \
        --calibration-npz /path/to/frame_000000.npz \
        --output-dir ./search_results/

Flow:
    1. Load model (via export_dynamic_onnx.load_model_and_hypes)
    2. install_plugin_patches() for traceable LSS
    3. HealForwardWrapper enumerates multi-agent paths
    4. DependencyGraphBuilder constructs dependency graph
    5. CoupledChannelGroupBuilder generates coupled channel groups
    6. ImportanceEstimator computes gradient-based importance
    7. GeneticSearchEngine executes evolutionary search
    8. Saves best_subnet_config.json and search curves
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from heal_compress.utils.io_utils import load_yaml, save_json, ensure_dir
from heal_compress.utils.model_utils import import_export_module, load_model
from heal_compress.tracer.forward_wrapper import HealForwardWrapper
from heal_compress.tracer.dependency_graph import DependencyGraphBuilder
from heal_compress.tracer.coupled_channel_group import CoupledChannelGroupBuilder
from heal_compress.search.importance import ImportanceEstimator
from heal_compress.search.search_space import SearchSpaceEncoder
from heal_compress.search.proxy_objective import (
    ProxyObjectiveEvaluator, TaylorProxy, SQNRProxy,
    BOPSProxy, LatencyLUTProxy, SizeProxy, DeployPenalty,
)
from heal_compress.search.genetic_search import GeneticSearchEngine, SearchConstraints

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="HEAL model pruning + quantization search.")
    parser.add_argument("--config", default="configs/default_config.yaml",
                        help="Path to search config YAML.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to HEAL model checkpoint.")
    parser.add_argument("--heal-config", default=None,
                        help="Path to HEAL YAML config (optional).")
    parser.add_argument("--export-script", required=True,
                        help="Path to export_dynamic_onnx.py.")
    parser.add_argument("--calibration-npz", required=True,
                        help="Path to calibration .npz file.")
    parser.add_argument("--output-dir", default="./search_results",
                        help="Output directory.")
    parser.add_argument("--device", default=None,
                        help="Device (e.g. cuda:0).")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    out_dir = ensure_dir(args.output_dir)

    device = args.device or ("cuda:0" if __import__("torch").cuda.is_available() else "cpu")
    import torch

    # Step 1: Load model
    logger.info("Loading model...")
    export_mod = import_export_module(args.export_script)
    model, hypes = export_mod.load_model_and_hypes(
        Path(args.checkpoint), args.heal_config, torch.device(device)
    )
    model.eval()

    # Step 2: Plugin patches
    logger.info("Applying plugin patches...")
    export_mod.install_plugin_patches(model)

    # Step 3: Trace forward paths
    logger.info("Tracing forward paths...")
    max_agents = cfg.get("calibration", {}).get("max_agents", 2)
    modality = cfg.get("model", {}).get("modality")
    wrapper = HealForwardWrapper(
        model, export_mod, max_agents=max_agents, device=device, modality=modality,
    )
    trace_graph = wrapper.trace_all_paths(args.calibration_npz)

    # Step 4: Build dependency graph
    logger.info("Building dependency graph...")
    from heal_compress.utils.model_utils import get_protected_layer_names
    protected = get_protected_layer_names(model)
    dep_builder = DependencyGraphBuilder(trace_graph, model, protected)
    dep_builder.build()
    dep_builder.save(str(out_dir / "dependency_graph.json"))

    # Step 5: Generate coupled channel groups
    logger.info("Building coupled channel groups...")
    group_builder = CoupledChannelGroupBuilder(dep_builder.to_dict(), model)
    groups = group_builder.build()
    group_builder.save(
        groups,
        str(out_dir / "coupled_channel_groups.json"),
        csv_path=str(out_dir / "coupled_channel_groups.csv"),
        summary_path=str(out_dir / "group_dependency_summary.txt"),
    )

    # Step 6: Compute importance (if Taylor/Fisher)
    search_cfg = cfg.get("search", {})
    method = search_cfg.get("importance_method", "first_order_taylor")
    importance = ImportanceEstimator(model, groups, method=method)
    # Note: gradient computation requires calibration data with labels
    # This is a simplified version using L1/L2 norms
    scores = importance.estimate()
    save_json(scores, str(out_dir / "importance_scores.json"))

    # Step 7: Build search space
    quantizable = dep_builder.get_prunable_modules()
    weight_bits = search_cfg.get("weight_bits", ["FP16", "INT8", "INT4"])
    search_space = SearchSpaceEncoder(model, groups, quantizable, weight_bits)

    # Build group -> layers mapping
    group_to_layers = {}
    for g in groups:
        group_to_layers[g.group_id] = g.source_modules

    # Build proxy objectives
    lambdas = {
        "taylor": search_cfg.get("lambda1", 1.0),
        "sqnr": search_cfg.get("lambda2", 0.5),
        "bops": search_cfg.get("lambda3", 2.0),
        "latency": search_cfg.get("lambda4", 2.0),
        "size": search_cfg.get("lambda5", 1.0),
        "deploy": search_cfg.get("lambda6", 10.0),
    }

    evaluator = ProxyObjectiveEvaluator(
        taylor=TaylorProxy(model, group_to_layers),
        sqnr=SQNRProxy(model, search_space),
        bops=BOPSProxy(model, group_to_layers,
                       target_ratio=search_cfg.get("bops_ratio_max", 0.5)),
        latency=LatencyLUTProxy(
            search_cfg.get("latency_lut"),
            target_ms=search_cfg.get("latency_ms_max", 50.0),
        ),
        size=SizeProxy(model, group_to_layers,
                       target_ratio=search_cfg.get("size_ratio_max", 0.4)),
        deploy=DeployPenalty(search_space),
        lambdas=lambdas,
    )

    # Step 8: Run GA search
    constraints = SearchConstraints(
        protected_group_ids=search_space.protected_group_ids,
        weight_bits=weight_bits,
        min_channels=search_cfg.get("min_channels", 8),
        min_stage_ratio=search_cfg.get("min_stage_ratio", 0.25),
        channel_alignment=search_cfg.get("channel_alignment", 8),
    )

    engine = GeneticSearchEngine(
        group_ids=search_space.group_ids,
        layer_names=quantizable,
        evaluator=evaluator,
        constraints=constraints,
        pop_size=search_cfg.get("pop_size", 50),
        max_generations=search_cfg.get("max_generations", 100),
        elite_size=search_cfg.get("elite_size", 5),
        mutation_rate=search_cfg.get("mutation_rate", 0.1),
        patience=search_cfg.get("patience", 20),
        weight_bits=weight_bits,
        output_dir=str(out_dir),
    )

    best_cand, best_score = engine.run()
    logger.info(f"Search complete. Best score: {best_score:.6g}")
    logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
