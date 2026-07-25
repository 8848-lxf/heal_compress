#!/usr/bin/env python3
"""Deterministic V2X-ViT R_BOPS=0.30 Greedy with cached gate+WQ+AQ risk."""
from __future__ import annotations
import argparse, csv, json, random, subprocess, sys
from dataclasses import replace
from pathlib import Path
from typing import Any
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
from scripts.audit_heal_transformer_search_models import MODEL_SPECS, _load, _sha256
from scripts.run_v2xvit_greedy005_full import _batch_content_hash, _build_full_space, _formal_space, _baseline_candidate
from scripts.run_v2xvit_greedy005_weight_only_abs import _type_coverage_forward
from scripts.smoke_transformer_unified_search import _multi_agent_validation_batch
from search.canonicalization import canonicalize_candidate
from search.hashing import candidate_hash, canonical_json_hash
from search.greedy.weight_only_abs import run_weight_only_abs_greedy
from search.proxy.joint_weight_taylor import JointWeightTaylorProxy
from search.proxy.conservative_gate_activation_taylor import (
    FunctionalGateTaylorProxy, build_activation_units, collect_activation_taylor_cache,
    collect_functional_gate_scores, rerank_domains_by_gate_scores,
)
from search.proxy.joint_weight_activation_taylor import taylor_units_from_transformer_precision

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str)+"\n", encoding="utf-8")
def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise RuntimeError(f"refusing_to_overwrite:{path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["step"], extrasaction="ignore"); w.writeheader(); w.writerows(rows)
def git(*args: str) -> str:
    return subprocess.run(["git","-C",str(REPO),*args], check=True, text=True, capture_output=True).stdout.strip()

def run(args: argparse.Namespace) -> int:
    if torch.cuda.device_count()!=1: raise RuntimeError(f"greedy030_requires_one_visible_gpu:{torch.cuda.device_count()}")
    device=torch.device("cuda:0"); torch.cuda.set_device(device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    model, adapter, hypes, _ = _load("v2xvit", device)
    batch, dataset_index, agent_count = _multi_agent_validation_batch(adapter, hypes, device)
    batch_hash=_batch_content_hash(batch)
    calibration_hash=canonical_json_hash({"model":"lidar_v2xvit","config_sha256":_sha256(MODEL_SPECS["v2xvit"]["config"]),"checkpoint_sha256":_sha256(MODEL_SPECS["v2xvit"]["checkpoint"]),"split":"validation","dataset_indices":[dataset_index],"agent_count":agent_count,"sample_count":1,"seed":args.seed,"batch_content_sha256":batch_hash,"hmsa_type_coverage":"type0_type1_pre_forward_hook_v1"})
    identity=_build_full_space(model, adapter, hypes, batch)
    formal=_formal_space(model, adapter, hypes, batch, identity, calibration_hash, fisher_forward_fn=lambda m,b: _type_coverage_forward(adapter,m,b))
    print("[greedy030] formal_space_ready", flush=True)
    space=formal["space"]
    # Functional gate statistics are collected once, then used to rebuild only
    # the nested ranking/width map.  Tracer dependency members remain unchanged.
    gate_scores, gate_mapping=collect_functional_gate_scores(model, space.pruning_domains, forward_fn=lambda m,b: _type_coverage_forward(adapter,m,b), loss_fn=adapter.compute_task_loss, batch=batch)
    print("[greedy030] gate_scores_ready", flush=True)
    reranked=rerank_domains_by_gate_scores(space.pruning_domains, gate_scores)
    space=replace(space, pruning_domains=reranked, pruning_unit_ids=[u for d in reranked for u in d.ordered_unit_ids])
    baseline=_baseline_candidate(space)
    base_bops=formal["bops"].evaluate_breakdown(canonicalize_candidate(baseline,space))
    if abs(float(base_bops["R_bops_vs_fp32"])-1.0)>1e-10: raise RuntimeError("greedy030_baseline_bops_invalid")
    weight_proxy=JointWeightTaylorProxy(model, statistics=formal["fisher"], unit_to_parameter_slices=formal["slices"], strict=True)
    structure_proxy=FunctionalGateTaylorProxy(gate_scores)
    transformer_units=taylor_units_from_transformer_precision(model, formal["components"].precision_units, active_module_paths=formal["active_paths"])
    activation_units, group_to_units=build_activation_units(model,space,transformer_units)
    activation_cache=collect_activation_taylor_cache(model, activation_units, group_to_units, forward_fn=lambda m,b: _type_coverage_forward(adapter,m,b), loss_fn=adapter.compute_task_loss, batch=batch)
    print("[greedy030] activation_cache_ready", flush=True)
    result=run_weight_only_abs_greedy(space, weight_proxy=weight_proxy, structure_proxy=structure_proxy, activation_cache=activation_cache, activation_taylor_weight=1.0, bops_evaluator=formal["bops"].evaluate_breakdown, size_evaluator=formal["size"].evaluate_breakdown, target=0.30, tolerance_abs=0.005, maximum_steps=args.maximum_steps)
    print("[greedy030] greedy_done", flush=True)
    root=args.output_root.resolve(); write_csv(root/"greedy_trace.csv",result["trace"])
    winner=result["winner_candidate"]; phenotype=result["winner_phenotype"]
    payload={"candidate_hash":candidate_hash(phenotype,space),"genotype":winner.to_dict(),"phenotype":phenotype.to_dict(),"metrics":result["winner_metrics"],"bops":result["winner_bops_breakdown"],"size":result["winner_size_breakdown"],"diagnostic_control":False}
    write_json(root/"winner/v2xvit_greedy030_winner.json",payload)
    write_json(root/"search/v2xvit_greedy030_search_manifest.json",{"model":"V2X-ViT","target_bops_retention":0.30,"tolerance_abs":0.005,"budget_band":[0.295,0.305],"budget_reached":result["budget_reached"],"budget_band_candidate_count":result["budget_band_candidate_count"],"selected_step_count":result["selected_step_count"],"visited_action_count":result["visited_action_count"],"termination_reason":result["termination_reason"],"winner":payload,"activation_taylor_used_for_fitness":True,"structure_proxy":"functional_gate_output_taylor","legacy_coupled_weight_taylor_used_for_fitness":False,"search_loop_runtime_audit":{k:result[k] for k in ("search_loop_forward_calls","search_loop_backward_calls","search_loop_physical_exports","search_loop_onnx_exports","search_loop_trt_builds")}})
    write_json(root/"proxy/activation_taylor_mapping.json",{"units":[dict(x) for x in activation_cache.mapping],"group_to_units":{k:list(v) for k,v in group_to_units.items()},"used_for_fitness":True,"formula":"sum_elementwise(abs(g_A*delta_A)+0.5*abs(h_A*delta_A^2))"})
    write_json(root/"proxy/structural_gate_mapping.json",{"domains":[{"domain_id":k,"unit_scores":dict(v.unit_scores),"semantic_root_tensor":v.semantic_root_tensor,"gate_tensor":v.gate_tensor,"physical_dependencies":list(v.physical_dependencies),"family":v.family} for k,v in gate_scores.items()],"mapping":gate_mapping,"legacy_coupled_weight_taylor_used_for_fitness":False})
    write_json(root/"reports/greedy030_winner.json",payload)
    write_json(root/"reports/activation_taylor_audit.json",{"activation_taylor_used_for_fitness":True,"weight":1.0,"joint_taylor_used_for_fitness":False,"cross_residual_used_for_fitness":False,"elementwise_abs_before_reduction":True,"activation_units":len(activation_units)})
    write_json(root/"reports/search_loop_runtime_audit.json",{k:result[k] for k in ("search_loop_forward_calls","search_loop_backward_calls","search_loop_physical_exports","search_loop_onnx_exports","search_loop_trt_builds")})
    write_json(root/"reports/input_provenance.json",{"branch":git("branch","--show-current"),"commit":git("rev-parse","HEAD"),"checkpoint":str(MODEL_SPECS["v2xvit"]["checkpoint"]),"checkpoint_sha256":_sha256(MODEL_SPECS["v2xvit"]["checkpoint"]),"calibration_manifest_hash":calibration_hash,"dataset_index":dataset_index,"agent_count":agent_count,"gpu_visible_index":0})
    print(json.dumps({"status":"ok","winner":payload["candidate_hash"],"budget_reached":result["budget_reached"],"steps":result["selected_step_count"],"budget_candidates":result["budget_band_candidate_count"]},sort_keys=True))
    return 0

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--seed",type=int,default=20260725); p.add_argument("--maximum-steps",type=int,default=10000); return run(p.parse_args())
if __name__=="__main__": raise SystemExit(main())
