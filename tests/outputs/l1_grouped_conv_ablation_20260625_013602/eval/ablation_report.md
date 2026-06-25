# L1 Grouped Conv Ablation Report

## 实验配置

- importance_mode: l1_norm
- selection_mode: constrained_global
- group_conv_prune_mode: keep_groups
- remove_groups: disabled

## 七个模型列表

- baseline_original: status=success
- shared_local_mean_p25: status=success
- shared_local_mean_p50: status=success
- shared_local_mean_p75: status=success
- independent_group_topk_p25: status=failed
- independent_group_topk_p50: status=failed
- independent_group_topk_p75: status=failed

## 目标剪枝率与实际剪枝率

- baseline_original: target=0.0 actual=0.0
- shared_local_mean_p25: target=0.25 actual=0.2669862397299366
- shared_local_mean_p50: target=0.5 actual=0.3012770296247377
- shared_local_mean_p75: target=0.75 actual=0.3012770296247377
- independent_group_topk_p25: target=0.25 actual=0.2669862397299366
- independent_group_topk_p50: target=0.5 actual=0.3012770296247377
- independent_group_topk_p75: target=0.75 actual=0.3012770296247377

## AP 与 Latency 对比

- baseline_original: mAP=0.73466 AP@0.5=0.779757 forward_p50=8.461 p90=8.666 p95=8.986 speedup=0.0
- shared_local_mean_p25: mAP=0.007441 AP@0.5=0.00747 forward_p50=7.956667 p90=8.1778 p95=8.3534 speedup=0.059607
- shared_local_mean_p50: mAP=0.007438 AP@0.5=0.007449 forward_p50=8.011 p90=8.3924 p95=8.7412 speedup=0.053185
- shared_local_mean_p75: mAP=0.007462 AP@0.5=0.007457 forward_p50=7.835333 p90=8.1874 p95=8.4992 speedup=0.073947
- independent_group_topk_p25: mAP=not_available AP@0.5=not_available forward_p50=not_available p90=not_available p95=not_available speedup=not_available
- independent_group_topk_p50: mAP=not_available AP@0.5=not_available forward_p50=not_available p90=not_available p95=not_available speedup=not_available
- independent_group_topk_p75: mAP=not_available AP@0.5=not_available forward_p50=not_available p90=not_available p95=not_available speedup=not_available

## 未达到目标剪枝率

- shared_local_mean_p50: target prune ratio was not reached under keep_groups grouped-conv constraints, 8-aligned per-group channel floors, residual-add protection, and disabled remove_groups
- shared_local_mean_p75: target prune ratio was not reached under keep_groups grouped-conv constraints, 8-aligned per-group channel floors, residual-add protection, and disabled remove_groups
- independent_group_topk_p50: target prune ratio was not reached under keep_groups grouped-conv constraints, 8-aligned per-group channel floors, residual-add protection, and disabled remove_groups
- independent_group_topk_p75: target prune ratio was not reached under keep_groups grouped-conv constraints, 8-aligned per-group channel floors, residual-add protection, and disabled remove_groups

## AP 崩塌检查

- no zero-mAP model detected in summary

## Forward Latency 变慢检查

- no forward_p50 slowdown detected in summary

## Grouped Conv Alignment Violations

- baseline_original: 
- shared_local_mean_p25: 0
- shared_local_mean_p50: 0
- shared_local_mean_p75: 0
- independent_group_topk_p25: 0
- independent_group_topk_p50: 0
- independent_group_topk_p75: 0

## 结论

根据 ablation_eval_summary.csv 中 mAP、forward_p50/p90/p95、结构合法性和 align violation 综合判断最终策略。
