# H800 V2X-ViT J_AQ=0 six-budget gen5 campaign

## 2026-07-29T11:35:00+08:00

- Branch: `exp/h800-v2xvit-jaq0-sixbudget-gen5-campaign`.
- Base: `5826b594ba868afa8f607e4a78ca59c2046c1938`, including HGT/AV/merge precision closure and the two-tier 300/500 evaluation protocol.
- The formal contract is exactly five evolution generations; generation zero remains initialization only.
- `J_AQ` defaults to zero in proxy, Greedy, and GA fitness, while mixed deployment keeps real activation quantization and calibration.
- Cross-GPU Stage-2 is fail-closed. The physical GPU argument must exactly match `CUDA_VISIBLE_DEVICES`, and every candidate lifecycle stays on the assigned card.
- The prior `.05` result used ten generations and is provenance-only; this campaign rebuilds `.05` under the five-generation contract.

---
