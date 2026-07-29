# H800 CoBEVT J_AQ=0 six-budget gen5 campaign

## 2026-07-29T11:50:00+08:00

- Branch: `fix/h800-cobevt-jaq0-sixbudget-gen5-campaign`.
- Base: `656266ee5307691b01184c1e04de8feb095a12a5`, which closes the CoBEVT functional precision and ranking provenance paths.
- Formal GA is one seed and exactly five evolution generations, with fixed300/warmup100 Top-5 screening and fixed500/warmup200 generation-winner validation.
- The main Greedy/GA path defaults to `J_AQ=0`; activation statistics are retained diagnostically and real activation quantization remains enabled in deployment.
- The process must own exactly its declared physical GPU through `CUDA_VISIBLE_DEVICES`.
- Historical fixed500 requested/realized failures were produced before the closure commit; this campaign revalidates the same boundary fail-closed before accepting any GA result.

---
