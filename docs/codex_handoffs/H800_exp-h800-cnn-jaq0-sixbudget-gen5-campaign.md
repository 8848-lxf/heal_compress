# H800 CNN J_AQ=0 six-budget gen5 campaign

## 2026-07-29T11:20:00+08:00

- Branch: `exp/h800-cnn-jaq0-sixbudget-gen5-campaign`.
- Base: `cefe210243ed87955f094503e06d5ce3f30f9af5`.
- Scope: Pyramid, DiscoNet, F-Cooper, and (after user name confirmation) AttFusion.
- Frozen GA protocol: one seed (`0`), five evolution generations, population/offspring/survivor 64, Stage-2 new quota 5.
- Frozen evaluation protocol: Stage-2 Top-5 warmup 100 + fixed300; each generation winner warmup 200 + fixed500.
- Search fitness defaults to `J_struct + J_WQ + 0 * J_AQ`. Activation quantization remains enabled in deployed mixed-precision engines.
- The existing runner previously defaulted to `J_AQ=1` and the experiment branch restricted `J_AQ=0` to a single Pyramid `.05` run. This campaign removes that experiment-only restriction while retaining an explicit audit field in every output.
- No GPU search was launched before the default and protocol tests were updated.

---
