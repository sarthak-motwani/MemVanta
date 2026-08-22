# OpenLLaMA 7B FFN kernel same-runner A/B

Exact same runner, exact same verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

| Metric | Baseline | Fused kernel | Improvement |
|---|---:|---:|---:|
| FFN GEMM ms | 36805.15 | 55685.41 | -51.30% |
| Prefill ms | 48204.30 | 67246.60 | -39.50% |
| Decode ms | 11429.20 | 11542.70 | -0.99% |
| QKV ms | 13989.61 | 14119.47 | -0.93% |

Positive improvement means lower elapsed time for the fused kernel.
