# OpenLLaMA 7B FFN kernel same-runner A/B

Exact same runner, exact same verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

| Metric | Baseline | Candidate | Improvement |
|---|---:|---:|---:|
| FFN GEMM ms | 39609.35 | 36585.80 | 7.63% |
| Prefill ms | 51351.50 | 46603.70 | 9.25% |
| Decode ms | 12785.80 | 12776.40 | 0.07% |
| QKV ms | 14989.77 | 13696.21 | 8.63% |

Positive improvement means lower elapsed time for the candidate kernel.
