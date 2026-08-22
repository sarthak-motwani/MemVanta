# OpenLLaMA 7B FFN kernel repeated same-runner A/B

Exact same runner, exact same verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV; 3 alternating-order pairs.

| Metric | Baseline mean ± SD | Candidate mean ± SD | Improvement |
|---|---:|---:|---:|
| FFN GEMM ms | 27999.41 ± 89.30 | 27908.42 ± 211.40 | 0.32% |
| Prefill ms | 35880.40 ± 71.38 | 35573.77 ± 254.73 | 0.85% |
| Decode ms | 9766.64 ± 22.89 | 9755.12 ± 15.32 | 0.12% |
| QKV ms | 10345.02 ± 23.01 | 10297.69 ± 59.39 | 0.46% |

Positive improvement means lower elapsed time for the candidate kernel.

Per-pair FFN improvements: -0.04%, 0.21%, 0.81%
