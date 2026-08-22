# OpenLLaMA 7B FFN kernel repeated same-runner A/B

Exact same runner, exact same verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV; 3 alternating-order pairs.

| Metric | Baseline mean ± SD | Candidate mean ± SD | Improvement |
|---|---:|---:|---:|
| FFN GEMM ms | 35772.87 ± 936.37 | 35444.66 ± 465.01 | 0.92% |
| Prefill ms | 46356.67 ± 1571.10 | 45815.67 ± 658.76 | 1.17% |
| Decode ms | 11411.70 ± 15.20 | 11443.77 ± 33.23 | -0.28% |
| QKV ms | 13464.49 ± 504.73 | 13362.77 ± 180.20 | 0.76% |

Positive improvement means lower elapsed time for the candidate kernel.

Per-pair FFN improvements: -0.57%, -2.08%, 5.21%
