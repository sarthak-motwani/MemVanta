# OpenLLaMA 7B FFN kernel repeated same-runner A/B

Exact same runner, exact same verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV; 3 alternating-order pairs.

| Metric | Baseline mean ± SD | Candidate mean ± SD | Improvement |
|---|---:|---:|---:|
| FFN GEMM ms | 37137.79 ± 365.79 | 35374.08 ± 743.86 | 4.75% |
| Prefill ms | 48613.07 ± 413.46 | 45654.50 ± 703.72 | 6.09% |
| Decode ms | 11402.07 ± 9.75 | 11390.97 ± 12.97 | 0.10% |
| QKV ms | 14113.02 ± 21.40 | 13298.83 ± 45.51 | 5.77% |

Positive improvement means lower elapsed time for the candidate kernel.

Per-pair FFN improvements: 4.24%, 2.49%, 7.49%
