# Throughput factorial screen — RSS gated

TinyStories 15M Q4_0; 32 configurations; 3 reps/config; production source unchanged.

Control drift: prefill -8.12%, decode -1.45%, RSS 0.23%

| Rank | Threads | Batch | Q8 | Bind | Prefill tok/s | Decode tok/s | Peak RSS MiB | Speed delta | RSS delta | Pass |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|:---:|
| 1 | 2 | 64 | 0 | close | 2111.20 | 665.83 | 26.30 | 0.00% | 0.00% | no |
| 2 | 2 | 32 | 0 | close | 2021.67 | 669.93 | 26.22 | 0.00% | 0.00% | no |
| 3 | 4 | 64 | 0 | spread | 2042.01 | 654.58 | 26.44 | 0.25% | 0.06% | no |
| 4 | 4 | 64 | 0 | close | 1987.87 | 658.18 | 26.42 | 0.00% | 0.00% | no |
| 5 | 2 | 32 | 0 | spread | 2016.36 | 650.94 | 26.21 | -2.21% | -0.04% | no |
| 6 | 2 | 16 | 0 | close | 1813.56 | 670.57 | 25.92 | 0.00% | 0.00% | no |
| 7 | 2 | 16 | 0 | spread | 1798.86 | 666.89 | 26.01 | -0.62% | 0.32% | no |
| 8 | 4 | 32 | 0 | close | 1866.63 | 656.10 | 26.26 | 0.00% | 0.00% | no |
| 9 | 4 | 32 | 0 | spread | 1731.69 | 654.43 | 26.05 | -2.17% | -0.79% | no |
| 10 | 4 | 16 | 0 | close | 1623.56 | 637.00 | 25.93 | 0.00% | 0.00% | no |
| 11 | 2 | 8 | 0 | close | 1430.98 | 668.32 | 25.85 | 0.00% | 0.00% | no |
| 12 | 2 | 8 | 0 | spread | 1407.64 | 669.57 | 25.79 | -0.40% | -0.23% | no |

Passing configurations: **0**.
Promotion gate: >=5% combined speed improvement, <=5% timing CV, decode regression <=2%, and peak RSS increase <=2%.

No configuration met both speed and memory gates. Keep current production configuration unchanged.
