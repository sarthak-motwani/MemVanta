# Q4 multi-row tile screen

Baseline plus 2 direct 2-row x 4-token Q4 candidates; 5 randomized reps; no activation packing; production source restored.

| Rank | Variant | Prefill tok/s | Decode tok/s | Gain | RSS delta | Prefill CV | Decode CV | Pass |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| 1 | rows2_all | 2065.98 | 630.31 | 1.78% | 0.17% | 1.50% | 0.79% | no |
| 2 | rows2_ffn | 1959.38 | 630.43 | 0.51% | 0.08% | 3.11% | 1.29% | no |
| 3 | baseline | 1868.40 | 636.07 | 0.00% | 0.00% | 6.35% | 0.32% | no |

Winner: **none**.
Gate: >=5% combined gain, <=2% RSS increase, <=5% timing CV baseline/candidate, <=2% decode regression.
