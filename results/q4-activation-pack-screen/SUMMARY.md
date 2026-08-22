# Q4 activation packing screen

Baseline plus 2 activation-layout candidates; 5 randomized reps each; production source restored.

| Rank | Variant | Prefill tok/s | Decode tok/s | Gain | RSS delta | Prefill CV | Decode CV | Pass |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| 1 | baseline | 1757.22 | 634.07 | 0.00% | 0.00% | 3.43% | 0.58% | no |
| 2 | pack_ffn | 1666.69 | 625.86 | -2.35% | 0.47% | 4.51% | 1.24% | no |
| 3 | pack_all | 1612.09 | 632.57 | -2.50% | 0.24% | 2.37% | 0.34% | no |

Winner: **none**.
Gate: >=5% combined gain, <=2% RSS increase, <=5% timing CV on baseline/candidate, <=2% decode regression.
