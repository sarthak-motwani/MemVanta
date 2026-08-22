# Q4 24-way structural throughput screen

Exact TinyStories 15M Q4_0; 24 variants x 3 randomized reps; production source restored.

| Rank | Variant | Prefill tok/s | Decode tok/s | Combined gain | RSS delta | Prefill CV | Decode CV | Pass |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| 1 | base__unroll | 1829.67 | 648.59 | 7.42% | 0.40% | 0.80% | 0.29% | no |
| 2 | batchlut__unroll | 1817.27 | 649.53 | 7.35% | 0.16% | 2.39% | 0.25% | no |
| 3 | base__unroll32 | 1826.76 | 647.90 | 7.29% | -0.02% | 1.84% | 0.81% | no |
| 4 | batchlut__unroll32 | 1830.55 | 647.35 | 7.28% | 0.34% | 0.94% | 0.44% | no |
| 5 | batchlut__mathsafe | 1828.74 | 646.33 | 7.13% | 0.03% | 0.43% | 0.42% | no |
| 6 | dotlut__unroll32 | 1845.82 | 643.29 | 7.02% | 0.06% | 1.31% | 0.95% | no |
| 7 | batchlut__align32 | 1831.33 | 630.03 | 5.16% | -0.11% | 1.48% | 0.54% | no |
| 8 | dotlut__base | 1855.07 | 624.69 | 4.84% | -0.04% | 0.98% | 0.58% | no |
| 9 | bothlut__align64 | 1830.53 | 627.16 | 4.79% | -0.14% | 0.41% | 0.52% | no |
| 10 | bothlut__base | 1836.25 | 621.95 | 4.22% | -0.36% | 1.27% | 1.38% | no |
| 11 | base__mathsafe | 1816.27 | 622.12 | 3.96% | -0.12% | 0.55% | 5.61% | no |
| 12 | dotlut__align32 | 1840.42 | 617.02 | 3.66% | -0.18% | 2.24% | 2.12% | no |
| 13 | bothlut__unroll32 | 1822.12 | 615.48 | 3.21% | 0.32% | 1.08% | 8.05% | no |
| 14 | bothlut__mathsafe | 1834.93 | 611.37 | 2.88% | -0.15% | 0.97% | 11.50% | no |
| 15 | dotlut__unroll | 1851.37 | 603.45 | 2.10% | 0.25% | 1.17% | 10.12% | no |
| 16 | base__base | 1810.92 | 591.32 | 0.00% | 0.00% | 0.98% | 10.61% | no |
| 17 | bothlut__unroll | 1841.46 | 588.07 | -0.01% | -0.08% | 1.82% | 14.71% | no |
| 18 | batchlut__align64 | 1815.00 | 588.20 | -0.34% | 0.27% | 0.95% | 11.02% | no |
| 19 | base__align32 | 1837.04 | 581.10 | -0.97% | -0.10% | 0.85% | 12.01% | no |
| 20 | batchlut__base | 1854.16 | 575.16 | -1.52% | 0.15% | 1.77% | 9.89% | no |
| 21 | bothlut__align32 | 1801.53 | 574.36 | -2.30% | -0.25% | 2.88% | 12.14% | no |
| 22 | dotlut__align64 | 1857.50 | 545.77 | -5.37% | -0.27% | 0.27% | 10.46% | no |
| 23 | base__align64 | 1836.02 | 545.22 | -5.69% | -0.09% | 0.75% | 10.27% | no |
| 24 | dotlut__mathsafe | 1841.48 | 483.30 | -14.12% | -0.25% | 2.40% | 43.16% | no |

Winner: **none**.
Promotion requires >=8% combined speed gain, <=2% RSS increase, <=5% timing CV, and <=2% decode regression.
