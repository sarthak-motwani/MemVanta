# Q4 multi-row strict repeat validation

9 randomized alternating reps; exact prior rows2_all kernel; production source restored.

Baseline: prefill 2392.00 tok/s, decode 794.96 tok/s, RSS 26.36 MiB, CV 3.75%/0.31%.
rows2_all: prefill 2592.69 tok/s (+8.39%), decode 797.79 tok/s (+0.36%), combined +2.25%, RSS -0.14%, CV 1.59%/0.53%.

Winner: **none**.
Gate: >=5% combined, >=8% prefill, >=-1% decode, <=2% RSS, <=3% CV on both baseline/candidate.
