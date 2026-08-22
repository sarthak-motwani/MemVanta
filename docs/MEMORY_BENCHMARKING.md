# MemVanta Memory Benchmarking Protocol

MemVanta is optimized for **memory-constrained CPU inference**. Throughput is a secondary metric and is reported for transparency, not as the primary optimization target.

## Primary claim

The project evaluates whether MemVanta can execute the same quantized GGUF workload with lower resident-memory usage and under tighter enforced memory ceilings than a pinned comparison runtime.

## Benchmark principles

1. **Exact same model artifact.** MemVanta and the comparison runtime must use the identical GGUF file. Record its filename, byte size, and SHA-256 digest.
2. **Pinned comparison runtime.** Record the exact llama.cpp commit SHA and build configuration used for every comparison.
3. **Matched workload.** Keep CPU-only execution, thread count, prompt length, generated-token count, context size, batch size, and KV-cache precision identical whenever both runtimes support the setting.
4. **Warm-up plus repeated runs.** Use at least one warm-up and five measured runs for peak-RSS comparisons. Publish every raw run rather than only the aggregate.
5. **Memory-pressure tests.** On Linux, use cgroup v2 with swap disabled. Record every tested `MemoryMax` ceiling and whether the process completed or was OOM-killed.
6. **No minimum-RAM overclaim.** A lowest successful tested ceiling is not an exact minimum physical-RAM requirement. Report it only as the lowest successful point in the tested sweep.
7. **Report the trade-off.** Always report throughput beside memory results. Do not describe MemVanta as faster when the comparison runtime has higher token throughput.
8. **Preserve evidence.** Commit machine-readable summaries, raw logs, environment metadata, and the commands used to produce the result.

## Required metadata

Each published benchmark should include:

- date and MemVanta commit SHA
- operating system and kernel
- CPU model and logical/physical core counts
- total system RAM
- compiler and build type
- model filename, size, quantization, and SHA-256
- comparison-runtime commit SHA
- thread count
- context and batch sizes
- prompt and generation token counts
- KV-cache format
- warm-up count and measured repetitions
- swap state
- cgroup configuration for constrained-memory runs

## Primary metrics

### Peak RSS

Report peak resident set size for every measured process run and summarize mean, standard deviation, median, minimum, and maximum. The headline comparison should use the same statistic for both runtimes.

### Lowest successful tested memory ceiling

For cgroup-v2 sweeps, report the tested ceilings in descending or ascending order and the outcome at each ceiling. The primary result is the lowest tested ceiling at which the workload completed. Also identify the highest tested ceiling that failed when one exists.

### Throughput

Report prompt-processing and token-generation throughput as secondary metrics so readers can see the memory/performance trade-off directly.

## Recommended validation matrix

For each major release, prioritize independent repetitions over adding many loosely controlled models.

- at least two 7B/8B-class GGUF models from different model families
- at least two physical CPU systems when available
- repeated peak-RSS A/B runs
- cgroup-v2 memory-pressure sweep on Linux
- F16 KV plus one lower-memory KV-cache mode where supported
- one fixed workload retained across releases for longitudinal comparison

## Result wording

Preferred:

> On the tested workload and hardware, MemVanta used X% lower peak RSS than pinned llama.cpp while remaining slower in token throughput.

Preferred for constrained memory:

> MemVanta completed at a tested memory ceiling of X MiB while pinned llama.cpp was OOM-killed at the same ceiling. This is a result over the tested sweep, not an exact minimum physical-RAM requirement.

Avoid:

- “MemVanta needs X GiB of RAM” unless independently established under a clearly defined environment.
- “X% more memory efficient” without defining the exact memory metric.
- universal claims derived from one model or machine.
- hiding throughput regressions.

## External reproduction

Independent reproduction is especially valuable. Reproducers should publish the exact model hash, both runtime commits, machine metadata, commands, raw outputs, and any deviations from the reference protocol. Results that disagree with the repository's published measurements are welcome and should be retained rather than filtered out.
