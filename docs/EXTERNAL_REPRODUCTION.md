# External Reproduction Guide

Independent reproduction is the most useful next validation step for MemVanta.

## What to reproduce

Use the same GGUF model for MemVanta and the pinned comparison runtime, then measure:

1. peak resident memory across repeated runs;
2. prompt-processing throughput;
3. token-generation throughput; and
4. where possible, success/failure under enforced Linux cgroup-v2 memory ceilings with swap disabled.

## Minimum evidence to publish

Please include:

- MemVanta commit SHA
- comparison-runtime commit SHA
- model filename and SHA-256
- CPU model
- operating system and kernel
- total RAM and swap state
- compiler/build information
- complete command lines
- raw outputs for every run
- a short machine-readable summary (CSV or JSON is ideal)

Do not publish only the best run. Keep warm-up runs separate from measured runs and retain failed/OOM runs in constrained-memory sweeps.

## Interpretation

MemVanta's optimization target is lower memory use under constrained CPU inference. It is acceptable—and expected in current releases—for a comparison runtime to deliver substantially higher throughput. A reproduction is useful whether it confirms, narrows, or contradicts the repository's existing memory results.

## Reporting a reproduction

Open a GitHub issue with:

- hardware and operating-system details;
- links or attachments containing raw evidence;
- the exact model hash and runtime commits;
- whether the result confirms or differs from the published benchmark.

If a reproduction exposes a methodology problem, please report it. Reproducibility and correction are more important than preserving a headline number.
