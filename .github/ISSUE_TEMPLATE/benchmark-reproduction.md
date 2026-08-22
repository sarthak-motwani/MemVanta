---
name: Benchmark reproduction
description: Report an independent MemVanta memory benchmark reproduction
title: "[reproduction] "
labels: []
assignees: []
---

## Result summary

Briefly state whether your result confirms, narrows, or contradicts a published MemVanta memory result.

## Hardware and operating system

- CPU:
- Physical cores / logical CPUs:
- Total RAM:
- Operating system:
- Kernel:
- Swap state:

## Builds

- MemVanta commit SHA:
- Compiler and version:
- Build flags/type:
- Comparison runtime and commit SHA:
- Comparison runtime build flags/type:

## Model

- Model name:
- GGUF filename:
- Quantization:
- File size:
- SHA-256:

## Workload

- Threads:
- Context size:
- Batch size:
- KV-cache format:
- Prompt tokens:
- Generated tokens:
- Warm-up runs:
- Measured runs:

## Peak-RSS results

Include every measured run and the aggregate statistics. Attach or link raw output when possible.

## Constrained-memory results

If tested, list each cgroup-v2 `MemoryMax` ceiling and whether MemVanta and the comparison runtime completed or were OOM-killed. Confirm whether swap was disabled.

## Throughput

Report prompt-processing and token-generation throughput for transparency even when memory, rather than speed, is the focus.

## Commands and raw evidence

Paste exact commands or provide a repository/link containing the raw logs and machine-readable summary.

## Deviations from the reference protocol

Describe anything that differs from `docs/MEMORY_BENCHMARKING.md`.
