# Contributing to MemVanta

Thanks for contributing to MemVanta.

## Development setup

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Pull requests

- Keep changes focused and explain the motivation.
- Add or update tests for behavioral changes.
- For performance changes, include reproducible benchmark commands and raw results.
- Do not present synthetic-model results as trained-model results.
- Do not claim superiority over another runtime unless the comparison uses the same model, hardware, thread settings, context, and benchmark protocol.
- Preserve model-license and third-party attribution requirements.

## Benchmark evidence

For performance-related PRs, please report:

- CPU and memory configuration
- compiler and build type
- model name, exact file hash, quantization, and context size
- thread count and batch settings
- repetitions and warm-up policy
- prefill throughput, decode throughput, and peak RSS where applicable

Raw benchmark output is preferred over screenshots.

## Coding style

MemVanta targets modern C++20. Prefer clear ownership, bounded memory use, explicit error handling, and deterministic tests. Avoid unnecessary dependencies in the core runtime.
